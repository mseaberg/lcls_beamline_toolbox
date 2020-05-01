from psana import *
from timeit import default_timer as timer
from scipy.integrate import cumtrapz
import numpy as np
import argparse
import ConfigParser
import h5py
import scipy.ndimage.interpolation as interpolate
import matplotlib.pyplot as plt
from Talbot_functions_beta import *
import psana_utility
from legendre_qr3 import *
from skimage.restoration import unwrap_phase
from wfs_utils import *
from beam import *
from pitch import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
assert size>1, 'At least 2 MPI ranks required'

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("run",help="run number", type=str)
parser.add_argument("-z","--z0",help="distance to detector at nav_z=0",default=-1, type=float)
parser.add_argument("-s","--scan",help="whether the run is scan or not (1 or 0)", default=0, type=bool)
parser.add_argument("-r","--roi",help="half width of ROI (square)",default=-1,type=int)
parser.add_argument("-e","--energy",help="photon energy",default=-1,type=float)
parser.add_argument("-d","--down",help="downsampling (power of 2)",default=-1,type=int)
parser.add_argument("-o","--order",help="Zernike order",default=-1,type=int)
parser.add_argument("-c","--config",help="config file",default='wfs',type=str)
parser.add_argument("-t","--thresh",help="image threshold",default=-1,type=float)
parser.add_argument("-n","--num",help="number of events to process",default=-1,type=int)
parser.add_argument("-a","--delta",help="number of events to skip", default=1,type=int)
args = parser.parse_args()

# parse configuration file. command line arguments supersede config file
config = ConfigParser.ConfigParser()
config.read('config/'+args.config+'.cfg')
exp_name = config.get('Main','exp_name')
hutch = config.get('Main','hutch')
if args.energy>0:
    energy = args.energy
else:
    energy = config.getfloat('Main','energy')
if args.z0 > 0:
    z0 = args.z0
else:
    z0 = config.getfloat('Setup','z0')
if args.thresh > 0:
    thresh = args.thresh
else:
    thresh = config.getint('Processing','thresh')
if args.down > 0:
    down = args.down
else:
    down = config.getint('Processing','downsample')
if args.order > 0:
    order = args.order
else:
    order = config.getint('Processing','order')
pixel = config.getfloat('Setup','pixel')
det_z = config.get('Setup','det_z')
grating_z = config.get('Setup','grating_z')
pitch = config.getfloat('Setup','pitch')
detName = config.get('Setup','detName')
angle = config.getfloat('Setup','angle')
zf0 = config.getfloat('Setup','zf')
lineout_width = config.getint('Processing','lineout_width')
xmin = config.getint('Processing','xmin')
xmax = config.getint('Processing','xmax')
ymin = config.getint('Processing','ymin')
ymax = config.getint('Processing','ymax')
padding = config.getint('Processing','padding')


Nx_roi = xmax-xmin
Ny_roi = ymax-ymin

# pick maximum of these two so that we have a square image
if Nx_roi>Ny_roi:
    Ny_roi = Nx_roi
    ymax = ymin + Ny_roi
else:
    Nx_roi = Ny_roi
    xmax = xmin + Nx_roi

N_roi = Nx_roi

#N_roi = 2048

# put all arguments in a dictionary to write to the file
arguments = {}
arguments['hutch'] = hutch
arguments['exp_name'] = exp_name
arguments['run'] = args.run
arguments['energy'] = energy
arguments['z0'] = z0
arguments['down'] = down
arguments['order'] = order
arguments['pixel'] = pixel
arguments['pitch'] = pitch
arguments['scan'] = args.scan
arguments['angle'] = angle

eventNum = 30000

if args.num >0:
    eventNum = args.num/args.delta
    arguments['N'] = eventNum
elif args.delta>1:
    eventNum = eventNum/args.delta
    arguments['N'] = eventNum
else:
    arguments['N'] = eventNum

# open temporary DataSource to read out some scan information
ds0 = DataSource('exp='+exp_name+':run='+args.run+':smd')

evt0 = ds0.events().next()

epics0 = ds0.env().epicsStore()



# open DataSource for processing
ds = DataSource('exp='+exp_name+':run='+args.run+':smd')

# access to epics variables
epics = ds.env().epicsStore()


# calculate down-sampled center of mass
downsample = 2**down
#x_avg = (x_avg - 256)/downsample
#y_avg = (y_avg-56)/downsample

# array sizes, before and after downsampling (we are cropping out the center 2048x2048 pixels)
Ndown = downsample
#N = 2048
# expand field of view by factor sqrt(2) so that the whole image fits in the unit circle

Nfit = np.ceil(N_roi/Ndown)
if np.mod(Nfit,2) == 1:
    Nfit += 1
Nfit = int(Nfit)
Npad = Nfit-int(N_roi/Ndown)

print(Nfit)
print(Npad)

fit1 = LegendreFit(Nfit,Nfit,order)

# pixel size (m) on Zyla
dx = pixel

# grating pitch (m)
dg = pitch

# wavelength (m)
lambda0 = 1239.8/(energy*1000)*1e-9

# initialize processed information
dataSummary = {}



# center of mass
dataSummary['x_COMp'] = np.zeros(int(eventNum/size)+1)
dataSummary['y_COMp'] = np.zeros(int(eventNum/size)+1)
# detector z position
dataSummary['zD'] = np.zeros(int(eventNum/size)+1)
# grating to focus
dataSummary['zf'] = np.zeros(int(eventNum/size)+1)
# grating z position
dataSummary['zG'] = np.zeros(int(eventNum/size)+1)
# talbot distance (z0+zyla_z-zG)
dataSummary['zT'] = np.zeros(int(eventNum/size)+1)
# summed image (intensity)
dataSummary['I0'] = np.zeros(int(eventNum/size)+1)
# average horizontal phase
dataSummary['h_mean'] = np.zeros(int(eventNum/size)+1)
# average vertical phase
dataSummary['v_mean'] = np.zeros(int(eventNum/size)+1)

dataSummary['gdet1'] = np.zeros(int(eventNum/size)+1)
dataSummary['gdet2'] = np.zeros(int(eventNum/size)+1)

# ipm values
if hutch == 'xpp':
    dataSummary['ipm2'] = np.zeros(int(eventNum/size)+1)
    dataSummary['ipm3'] = np.zeros(int(eventNum/size)+1)
elif hutch == 'xcs':
    dataSummary['ipm5'] = np.zeros(int(eventNum/size)+1)
elif hutch == 'cxi':
    dataSummary['kb_vp'] = np.zeros(int(eventNum/size)+1)
    dataSummary['kb_hp'] = np.zeros(int(eventNum/size)+1)
    dataSummary['kb_hl'] = np.zeros(int(eventNum/size)+1)
# ebeam information
dataSummary['L3E'] = np.zeros(int(eventNum/size)+1)
dataSummary['BC2'] = np.zeros(int(eventNum/size)+1)
# second order coefficient to add back to zernike coeffs because of linear phase removal
dataSummary['p0'] = np.zeros(int(eventNum/size)+1)
dataSummary['p0x'] = np.zeros(int(eventNum/size)+1)
dataSummary['p0y'] = np.zeros(int(eventNum/size)+1)

# calculated pitch
dataSummary['x_pitch'] = np.zeros(int(eventNum/size)+1)
dataSummary['y_pitch'] = np.zeros(int(eventNum/size)+1)
dataSummary['x_res'] = np.zeros((int(eventNum/size)+1,N_roi))
dataSummary['y_res'] = np.zeros((int(eventNum/size)+1,N_roi))
dataSummary['x_prime'] = np.zeros((int(eventNum/size)+1,N_roi))
dataSummary['y_prime'] = np.zeros((int(eventNum/size)+1,N_roi))
dataSummary['x_rms'] = np.zeros(int(eventNum/size)+1)
dataSummary['y_rms'] = np.zeros(int(eventNum/size)+1)

# access to ebeam information
eBeam = Detector('EBeam')
gasDet = Detector('FEEGasDetEnergy')
# access to zyla images
zyla = Detector(detName)



# zernike coefficients (fit)
dataSummary['zernike_coeff'] = np.zeros((int(eventNum/size)+1,fit1.P))

# event numbers for sorting
dataSummary['eventNums'] = np.ones(int(eventNum/size)+1)*-1

# global iterator
n0 = -1
# local iterator (per rank)
n = -1

# downsampled pixel size
dx2 = dx*Ndown



# downsampled coordinates (real space)
x2 = np.linspace(-Nfit/2,Nfit/2-1,Nfit)
x2,y2 = np.meshgrid(x2,x2)

# spatial frequencies
fxMax = 1.0/(2.0*dx2)
dfx = fxMax / (Nfit)
fx = np.linspace(-fxMax,fxMax - dfx, Nfit)
fx, fy = np.meshgrid(fx,fx)

# recovered beam and focus (2d downsampled images)
dataSummary['recoveredBeam'] = np.zeros((int(eventNum/size)+1,Nfit,Nfit)).astype(complex)
dataSummary['recoveredFocus'] = np.zeros((int(eventNum/size)+1,Nfit,Nfit)).astype(complex)
dataSummary['h_grad'] = np.zeros((int(eventNum/size)+1,Nfit,Nfit)).astype(float)
dataSummary['v_grad'] = np.zeros((int(eventNum/size)+1,Nfit,Nfit)).astype(float)

print('looping through events')

# loop through events
for nevent,evt in enumerate(ds.events()):

    # check if we are at the last event to process
    if nevent>eventNum*args.delta-1: break

    # only take an event every once in a while
    if np.mod(nevent,args.delta)!=0: continue
    
    # update global iterator
    n0 += 1

    # skip over events that belong to other ranks
    if n0%size!=rank: continue


    # update progress to logfile
    if (n0)%10 == 0:
        print(nevent)

    # check for damage
    if zyla.image(evt) is None: continue

    # update local iterator
    n += 1

    # get z positions
    if epics.value(det_z) is None:
        zD = 0.0
    else:
        zD = epics.value(det_z)*1.e-3
    zG = epics.value(grating_z)*1.e-3
    # distance from grating to focus
    zf = zf0 + zG
    # distance from grating to detector
    zT = z0 + zD - zG

    # check for damage
    if zD is None: continue
    dataSummary['zD'][int(n)] = zD
    dataSummary['zG'][n] = zG
    dataSummary['zf'][n] = zf
    dataSummary['zT'][n] = zT

    # get ebeam parameters
    if eBeam.get(evt) is not None:
        dataSummary['BC2'][n] = eBeam.get(evt).ebeamPkCurrBC2()
        dataSummary['L3E'][n] = eBeam.get(evt).ebeamL3Energy()

    # get pulse energy
    if gasDet.get(evt) is not None:
        dataSummary['gdet1'][n] = gasDet.get(evt).f_11_ENRC()
        dataSummary['gdet2'][n] = gasDet.get(evt).f_12_ENRC()

    # get ipm data
    if hutch == 'xpp':
        ipm2 = evt.get(Lusi.IpmFexV1, Source('XppSb2_Ipm'))
        ipm3 = evt.get(Lusi.IpmFexV1, Source('XppSb3_Ipm'))
        if ipm2 is not None:
            dataSummary['ipm2'][n] = ipm2.sum()
        if ipm3 is not None:
            dataSummary['ipm3'][n] = ipm3.sum()
    elif hutch == 'xcs':
        ipm5 = evt.get(Lusi.IpmFexV1, Source('XCS-IPM-05'))
        if ipm5 is not None:
            dataSummary['ipm5'][n] = ipm5.sum()
    elif hutch == 'cxi':
        if epics.value('kb2_vp') is not None:
            dataSummary['kb_vp'][n] = epics.value('kb2_vp')
        if epics.value('kb2_hp') is not None:
            dataSummary['kb_hp'][n] = epics.value('kb2_hp')
        if epics.value('kb2_hl') is not None:
            dataSummary['kb_hl'][n] = epics.value('kb2_hl')

    # get image from zyla and subtract dark image
    im1 = zyla.image(evt).astype(float)
    #im1 = (zyla.raw_data(evt)-zyla.pedestals(evt)).astype(float)
    im1[im1<10] = 0
    im1 = interpolate.rotate(im1,angle,reshape=False)

    # calculate summed intensity
    dataSummary['I0'][n] = np.sum(im1)

    # calculate step number


    im2 = im1[ymin:ymax,xmin:xmax]
    im1 = im1[ymin:ymax,xmin:xmax]
    Nl,Ml = np.shape(im2)
   
    lineout_x = np.sum(im2[int(Nl/2-lineout_width/2):int(Nl/2+lineout_width/2),:],axis=0)
    lineout_y = np.sum(im2[:,int(Ml/2-lineout_width/2):int(Ml/2+lineout_width/2)],axis=1)

    mag = (zT + zf) / zf

    peak = 1./mag/dg


    fc = peak*dx

    x_pitch, x_res, x_prime = calc_pitch(lineout_x,fc,1)
    y_pitch, y_res, y_prime = calc_pitch(lineout_y,fc,1)


    mag_x = x_pitch*dx/dg
    mag_y = y_pitch*dx/dg

    zfd_x = zT*mag_x/(mag_x-1.)
    zfd_y = zT*mag_y/(mag_y-1.)

    zf_x = -(zT*mag_x/(mag_x-1.)-zT - zf)
    zf_y = -(zT*mag_y/(mag_y-1.)-zT-zf)

    dataSummary['x_pitch'][n] = zf_x
    dataSummary['y_pitch'][n] = zf_y

    dx_prime = x_prime[1]-x_prime[0]
    dy_prime = y_prime[1]-y_prime[0]
    
    x_res = np.cumsum(x_res)*dg/lambda0/zT*dx_prime*dx
    y_res = np.cumsum(y_res)*dg/lambda0/zT*dy_prime*dx

    x_prime = x_prime*dx*1e6
    y_prime = y_prime*dx*1e6

    xN = np.size(x_res)
    yN = np.size(y_res)

    dataSummary['x_res'][n,0:xN] = x_res
    dataSummary['y_res'][n,0:yN] = y_res
    dataSummary['x_prime'][n,0:xN] = x_prime
    dataSummary['y_prime'][n,0:yN] = y_prime

    start = timer()

    # calculate gradients based on Talbot image
    h_grad, v_grad, params = calc_gradients(im1, dx, dg, lambda0, peak, zT, downsample=down)

    end = timer()
    print('%.2f seconds for Fourier processing' % (end-start))
    
    # define "zero order" based on magnitude of gradients
    zero_order = (np.abs(h_grad) + np.abs(v_grad)) / 2.0

    zo = beam_threshold(zero_order,0.04)

    zo[zo>0] = 1
    Ni,Mi = np.shape(zo)

    xp = params['x1']
    yp = params['y1']

    x_COM = np.sum(xp*zo)/np.sum(zo)
    y_COM = np.sum(yp*zo)/np.sum(zo)
    x_COMp = x_COM/dx2+Mi/2
    y_COMp = y_COM/dx2+Ni/2
    

    # threshold above noise
    zeroMask = zero_order>(thresh*downsample)
    #zeroMask = np.logical_and(zeroMask, x2>-36)
    #zeroMask = np.logical_and(zeroMask, x2<80/2)
    
    # unwrap phase in 2D, multiply by shear factor
    h_grad2 = unwrap_phase(np.angle(h_grad), seed=0) * dg / lambda0 / zT
    v_grad2 = unwrap_phase(np.angle(v_grad), seed=0) * dg / lambda0 / zT

    N2, M2 = np.shape(h_grad2)
    zeroMask = np.pad(zeroMask, ((Npad,0),(Npad,0)), 'constant')
    h_grad2 = np.pad(h_grad2, ((Npad,0),(Npad,0)),'constant')
    v_grad2 = np.pad(v_grad2, ((Npad,0),(Npad,0)),'constant')
    zero_order = np.pad(zero_order, ((Npad,0),(Npad,0)),'constant')

    h_mean = np.mean(h_grad2[zeroMask])
    v_mean = np.mean(v_grad2[zeroMask])

    dataSummary['h_mean'][n] = h_mean
    dataSummary['v_mean'][n] = v_mean

    h_grad2 -= h_mean
    v_grad2 -= v_mean
    #h_grad2 -= 165500
    #v_grad2 += 117000

    dataSummary['h_grad'][n,:,:] = h_grad2*zeroMask
    dataSummary['v_grad'][n,:,:] = v_grad2*zeroMask

    p0 = params['p0']
    dataSummary['p0'][n] = p0
    # fit gradient using zernike polynomials (making sure there is intensity above the threshold)
    if np.sum(zeroMask)>0:
    
        # fit Zernike coefficients by projecting gradients onto orthonormal basis
        zernikeTemp = fit1.z_coeff_grad(h_grad2, v_grad2, dx2,zeroMask).flatten()
        # put into the data dictionary
        dataSummary['zernike_coeff'][n,:] = zernikeTemp
    else:
        # just set everything to zero if intensity is too low
        dataSummary['zernike_coeff'][n,:] = np.zeros(fit1.P)

    

    # reconstructed phase
    wave = fit1.wavefront_fit(dataSummary['zernike_coeff'][n,:])

    # recovered beam
    recovered = np.exp(1j * wave) * np.sqrt(zero_order)*zeroMask

    dataSummary['p0x'][n] = params['p0x']+np.pi/lambda0/zT
    dataSummary['p0y'][n] = params['p0y']+np.pi/lambda0/zT

    #px = params['p0x'] - p0
    #py = params['p0y'] - p0

    px = params['p0x'] + np.pi/lambda0/zT
    py = params['p0y'] + np.pi/lambda0/zT

    p0 = (px+py)/2.

    # correct defocus
    defocus_phase = np.exp(1j*(xp**2*(px-p0) + yp**2*(py-p0)))

    recovered *= defocus_phase

    # distance to focal plane (may be offset from true focus if Fourier peak isn't centered)
    f0 = np.pi/lambda0/p0


    # coordinates at focus
    xf = fx * lambda0 * f0
    yf = fy * lambda0 * f0

    dxf = xf[0,1]-xf[0,0]
    print('pixel size: ' +str(dxf))

    # phase to multiply by at focus
    phase2 = np.exp(-1j * np.pi / lambda0 / f0 * (xf ** 2 + yf ** 2))

    # calculate beam at focus
    focus = Beam.INFFT(recovered) * phase2
    print(str(end-start)+' seconds')
    # populate recovered beam into dataSummary
    dataSummary['recoveredBeam'][n,:,:] = recovered
    # focus can also be calculated later using recoveredBeam
    dataSummary['recoveredFocus'][n,:,:] = focus

    # populate event number for sorting
    dataSummary['eventNums'][n] = nevent

fileName = '/reg/d/psdm/'+hutch+'/'+hutch+exp_name+'/results/wfs/Run'+args.run+'_focus_legendre_beta.h5'

psana_utility.save_all(comm,rank,dataSummary,arguments,fileName)

MPI.Finalize()
