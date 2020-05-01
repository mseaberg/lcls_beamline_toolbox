import sys
#sys.path.append('/reg/neh/home/seaberg/wfs_beta/')
from Talbot_functions_beta import *
from beam import *
from psana import *
import numpy as np
from mpidata import mpidata 
from psmon import publish
import psmon.plots as psplt
from psmon.plots import XYPlot,Image,Hist
import h5py
import scipy.ndimage.interpolation as interpolate
import pandas
from pitch import *
from legendre_qr3 import *
from skimage.restoration import unwrap_phase
import scipy.optimize as optimization
from wfs_utils import *
from timeit import default_timer as timer

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def runclient(args,pars):

    sh_mem = pars['live']
    thresh = pars['thresh']
    expName = pars['exp_name']
    hutchName = pars['hutch']
    runNum = args.run
    detName = pars['detName']
    lineout_width = pars['lineout_width']
    update = pars['update_events']
    runString = 'exp=%s:run=%s:smd' % (expName, runNum)
    #runString += ':dir=/reg/d/ffb/%s/%s/xtc:live' % (hutchName,expName)
    print(runString)
    downsample=pars['downsample']

    roi = pars['roi']
    padSize = pars['pad']-1
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]

    Ndown = 2**downsample
    N = 2048
    # expand field of view by factor sqrt(2) so that the whole image fits in the unit circle
    Nfit_x, Nfit_y, Npad_x, Npad_y = image_padding_legendre(Ndown, (0,2048,0,2048))

    #Npad_x = int((N-Nfit_x)/2)
    #Npad_y = int((N-Nfit_y)/2)

    lambda0 = 1239.8/pars['energy']/1000.*1.e-9
    shift = lambda0*(1.59-.494)/10.e-6/(Ndown*1.625e-6)

    start = timer()
    #fit1 = ZernikeFit(Nfit,Nfit,32)
    #fit1 = LegendreFit(Nfit_y,Nfit_x,6,shift)
    fit1 = LegendreFit(Nfit_y,Nfit_x,12)
    stop = timer()

    Npad2_x = int((N/Ndown-Nfit_x)/2)
    Npad2_y = int((N/Ndown-Nfit_y)/2)

    print(str(stop-start)+' seconds to generate basis')
    zernikeTemp = np.zeros(fit1.P)

    # miscellaneous parameters
    dx = pars['pixel']
    dg = pars['pitch']
    dx2 = dx*Ndown

    calibDir = '/reg/d/psdm/%s/%s/calib' % (hutchName, expName)
    
    ds = []
    if sh_mem:
        
        setOption('psana.calib-dir', calibDir)
        ds = DataSource('shmem=psana.0:stop=no')
        
    else:
        ds = DataSource(runString)

    det0 = Detector(detName)
    epics = ds.env().epicsStore()
        
    gasDet = Detector('FEEGasDetEnergy')

    nevents = np.empty(0)

    # initialize instance of the mpidata class for communication with the master process
#    md = mpidata()

    i1 = -1
    i0 = -1
   
    md = mpidata()

    dfx = 1./(N/Ndown*dx2)
    fx = np.linspace(-N/2/Ndown,N/2/Ndown-1,N/Ndown,dtype=float)*dfx
    fx, fy = np.meshgrid(fx,fx)

    N2 = N/Ndown

    xd = np.linspace(-N2/2,N2/2-1,N2,dtype=float)*dx2
    xd,yd = np.meshgrid(xd,xd)

    # event loop
    for nevent,evt in enumerate(ds.events()):


        # check if we've reached the event limit        
        if nevent == args.noe : break
        if nevent%(size-1)!=rank-1: continue # different ranks look at different events

        #if det0.image(evt) is None: continue
        # increment counter
        i1 += 1
        i0 += 1
        

        nevents = np.append(nevents,nevent)
        
        
        if (i1==update): # send mpi data object to master when desired
            md=mpidata()
            print(nevent)
            if det0.image(evt) is None: 
                print('no image')
                continue

            pulseEnergy = gasDet.get(evt).f_11_ENRC()

            # select image ROI
            img0 = np.copy(det0.image(evt).astype(float))
            #img0[img0<20] = 0
            # check if the image needs to be rotated
            img0 = interpolate.rotate(img0,pars['angle'],reshape=False)

            

            img0 = img0[ymin:ymax,xmin:xmax]

            img1 = np.copy(img0)

            if np.sum(img0)<0.3e7:continue

            N,M = np.shape(img0)
            print(N)
            print(M)
            #img0 = np.pad(img0,((int(N/2*padSize),int(N/2*padSize)),
            #                    (int(M/2*padSize),int(M/2*padSize))),'constant')
            #### find peaks ####

            # calculate Talbot magnification
            zD = epics.value(pars['det_z'])
            if zD is None:
                zD = 0.0
            else:
                zD *= 1.e-3
            zG = epics.value(pars['grating_z'])*1.e-3
            # distance from grating to focus
            zf = pars['zf'] + zG
            # distance from grating to detector
            zT = pars['z0'] + zD - zG
            # magnification
            mag = (zT + zf) / zf
            #mag = 1.
            peak = 1./mag/dg
           
            start = timer()
            h_grad, v_grad, params = calc_gradients(img1,dx,dg,lambda0,peak,zT,downsample=downsample)
            stop = timer()

            print(str(stop-start)+' seconds for Fourier processing')
            zero_order = (np.abs(h_grad)+np.abs(v_grad))/2.0
            h_grad2 = unwrap_phase(np.angle(h_grad), seed=0) * dg/lambda0/zT
            v_grad2 = unwrap_phase(np.angle(v_grad), seed=0) * dg/lambda0/zT

            N2,M2 = np.shape(h_grad2)

            #h_grad2 = h_grad2[N2/2-Nfit_y/2:N2/2+Nfit_y/2,M2/2-Nfit_x/2:M2/2+Nfit_x/2]
            #v_grad2 = v_grad2[N2/2-Nfit_y/2:N2/2+Nfit_y/2,M2/2-Nfit_x/2:M2/2+Nfit_x/2]
            #zero_order = zero_order[N2/2-Nfit_y/2:N2/2+Nfit_y/2,M2/2-Nfit_x/2:M2/2+Nfit_x/2]


           # if N2>Nfit_y:
           #     h_grad2 = h_grad2[:Nfit_y,:]
           #     v_grad2 = v_grad2[:Nfit_y,:]
           #     zero_order = zero_order[:Nfit_y,:]
           # elif N2<Nfit_y:
           #     h_grad2 = np.pad(h_grad2,((0,Nfit_y-N2),(0,0)),'constant')
           #     v_grad2 = np.pad(v_grad2,((0,Nfit_y-N2),(0,0)),'constant')
           #     zero_order = np.pad(zero_order,((0,Nfit_y-N2),(0,0)),'constant')
           # 
           # if M2>Nfit_x:
           #     h_grad2 = h_grad2[:,:Nfit_x]
           #     v_grad2 = v_grad2[:,:Nfit_x]
           #     zero_order = zero_order[:,:Nfit_x]
           # elif M2<Nfit_x:
           #     h_grad2 = np.pad(h_grad2,((0,0),(0,Nfit_x-M2)),'constant')
           #     v_grad2 = np.pad(v_grad2,((0,0),(0,Nfit_x-M2)),'constant')
           #     zero_order = np.pad(zero_order,((0,0),(0,Nfit_x-M2)),'constant')
           #



            zeroMask = zero_order>thresh*Ndown

            print(np.shape(h_grad2))
            print(Npad_y)
            print(Npad_x)
            N2,M2 = np.shape(h_grad2)
            x2 = np.linspace(-M2/2,M2/2-1,M2)
            y2 = np.linspace(-N2/2,N2/2-1,N2)
            x2,y2 = np.meshgrid(x2,y2)

            #h_grad2 = h_grad2[ymin:ymax,xmin:xmax]
            #v_grad2 = v_grad2[ymin:ymax,xmin:xmax]
            #zeroMask = zeroMask[ymin:ymax,xmin:xmax]
            #zero_order = zero_order[ymin:ymax,xmin:xmax]

            #zeroMask = np.logical_and(zeroMask,x2<40)
            #zeroMask = np.logical_and(zeroMask,x2>-36)

            if np.sum(zeroMask) < 1: continue

            x_c = np.sum(x2*zeroMask)/np.sum(zeroMask)
            y_c = np.sum(y2*zeroMask)/np.sum(zeroMask)

            zeroMask = np.roll(zeroMask,-int(y_c),axis=0)
            zeroMask = np.roll(zeroMask,-int(x_c),axis=1)
            #zeroMask = np.pad(np.roll(zeroMask,-int(x_c),axis=1),((Npad,Npad),(Npad,Npad)),'constant')
            #zeroMask = np.pad(zeroMask,((Npad_y,Npad_y),(Npad_x,Npad_x)),'constant')
            zero_order = np.roll(zero_order,-int(y_c),axis=0)
            zero_order = np.roll(zero_order,-int(x_c),axis=1)
            #zero_order = np.pad(np.roll(zero_order,-int(x_c),axis=1),((Npad,Npad),(Npad,Npad)),'constant')
            #zero_order = np.pad(zero_order,((Npad_y,Npad_y),(Npad_x,Npad_x)),'constant')
            h_grad2 = np.roll(h_grad2,-int(y_c),axis=0)
            h_grad2 = np.roll(h_grad2,-int(x_c),axis=1)
            #h_grad2 = np.pad(np.roll(h_grad2,-int(x_c),axis=1),((Npad,Npad),(Npad,Npad)),'constant')
            #h_grad2 = np.pad(h_grad2,((Npad_y,Npad_y),(Npad_x,Npad_x)),'constant')

            v_grad2 = np.roll(v_grad2,-int(y_c),axis=0)
            v_grad2 = np.roll(v_grad2,-int(x_c),axis=1)
            #v_grad2 = np.pad(np.roll(v_grad2,-int(x_c),axis=1),((Npad,Npad),(Npad,Npad)),'constant')
            #v_grad2 = np.pad(v_grad2,((Npad_y,Npad_y),(Npad_x,Npad_x)),'constant')

            h_grad2 -= np.mean(h_grad2[zeroMask])
            v_grad2 -= np.mean(v_grad2[zeroMask])
           

            #h_grad2 = h_grad2[:Nfit_y,:Nfit_x]
            #v_grad2 = v_grad2[:Nfit_y,:Nfit_x]
            #zeroMask = zeroMask[:Nfit_y,:Nfit_x]
            #zero_order = zero_order[:Nfit_y,:Nfit_x]

            fc = peak*dx
            
            lineout_x = np.sum(img0[int(N/2-lineout_width/2):int(N/2+lineout_width/2),:],axis=0)
            lineout_y = np.sum(img0[:,int(M/2-lineout_width/2):int(M/2+lineout_width/2)],axis=1)

            x_pitch,x_res,x_prime = calc_pitch(lineout_x,fc,1)
            y_pitch,y_res,y_prime = calc_pitch(lineout_y,fc,1)

            mag_x = x_pitch*dx/dg
            mag_y = y_pitch*dx/dg

            zfd_x = zT*mag_x/(mag_x-1.)
            zfd_y = zT*mag_y/(mag_y-1.)

            zfd_mid = (zfd_x+zfd_y)/2.


            xf = fx*lambda0*zfd_mid
            yf = fy*lambda0*zfd_mid

            Nf, Mf = np.shape(xf)        
            dxf = xf[0,1]-xf[0,0]
            dxf2 = dxf*Nf/512

            f0 = np.pi/lambda0/params['p0']
            phase1 = np.exp(-1j*np.pi/lambda0*(1./zfd_y - 1./f0)*(xd**2+yd**2))
            phase2 = np.exp(-1j*np.pi/lambda0/zfd_mid*(xf**2+yf**2))
            

            # 2d fit
            zernikeTemp = np.zeros(fit1.P)
            
            if np.sum(zeroMask)>0:
                start = timer()
                zernikeTemp = fit1.z_coeff_grad(h_grad2,v_grad2,dx2,zeroMask).flatten()
                stop = timer()
                print(str(stop-start)+' seconds for projection')
                print(str(np.size(zernikeTemp)) + ' Zernike terms')
            #zernikeTemp[0:2] = 0
            #zernikeTemp[3] = 0

            wave = fit1.wavefront_fit(zernikeTemp)

            #wave = np.pad(wave,((Npad2_y,Npad2_y),(Npad2_x,Npad2_x)),'constant')
            #zero_order = np.pad(zero_order,((Npad2_y,Npad2_y),(Npad2_x,Npad2_x)),'constant')
            #zeroMask = np.pad(zeroMask,((Npad2_y,Npad2_y),(Npad2_x,Npad2_x)),'constant')

            recovered = np.exp(1j*wave)*np.sqrt(zero_order)
            #recovered = np.sqrt(zero_order)
            recovered *= zeroMask

            px = params['p0x'] + np.pi/lambda0/zT
            py = params['p0y'] + np.pi/lambda0/zT

            p0 = (px+py)/2.
           
            xp = params['x1']
            yp = params['y1']

            #correct defocus
            defocus_phase = np.exp(1j*(xp**2*(px-p0) + yp**2*(py-p0)))

            recovered *= defocus_phase

            wave += xp**2*(px-p0) + yp**2*(py-p0)
            phase22 = np.zeros((512,512),dtype=complex)
            #phaseF = Beam.NFFT(np.pad(phase2,((Npad2_y,Npad2_y),(Npad2_x,Npad2_x)),'constant'))
            phaseF = Beam.NFFT(phase2)
            phase22[256-2048/Ndown/2:256+2048/Ndown/2,256-2048/Ndown/2:256+2048/Ndown/2] = phaseF
            phase2 = Beam.INFFT(phase22)

            recovered2 = np.zeros((512,512),dtype=complex)
            recovered2[256-2048/Ndown/2:256+2048/Ndown/2,256-2048/Ndown/2:256+2048/Ndown/2] = recovered

            x3 = np.linspace(-256,255,512)
            x3,y3 = np.meshgrid(x3,x3)

            #def my_func(coeff):
            #    quad1 = np.exp(1j*(x3**2+y3**2)*coeff[0])
            #    return np.sum(np.angle(recovered2*quad1))

            #res = optimization.minimize(my_func,[0.0])
            #print(res.x)

            print('pixel size: '+str(dxf2))
            focus = (Beam.INFFT(recovered2)*phase2)

            focus = focus[256-148:256+148,256-148:256+148]
            b1 = Beam(focus,dxf2,lambda0)

            xff = np.linspace(-2048/Ndown/2,2048/Ndown/2-1,296)
            xff,yff = np.meshgrid(xff,xff)

            intensity = np.zeros(21)
            beam_area = np.zeros(21)
            b1.beam_prop(-500e-6)
            for i in range(21):
                intensity[i] = np.max(np.abs(b1.wave)**2)
#
                focus = np.abs(b1.wave)**2
                x_com = np.sum(focus*xff)/np.sum(focus)
                y_com = np.sum(focus*yff)/np.sum(focus)
                x_moment = np.sqrt(np.sum(focus*(xff-x_com)**2)/np.sum(focus))
                y_moment = np.sqrt(np.sum(focus*(yff-y_com)**2)/np.sum(focus))
                beam_area[i] = x_moment*y_moment

                b1.beam_prop(50e-6)

            i1 = np.argmax(intensity)

            b1.beam_prop(-1000e-6+i1*50e-6)
            focus = np.abs(focus)**2
            focus = np.abs(b1.wave)**2
            focus = focus/np.sum(focus)*1000.
            norm = np.sum(focus)
            intensity = np.max(focus)/norm*3.0e-3/dxf2**2/60.e-15/100.**2            


            zf_x = -(zT*mag_x/(mag_x-1.) - zT - zf)*1e3
            zf_y = -(zT*mag_y/(mag_y-1.) - zT - zf)*1e3


            h_peak = 1./x_pitch
            v_peak = 1./y_pitch
            F0 = np.abs(Beam.NFFT(img0))

            dx_prime = x_prime[1]-x_prime[0]
            dy_prime = y_prime[1]-y_prime[0]

            x_grad = np.copy(x_res)
            y_grad = np.copy(y_res)

            x_res = np.cumsum(x_res)*dg/lambda0/zT*dx_prime*dx
            y_res = np.cumsum(y_res)*dg/lambda0/zT*dy_prime*dx
            x_prime = x_prime*dx*1e6
            y_prime = y_prime*dx*1e6

            h_width = np.std(x_res)
            v_width = np.std(y_res)

            print(np.sum(np.angle(h_grad)))

            F0 = np.abs(Beam.NFFT(img0))
            # normalize to maximum
            F0 = F0/np.max(F0)
            # normalize to maximum
            #img0 = img0/np.max(img0)
            md.addarray('F0',F0)
            md.addarray('img0',wave*zeroMask)
            md.addarray('x_res',x_res)
            md.addarray('y_res',y_res)
            md.addarray('x_prime',x_prime)
            md.addarray('y_prime',y_prime)
            #md.addarray('x_grad',x_grad)
            #md.addarray('y_grad',y_grad)
            md.addarray('focus',focus) 
            md.addarray('nevents',nevents[-1])
            md.addarray('h_peak',zf_x)
            md.addarray('v_peak',zf_y)
            md.addarray('h_width',h_width)
            md.addarray('v_width',v_width)
            md.addarray('focus_distance',f0)
            md.addarray('intensity',intensity)
            md.addarray('amp',zero_order)
            #md.addarray('v_grad',np.angle(v_grad))
            md.small.event = nevent
            
            md.send()

            nevents = np.empty(0)
            i1 = 0

           # 
    md.endrun()	


def runmaster(nClients,pars,args):




    # initialize arrays

    dataDict = {}
    dataDict['nevents'] = np.ones(10000)*-1
    dataDict['h_peak'] = np.zeros(10000)
    dataDict['v_peak'] = np.zeros(10000)
    dataDict['h_width'] = np.zeros(10000)
    dataDict['v_width'] = np.zeros(10000)
    dataDict['x_prime'] = np.zeros(10000)
    dataDict['x_res'] = np.zeros(10000)
    dataDict['y_prime'] = np.zeros(10000)
    dataDict['y_res'] = np.zeros(10000)
    dataDict['x_grad'] = np.zeros(10000)
    dataDict['y_grad'] = np.zeros(10000)
    dataDict['intensity'] = np.zeros(10000)

    roi = pars['roi']
    xmin = roi[0]
    xmax = roi[1]
    ymin = roi[2]
    ymax = roi[3]

    padSize = pars['pad']
    xSize = (xmax-xmin)*padSize
    ySize = (ymax-ymin)*padSize


    # initialize plots
    img_measured = Image(0,"RAW",np.zeros((ySize,xSize)))
    img_fft = Image(0,"FFT",np.zeros((ySize,xSize)))
    img_focus = Image(0,"FOCUS",np.zeros((ySize,xSize)))
    img_amp = Image(0,"AMP",np.zeros((ySize,xSize)))

#    peak_plot = XYPlot(0, "peak", [dataDict['nevents'],dataDict['nevents'],dataDict['nevents']],[dataDict['h_peak'],dataDict['v_peak'],dataDict['h_peak']], formats=['bo','ro','bo'],leg_label=['Horizontal','Vertical','smooth'])
    focus_dist_plot = XYPlot(0,"focus_dist",[dataDict['nevents'],dataDict['nevents'],dataDict['nevents'],dataDict['nevents']],[dataDict['h_peak'],dataDict['v_peak'],dataDict['h_peak'],dataDict['v_peak']],formats=['bo','ro','c.','m.'],leg_label=['Horizontal','Vertical','Horizontal Smooth','Vertical Smooth'])
    focus_dist_plot.xlabel = 'Event number'
    focus_dist_plot.ylabel = 'Focus position (mm)'

    x_grad_plot = XYPlot(0,"x_grad",dataDict['x_prime'],dataDict['x_grad'])
    x_grad_plot.xlabel = 'x (microns)'
    x_grad_plot.ylabel = 'Phase gradient'

    y_grad_plot = XYPlot(0,"y_grad",dataDict['y_prime'],dataDict['y_grad'])
    y_grad_plot.xlabel = 'y (microns)'
    y_grad_plot.ylabel = 'Phase gradient'

    x_res_plot = XYPlot(0,"x_res",dataDict['x_prime'],dataDict['x_res'])
    x_res_plot.xlabel = 'x (microns)'
    x_res_plot.ylabel = 'Residual phase (rad)'

    y_res_plot = XYPlot(0,"y_res",dataDict['y_prime'],dataDict['y_res'])
    y_res_plot.xlabel = 'y (microns)'
    y_res_plot.ylabel = 'Residual phase (rad)'

    rms_plot = XYPlot(0, "rms", [dataDict['nevents'],dataDict['nevents'],0,0], [dataDict['h_width'],dataDict['v_width'],0,0], formats=['bo','ro','c.','m.'],leg_label=['Horizontal','Vertical','Horizontal Smooth','Vertical Smooth'])
    rms_plot.xlabel = 'Event number'
    rms_plot.ylabel = 'RMS Residual Phase (rad)'


    intensity_plot = XYPlot(0,"intensity",dataDict['nevents'],dataDict['intensity'],formats='bo')
    intensity_plot.xlabel = 'Event number'
    intensity_plot.ylabel = 'Peak intensity (a.u.)'


    publish.send("peak",focus_dist_plot)
    publish.send("rms",rms_plot)
    publish.send("FFT",img_fft)
    publish.send("RAW",img_measured)
    publish.send("x_res",x_res_plot)
    publish.send("y_res",y_res_plot)
    #publish.send("x_grad",x_grad_plot)
    #publish.send("y_grad",y_grad_plot)
    publish.send("FOCUS",img_focus)
    publish.send("intensity",intensity_plot)
    #publish.send("v_grad",img_v_grad)
    publish.send("AMP",img_amp)

    nevent = -1

    while nClients > 0:
        # Remove client if the run ended
        md = mpidata()
        rank1 = md.recv()
        print(rank1)
        if md.small.endrun:
            nClients -= 1
        else:

            #nevents = np.append(nevents,md.nevents)
            dataDict['nevents'] = update(md.nevents,dataDict['nevents']) 
            dataDict['h_peak'] = update(md.h_peak,dataDict['h_peak'])
            dataDict['v_peak'] = update(md.v_peak,dataDict['v_peak'])
            dataDict['h_width'] = update(md.h_width,dataDict['h_width'])
            dataDict['v_width'] = update(md.v_width,dataDict['v_width'])
            dataDict['intensity'] = update(md.intensity,dataDict['intensity'])

            if md.nevents>nevent:
                nevent = md.nevents
                F0 = md.F0
                focus = md.focus
                img0 = md.img0
                x_res = md.x_res
                y_res = md.y_res
                focus_distance = md.focus_distance
                amp = md.amp
            #if len(nevents)>1000:

            #    nevents = nevents[len(nevents)-1000:]
            
            
            #counterSum += md.counter
            if rank1 == size-1:
            #if True:
                #plot(md,img_measured,img_fft,peak_plot,width_plot,dataDict)

                mask = dataDict['nevents']>0

                eventMask = dataDict['nevents'][mask]

                order = np.argsort(eventMask)
                eventMask = eventMask[order]

                h_peak = dataDict['h_peak'][mask][order]
                v_peak = dataDict['v_peak'][mask][order]
                h_width = dataDict['h_width'][mask][order]
                v_width = dataDict['v_width'][mask][order]
                intensity = dataDict['intensity'][mask][order]

                h_smooth = pandas.rolling_mean(h_peak,10)
                v_smooth = pandas.rolling_mean(v_peak,10)
                hw_smooth = pandas.rolling_mean(h_width,10)
                vw_smooth = pandas.rolling_mean(v_width,10)

                

                plot(focus_dist_plot,"focus_dist",[eventMask,eventMask,eventMask[10:],eventMask[10:]],[h_peak,v_peak,h_smooth[10:],v_smooth[10:]])
                plot(rms_plot,"rms",[eventMask,eventMask,eventMask[10:],eventMask[10:]],[h_width,v_width,hw_smooth[10:],vw_smooth[10:]])
                plot(intensity_plot,"intensity",eventMask,intensity)
                imshow(img_measured,"RAW",img0,nevent)
                imshow(img_fft,"FFT",F0,nevent)
                imshow(img_focus,"FOCUS",focus,focus_distance)
                #imshow(img_v_grad,"v_grad",md.v_grad,md.small.event)
                imshow(img_amp,"AMP",amp,nevent)
                plot(x_res_plot,"x_res",md.x_prime,md.x_res)
                plot(y_res_plot,"y_res",md.y_prime,md.y_res)
                #plot(x_grad_plot,"x_grad",md.x_prime,md.x_grad)
                #plot(y_grad_plot,"y_grad",md.y_prime,md.y_grad)

    fileName = '/reg/d/psdm/'+pars['hutch']+'/'+pars['hutch']+pars['exp_name']+'/results/wfs/'+args.run+'_2d_data.h5'

    mask = dataDict['nevents'] >= 0

    for key in dataDict.keys():
        dataDict[key] = dataDict[key][mask]

    i1 = np.argsort(dataDict['nevents'])
    for key in dataDict.keys():
        dataDict[key] = dataDict[key][i1]

    with h5py.File(fileName,'w') as f:
        for key in dataDict.keys():
            f.create_dataset(key, data=dataDict[key])
                
                
def update(newValue,currentArray):
    currentArray = np.roll(currentArray,-1)
    currentArray[-1] = newValue
    return currentArray

def plot(plot0,plotName,xdata,ydata):
    plot0.xdata = xdata
    plot0.ydata = ydata
    publish.send(plotName,plot0)

def imshow(im0,imName,img,eventNum):
    im0.image = img
    im0.ts = np.max(eventNum,0)
    publish.send(imName,im0)

def plotOld(md,img_raw,img_fft,peak_plot,width_plot,dataDict):


    #plot1.ts = np.mean(normed)
    img_raw.image = md.img0
    img_fft.image = md.F0
    img_raw.ts = dataDict['nevents'][-1]
    img_fft.ts = dataDict['nevents'][-1]
    mask = dataDict['nevents']>0

    eventMask = dataDict['nevents'][mask]

    order = np.argsort(eventMask)
    eventMask = eventMask[order]

    h_peak = dataDict['h_peak'][mask][order]
    v_peak = dataDict['v_peak'][mask][order]
    h_width = dataDict['h_width'][mask][order]
    v_width = dataDict['v_width'][mask][order]

    h_smooth = pandas.rolling_mean(h_peak,10)
    v_smooth = pandas.rolling_mean(v_peak,10)
    hw_smooth = pandas.rolling_mean(h_width,10)
    vw_smooth = pandas.rolling_mean(v_width,10)

    peak_plot.xdata = [eventMask,eventMask,eventMask,eventMask]
    peak_plot.ydata = [h_peak,v_peak,h_smooth,v_smooth]
    width_plot.xdata = [eventMask,eventMask,eventMask,eventMask]
    width_plot.ydata = [h_width,v_width,hw_smooth,vw_smooth]
    
    publish.send("RAW", img_raw) # send to the display
    publish.send("FFT",img_fft)
    publish.send("peak",peak_plot)
    publish.send("width",width_plot)

