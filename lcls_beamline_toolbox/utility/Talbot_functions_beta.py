import numpy as np
from beam import Beam
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import time


# this function selects an ROI around the Fourier peak in order to downsample the gradient
def calc_gradients(image, dx, dg, lambda0, peak, zT, downsample=3):

    """This function performs Fourier fringe analysis of the Talbot image,
    and downsamples the gradient by selecting an ROI around the Fourier peak
    Arguments:
        image -- Talbot image (NxM)
        dx -- pixel size in Talbot image (m)
        dg -- grating period (m)
        lambda0 -- wavelength (m)
        mag -- Talbot magnification (approximate)
        peak -- peak location in Fourier space for subtraction of linear phase
        zT -- Talbot distance, distance between grating and detector (m)
        downsample -- amount to downsample (power of 2) (default by factor of 8)
    Returns:
        h_grad -- downsampled gradient
        v_grad -- downsampled gradient
        zero_order -- downsampled zero order
        x1, y1 -- downsampled coordinates
        mid_peak -- average Fourier location of horizontal and vertical peaks
        p0 -- base second order coefficient
    """

    # calculate second order coefficient corresponding to peak shift that will be applied
    # R2 is the distance from focus to detector corresponding to this peak
    R2 = zT / (1 - dg * peak)
    p0 = np.pi / lambda0 / R2
    
    print(zT)

    mag = R2/(R2-zT)

    print('magnification: %.1f' % mag)

    # get image dimensions
    N, M = np.shape(image)
    
    # fourier transform
    start = time.time()
    fourier_plane = Beam.NFFT(image)
    end = time.time()
    print(end-start)
    #plt.figure()
    #plt.imshow(np.abs(fourier_plane))

    # spatial frequencies
    fxmax = 1.0/(dx*2)
    dfx = 2.*fxmax/M
    fx = np.linspace(-M/2.,M/2.-1,M)*dfx
    dfy = 2.*fxmax/N
    fy = np.linspace(-N/2.,N/2.-1,N)*dfy
    fx, fy = np.meshgrid(fx,fy)

    # spatial frequency of grating (m^-1)
    fG = 1.0/dg

    # mask off first order peaks in fourier space
    v_mask = fx**2 + (fy-fG/mag)**2 < (fG/mag/4)**2

    h_mask = (fx-fG/mag)**2 + fy**2 < (fG/mag/4)**2

    # mask off zero order peak in Fourier space
    zero_mask = fx**2 + fy**2 < (fG/mag/4)**2

    # multiply fourier plane by mask
    v_mask = fourier_plane*v_mask
    h_mask = fourier_plane*h_mask

    #plt.figure()
    #plt.imshow(np.abs(v_mask))
    #plt.figure()
    #plt.imshow(np.abs(h_mask))
    #plt.show()

    # multiply fourier plane by zero order mask
    zero_fourier = fourier_plane*zero_mask

    # find peak location for both horizontal and vertical
    # project along each dimension

    # thresholding of masked Fourier peaks to calculate peak location
    h_2 = beam_threshold(h_mask,.2)
    v_2 = beam_threshold(v_mask,.2)

    # set up coordinates (Talbot image plane)
    xp = np.linspace(-M/2,M/2-1,M)
    yp = np.linspace(-N/2,N/2-1,N)
    xp,yp = np.meshgrid(xp,yp)
    x1 = xp*dx
    y1 = yp*dx

    # find peaks in Fourier space
    h_peak = np.sum(h_2*fx)/np.sum(np.abs(h_2))
    v_peak = np.sum(v_2*fy)/np.sum(np.abs(v_2))

    h_mask = (fx-h_peak)**2 + fy**2<(fG/mag/4)**2
    v_mask = fx**2 + (fy-v_peak)**2<(fG/mag/4)**2

    h_mask = fourier_plane*h_mask
    v_mask = fourier_plane*v_mask
    # thresholding of masked Fourier peaks to calculate peak location
    h_2 = beam_threshold(h_mask,.2)
    v_2 = beam_threshold(v_mask,.2)
    # find peaks in Fourier space
    h_peak = np.sum(h_2*fx)/np.sum(np.abs(h_2))
    v_peak = np.sum(v_2*fy)/np.sum(np.abs(v_2))

    # find peak widths in Fourier space
    h_width = np.sqrt(np.sum(h_2*(fx-h_peak)**2)/np.sum(np.abs(h_2)))
    v_width = np.sqrt(np.sum(v_2*(fy-v_peak)**2)/np.sum(np.abs(v_2)))


    # calculate average position of horizontal/vertical peaks
    mid_peak = (v_peak+h_peak)/2.

    peak = mid_peak

    #R2x = zT / (1 - dg * h_peak)
    #p0x = np.pi / lambda0 / R2x
    #R2y = zT / (1 - dg * v_peak)
    #p0y = np.pi / lambda0 / R2y
    
    p0x = -np.pi/lambda0/zT * dg * h_peak
    p0y = -np.pi/lambda0/zT * dg * v_peak

    R2 = zT / (1 - dg * mid_peak)
    p0 = np.pi / lambda0 / R2

    # define linear phase related to approximate peak location
    #h_grating = np.exp(-1j*2.*np.pi*h_peak*x1)
    #v_grating = np.exp(-1j*2.*np.pi*v_peak*y1)
    h_grating = np.exp(-1j*2.*np.pi*h_peak*x1)
    v_grating = np.exp(-1j*2.*np.pi*v_peak*y1)


    # Fourier transform back to real space, and multiply by linear phase
    h_grad = np.conj(Beam.INFFT(h_mask)*h_grating)
    v_grad = np.conj(Beam.INFFT(v_mask)*v_grating)

    # back to Fourier space, now peaks have been shifted to zero
    h_fourier = Beam.NFFT(h_grad)
    v_fourier = Beam.NFFT(v_grad)

    # crop out center of Fourier pattern to downsample
    down = (2**downsample)*2

    v_fourier = v_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]
    h_fourier = h_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]
    zero_fourier = zero_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]

    # downsampled array size
    N2,M2 = np.shape(v_fourier)

    # downsampled image coordinates
    xp = np.linspace(-M2/2,M2/2-1,M2)
    yp = np.linspace(-N2/2,N2/2-1,N2)
    xp,yp = np.meshgrid(xp,yp)
    x1 = xp*dx*M/M2
    y1 = yp*dx*N/N2

    # back to real space, now downsampled
    h_grad = Beam.INFFT(h_fourier)
    v_grad = Beam.INFFT(v_fourier)

    # calculate zero order (downsampled)
    zero_order = np.abs(Beam.INFFT(zero_fourier))

    params = {}
    params['zero_order'] = zero_order
    params['x1'] = x1
    params['y1'] = y1
    params['h_peak'] = h_peak
    params['v_peak'] = v_peak
    params['h_width'] = h_width
    params['v_width'] = v_width
    params['p0x'] = p0x
    params['p0y'] = p0y
    params['p0'] = p0
    params['fourier'] = fourier_plane


    # output
    return h_grad, v_grad, params

def calc_gradients_static(image, dx, dg, lambda0, peak, zT, downsample=3):

    """This function performs Fourier fringe analysis of the Talbot image,
    and downsamples the gradient by selecting an ROI around the Fourier peak
    Arguments:
        image -- Talbot image (NxM)
        dx -- pixel size in Talbot image (m)
        dg -- grating period (m)
        lambda0 -- wavelength (m)
        mag -- Talbot magnification (approximate)
        peak -- peak location in Fourier space for subtraction of linear phase
        zT -- Talbot distance, distance between grating and detector (m)
        downsample -- amount to downsample (power of 2) (default by factor of 8)
    Returns:
        h_grad -- downsampled gradient
        v_grad -- downsampled gradient
        zero_order -- downsampled zero order
        x1, y1 -- downsampled coordinates
        mid_peak -- average Fourier location of horizontal and vertical peaks
        p0 -- base second order coefficient
    """

    # calculate second order coefficient corresponding to peak shift that will be applied
    # R2 is the distance from focus to detector corresponding to this peak
    R2 = zT / (1 - dg * peak)
    p0 = np.pi / lambda0 / R2
    
    print(zT)

    mag = R2/(R2-zT)

    print('magnification: %.1f' % mag)

    # get image dimensions
    N, M = np.shape(image)
    
    # fourier transform
    fourier_plane = Beam.NFFT(image)

    #plt.figure()
    #plt.imshow(np.abs(fourier_plane))

    # spatial frequencies
    fxmax = 1.0/(dx*2)
    dfx = 2.*fxmax/M
    fx = np.linspace(-M/2.,M/2.-1,M)*dfx
    dfy = 2.*fxmax/N
    fy = np.linspace(-N/2.,N/2.-1,N)*dfy
    fx, fy = np.meshgrid(fx,fy)

    # spatial frequency of grating (m^-1)
    fG = 1.0/dg

    # mask off first order peaks in fourier space
    v_mask = fx**2 + (fy-fG/mag)**2 < (fG/mag/2)**2

    h_mask = (fx-fG/mag)**2 + fy**2 < (fG/mag/2)**2

    # mask off zero order peak in Fourier space
    zero_mask = fx**2 + fy**2 < (fG/mag/2)**2

    # multiply fourier plane by mask
    v_mask = fourier_plane*v_mask
    h_mask = fourier_plane*h_mask

    h_2 = beam_threshold(h_mask,.2)
    v_2 = beam_threshold(v_mask,.2)
    # find peaks in Fourier space
    h_peak = np.sum(h_2*fx)/np.sum(np.abs(h_2))
    v_peak = np.sum(v_2*fy)/np.sum(np.abs(v_2))

    # find peak widths in Fourier space
    h_width = np.sqrt(np.sum(h_2*(fx-h_peak)**2)/np.sum(np.abs(h_2)))
    v_width = np.sqrt(np.sum(v_2*(fy-v_peak)**2)/np.sum(np.abs(v_2)))



    #plt.figure()
    #plt.imshow(np.abs(v_mask))
    #plt.figure()
    #plt.imshow(np.abs(h_mask))
    #plt.show()

    # multiply fourier plane by zero order mask
    zero_fourier = fourier_plane*zero_mask

    # find peak location for both horizontal and vertical
    # project along each dimension


    x_vis = np.sum(np.abs(h_mask))/np.sum(np.abs(zero_fourier))
    y_vis = np.sum(np.abs(v_mask))/np.sum(np.abs(zero_fourier))

    # set up coordinates (Talbot image plane)
    xp = np.linspace(-M/2,M/2-1,M)
    yp = np.linspace(-N/2,N/2-1,N)
    xp,yp = np.meshgrid(xp,yp)
    x1 = xp*dx
    y1 = yp*dx




    peak = fG/mag

    R2x = zT / (1 - dg * peak)
    p0x = np.pi / lambda0 / R2x
    R2y = zT / (1 - dg * peak)
    p0y = np.pi / lambda0 / R2y
    
    R2 = zT / (1 - dg * peak)
    p0 = np.pi / lambda0 / R2

    # define linear phase related to approximate peak location
    #h_grating = np.exp(-1j*2.*np.pi*h_peak*x1)
    #v_grating = np.exp(-1j*2.*np.pi*v_peak*y1)
    h_grating = np.exp(-1j*2.*np.pi*peak*x1)
    v_grating = np.exp(-1j*2.*np.pi*peak*y1)


    # Fourier transform back to real space, and multiply by linear phase
    h_grad = np.conj(Beam.INFFT(h_mask)*h_grating)
    v_grad = np.conj(Beam.INFFT(v_mask)*v_grating)

    # back to Fourier space, now peaks have been shifted to zero
    h_fourier = Beam.NFFT(h_grad)
    v_fourier = Beam.NFFT(v_grad)

    # crop out center of Fourier pattern to downsample
    down = (2**downsample)*2

    v_fourier = v_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]
    h_fourier = h_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]
    zero_fourier = zero_fourier[N/2-N/down:N/2+N/down,M/2-M/down:M/2+M/down]

    # downsampled array size
    N2,M2 = np.shape(v_fourier)

    # downsampled image coordinates
    xp = np.linspace(-M2/2,M2/2-1,M2)
    yp = np.linspace(-N2/2,N2/2-1,N2)
    xp,yp = np.meshgrid(xp,yp)
    x1 = xp*dx*M/M2
    y1 = yp*dx*N/N2

    # back to real space, now downsampled
    h_grad = Beam.INFFT(h_fourier)
    v_grad = Beam.INFFT(v_fourier)

    # calculate zero order (downsampled)
    zero_order = np.abs(Beam.INFFT(zero_fourier))



    params = {}
    params['zero_order'] = zero_order
    params['x1'] = x1
    params['y1'] = y1
    params['h_peak'] = h_peak
    params['v_peak'] = v_peak
    params['h_width'] = h_width
    params['v_width'] = v_width
    params['p0x'] = p0x
    params['p0y'] = p0y
    params['p0'] = p0
    params['fourier'] = fourier_plane
    params['x_vis'] = x_vis
    params['y_vis'] = y_vis


    # output
    return h_grad, v_grad, params



def get_amplitude(image, dx, dg, mag):
    """This function performs Fourier fringe analysis of the Talbot image,
    and downsamples the gradient by selecting an ROI around the Fourier peak
    Arguments:
        image -- Talbot image (NxM)
        dx -- pixel size in Talbot image (m)
        dg -- grating period (m)
        lambda0 -- wavelength (m)
        mag -- Talbot magnification (approximate)
        peak -- peak location in Fourier space for subtraction of linear phase
        zT -- Talbot distance, distance between grating and detector (m)
        downsample -- amount to downsample (power of 2) (default by factor of 8)
    Returns:
        h_grad -- downsampled gradient
        v_grad -- downsampled gradient
        zero_order -- downsampled zero order
        x1, y1 -- downsampled coordinates
        mid_peak -- average Fourier location of horizontal and vertical peaks
        p0 -- base second order coefficient
    """

    # get image dimensions
    N, M = np.shape(image)

    # fourier transform
    fourier_plane = Beam.NFFT(image)

    # spatial frequencies
    fxmax = 1.0 / (dx * 2)
    dfx = 2. * fxmax / M
    fx = np.linspace(-M / 2., M / 2. - 1, M) * dfx
    dfy = 2. * fxmax / N
    fy = np.linspace(-N / 2., N / 2. - 1, N) * dfy
    fx, fy = np.meshgrid(fx, fy)

    # spatial frequency of grating (m^-1)
    fG = 1.0 / dg

    # mask off first order peaks in fourier space
    v_mask = fx ** 2 + (fy - fG / mag) ** 2 < (fG / mag / 2) ** 2

    h_mask = (fx - fG / mag) ** 2 + fy ** 2 < (fG / mag / 2) ** 2

    # mask off zero order peak in Fourier space
    zero_mask = fx ** 2 + fy ** 2 < (fG / mag / 2) ** 2

    # multiply fourier plane by mask
    v_mask = fourier_plane * v_mask
    h_mask = fourier_plane * h_mask

    # multiply fourier plane by zero order mask
    zero_fourier = fourier_plane * zero_mask

    # find peak location for both horizontal and vertical
    # project along each dimension

    # thresholding of masked Fourier peaks to calculate peak location
    h_2 = beam_threshold(h_mask, .2)
    v_2 = beam_threshold(v_mask, .2)

    # set up coordinates (Talbot image plane)
    xp = np.linspace(-M / 2, M / 2 - 1, M)
    yp = np.linspace(-N / 2, N / 2 - 1, N)
    xp, yp = np.meshgrid(xp, yp)
    x1 = xp * dx
    y1 = yp * dx

    # find peaks in Fourier space
    h_peak = np.sum(h_2 * fx) / np.sum(np.abs(h_2))
    v_peak = np.sum(v_2 * fy) / np.sum(np.abs(v_2))

    # calculate average position of horizontal/vertical peaks
    mid_peak = (v_peak + h_peak) / 2.

    # Fourier transform back to real space, and multiply by linear phase
    h_grad = np.conj(Beam.INFFT(h_mask))
    v_grad = np.conj(Beam.INFFT(v_mask))

    params = {}
    params['x1'] = x1
    params['y1'] = y1
    params['mid_peak'] = mid_peak

    zero_order = (np.abs(h_grad) + np.abs(v_grad)) / 2

    # output
    return zero_order, params


def get_coeffs(line1,x1,zT,lambda0,dg,num):

    """Function to fit 1d lineouts of the gradient
    Arguments:
        line1 -- gradient lineout
        x1 -- coordinates for lineout
        zT -- talbot distance
        lambda0 -- wavelength
        dg -- grating pitch
        num -- polynomial order
    Returns:
        int1 -- integrated gradient
        p -- polynomial fit
        x1 -- coordinates
    """

    # get pixel size in image coordinates
    dx = x1[1]-x1[0]

    # get length of lineout
    N = line1.size

    # generate coordinates
    x1 = np.linspace(-N/2,N/2-1,N)*dx

    # cumulative sum to get wavefront (1d)
    int1 = cumtrapz(line1*dg/lambda0/zT,initial=0)*dx

    # polynomial fit to wavefront
    p = np.polyfit(x1,int1,num)

    return int1, p, x1

def get_z(p,lambda0):

    """Function to calculate distance to source/focus based on polynomial fit
    Arguments:
        p -- polynomial fit
        lambda0 -- wavelength (m)
    Returns:
        z -- distance to source/focus
    """

    # figure out polynomial order
    N = p.size

    # get second order term
    p2 = p[N-3]
    # calculate source distance based on curvature
    z = np.pi/lambda0/p2
    
    return z

def beam_threshold(im, frac):

    """Function for thresholding an image, useful for calculating center of mass
    Arguments:
        im -- image (2d)
        frac -- threshold fraction of image maximum
    Returns:
        im -- thresholded image
    """

    # make sure the image is not complex
    imOut = np.abs(im)

    # get maximum image value
    Imax = np.max(imOut)
    # get thresholding level
    thresh = Imax*frac
    # subtract threshold level
    imOut = imOut - thresh
    # set anything below threshold (now 0) to zero
    imOut[imOut<0] = 0

    return imOut

