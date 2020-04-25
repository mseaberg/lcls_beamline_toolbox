"""
pitch module

Part of the xraybeamline2d package

This module is used for calculating the pitch of a Talbot interference pattern.
Currently implements the single class TalbotLineout.
"""

import numpy as np
# from .legendre_1d import LegendreFit
from ..polyprojection.legendre import LegendreFit1D
from .util import Util
import matplotlib.pyplot as plt


class TalbotLineout:
    """
    Class for calculating 1D wavefronts from Talbot pattern lineouts.

    Attributes
    ----------
    lineout: (N,) ndarray
        Lineout from the image.
    fc: float
        estimated spatial frequency of Talbot pattern. Units of 1/pixel.
    factor: int
        Factor that reduces the selected width around the peak by an amount 1/factor.
    x_pitch: float
        Talbot pattern period (in pixels)
    residual: (N,) ndarray
        Residual phase of Talbot pattern, beyond linear.
    x_prime: (N,) ndarray
        coordinates of residual, units are pixels of lineout.
    dx_prime: float
        pixel size of x_prime (should be close to 1)
    x_vis: float
        estimate of the fringe visibility of the Talbot pattern
    vis2: float
        another estimate of the fringe visibility of the Talbot pattern
    """

    def __init__(self, lineout, fc, factor):
        """
        Initialize TalbotLineout
        :param lineout: (N,) ndarray
            Lineout from the image.
        :param fc: float
            estimated spatial frequency of Talbot pattern. Units of 1/pixel.
        :param factor: int
            Factor that reduces the selected width around the peak by an amount 1/factor.
        """
        self.lineout = lineout

        # fc has units of 1/pixel
        self.fc = fc
        self.factor = factor

        # initialize some calculated parameters
        self.x_pitch = 0.
        self.residual = np.zeros_like(lineout)
        self.x_prime = np.zeros_like(lineout)
        self.dx_prime = 0.
        self.x_vis = 0.
        self.vis2 = 0.

        # calculate pitch
        self.calc_pitch()

    def get_legendre(self, param):
        """
        Method to calculate Legendre coefficients of wavefront based on lineout.
        :param param: dict
            Relevant entries:
            dg: float
                grating pitch (meters)
            fraction: int
                Scaling factor for Talbot pattern based on which fractional plane we use. Can be 1, 2, or 3.
            dx: float
                PPM pixel size (meters)
            zT: float
                distance between WFS and PPM
            lambda0: float
                beam wavelength (meters)
        :return z_x: float
            distance to focus (from PPM). Positive number means focus is upstream of PPM.
        :return W: (k,) ndarray
            Array of Legendre coefficients. First coefficient corresponds to coefficient for P_1.
        :return xcoord: (N,) ndarray
            coordinates centered on beam (meters)
        :return wave: (N,) ndarray
            Legendre fit to recovered wavefront, 3rd order and above.
        """
        # get WFS parameters
        dg = param['dg']
        fraction = param['fraction']
        dx = param['dx']
        zT = param['zT']
        lambda0 = param['lambda0']

        # actual x coordinates (in meters)
        xcoord = self.x_prime * dx

        # pixel size for coordinates
        dx2 = xcoord[1]-xcoord[0]

        # magnification of Talbot pattern. Grating pitch is scaled by 1/fraction.
        mag_x = self.x_pitch * dx / (dg / fraction)

        # position of focus (positive means upstream of device)
        zf = zT * mag_x / (mag_x - 1.)

        print('zf: '+str(zf))

        # residual phase gradient
        grad = -self.residual * dg / fraction / lambda0 / zT

        # generate the Legendre polynomial basis
        print('generating basis')
        fit_object = LegendreFit1D(np.size(grad), 16)
        print('basis generated')

        # get Legendre coefficients. Nothing is masked out for now.
        W = fit_object.coeff_from_grad(grad, dx2, np.ones(np.size(grad), dtype=bool)).flatten()

        # second order coefficient based on distance to focus
        max_x = dx2*np.size(xcoord)/2
        print('max1: ' + str(max_x))
        print('max2: '+str(np.max(xcoord)))
        # A = np.pi / lambda0 / zf * np.max(xcoord) ** 2
        A = np.pi / lambda0 / zf * max_x**2
        # convert to Legendre coefficient for P_2.
        C2 = 2. / 3. * A
        C0 = 1. / 3. * A

        # add any second order found from Legendre fit to residual.
        C2 += W[1]

        # updated distance to focus
        # z_x = 2 * np.pi / (3. * lambda0 * C2) * np.max(xcoord) ** 2
        z_x = 2 * np.pi / (3. * lambda0 * C2) * max_x**2

        # now set first and second order coefficients to zero to only get high order phase
        W[1] = 0
        W[0] = 0

        # get high order wavefront
        wave = fit_object.wavefront_fit(W)

        # return
        return z_x, W, xcoord, wave

    def get_pitch(self):
        """
        Method to return the calculated pitch.
        :return x_pitch: float
            Period of Talbot pattern (pixels)
        """
        # just return the pitch.
        return self.x_pitch

    def get_residual(self):
        """
        Method to return residual wavefront
        :return residual: (N,) ndarray
            Residual wavefront (higher than 2nd order). Units are radians.
        :return x_prime: (N,) ndarray
            coordinates. Units are pixels from the original lineout.
        """
        return self.residual, self.x_prime

    def calc_gradients(self, param):
        """
        :param param: dict
            Relevant entries:
            dg: float
                grating pitch (meters)
            fraction: int
                Scaling factor for Talbot pattern based on which fractional plane we use. Can be 1, 2, or 3.
            dx: float
                PPM pixel size (meters)
            zT: float
                distance between WFS and PPM
            lambda0: float
                beam wavelength (meters)
        Returns:
        :return h_grad: (N,) ndarray
            downsampled gradient
        :return params: dict
            Entries include:
            zero_order: (N,) ndarray
                filtered lineout with high spatial frequencies removed
            x1: (N,) ndarray
                downsampled coordinates (meters)
            h_peak: float
                Peak in Fourier space (1/pixels)
            h_width: float
                Width of peak in Fourier space (1/pixels)
            p0x: float
                2nd order phase coefficient
            p0: float
                2nd order phase coefficient
            fourier: (N,) ndarray
                Fourier domain of lineout
        """

        # get WFS parameters        
        dg = param['dg']
        fraction = param['fraction']
        dx = param['dx']
        zT = param['zT']
        lambda0 = param['lambda0']

        # get image dimensions
        N = np.size(self.lineout)
        # calculate spatial frequencies
        dfx = 1./N

        # spatial frequencies
        fx = np.linspace(-N/2, N/2-1, N, dtype=float) * dfx

        # Mask around the peak
        mask0 = (fx - self.fc)**2 < (self.fc / 2 / fraction)**2

        # fourier transform
        fourier_plane = Util.nfft1(self.lineout)

        # FT with everything but the peak masked out
        x_fft = fourier_plane * mask0

        # find peak in Fourier domain
        x_peak = np.argmax(np.abs(x_fft) * mask0)

        # mask for zero order peak in Fourier space
        zero_mask = fx**2 < (self.fc / 2 / fraction)**2

        # multiply fourier plane by zero order mask
        zero_fourier = fourier_plane * zero_mask

        # find peak location for both horizontal and vertical
        # project along each dimension

        # thresholding of masked Fourier peaks to calculate peak location
        h_2 = Util.threshold_array(x_fft, .2)

        # set up coordinates (Talbot image plane) units are pixels
        xp = np.linspace(-N/2, N/2-1, N, dtype=float)

        # find peaks in Fourier space
        h_peak = np.sum(h_2 * fx) / np.sum(np.abs(h_2))

        # updated mask centered on peak
        h_mask = (fx - h_peak)**2 < (self.fc / 2 / fraction)**2

        # peak with updated mask
        h_2 = fourier_plane * h_mask
        # thresholding of masked Fourier peaks to calculate peak location
        h_2 = Util.threshold_array(h_2, .2)
        # find peaks in Fourier space
        h_peak = np.sum(h_2*fx)/np.sum(np.abs(h_2))

        # find peak widths in Fourier space
        h_width = np.sqrt(np.sum(h_2*(fx-h_peak)**2)/np.sum(np.abs(h_2)))

        # max spatial frequency
        fxmax = 1.0/(dx*2)

        # calculate 2nd order coefficient based on peak location
        p0x = -np.pi / lambda0 / zT * dg * h_peak * (fxmax / .5) / fraction

        # calculate 2nd order coefficient another way
        R2 = zT / (1 - dg * h_peak)
        p0 = np.pi / lambda0 / R2

        # define linear phase related to approximate peak location
        h_grating = np.exp(-1j * 2. * np.pi * h_peak * xp)

        # phase gradient back in real space, multiplied by linear phase to remove linear term
        h_grad = np.conj(Util.infft1(h_mask) * h_grating)

        # back to Fourier space, now peaks have been shifted to zero
        h_fourier = Util.nfft1(h_grad)

        # crop out center of Fourier pattern to downsample. Downsampling hard-coded to 16.
        downsample = 4
        down = (2**downsample) * 2

        # crop out the center of the Fourier space pattern. Size is based on amount of downsampling.
        h_fourier = h_fourier[int(N / 2 - N / down):int(N / 2 + N / down)]
        zero_fourier = zero_fourier[int(N / 2 - N / down):int(N / 2 + N / down)]

        # downsampled array size
        N2 = np.size(h_fourier)

        # downsampled image coordinates
        xp = np.linspace(-N2 / 2, N2 / 2 - 1, N2)
        # multiply by original pixel size, and scale by amount of downsampling.
        x1 = xp * dx * N / N2

        # gradient back in real space, now downsampled
        h_grad = Util.infft1(h_fourier)

        # calculate zero order back in real space (downsampled)
        zero_order = Util.infft1(zero_fourier)

        params = {'zero_order': zero_order,
                  'x1': x1,
                  'h_peak': h_peak,
                  'h_width': h_width,
                  'p0x': p0x,
                  'p0': p0,
                  'fourier': fourier_plane}

        # output
        return h_grad, params

    def calc_pitch(self):
        """
        Method to calculate lineout pitch
        :return: None
        """
        # get lineout length
        N = np.size(self.lineout)

        # calculate spatial frequencies
        dfx = 1./N

        # spatial frequencies
        fx = np.linspace(0, N-1, N, dtype=float) * dfx

        # Fourier mask to mask out peak
        mask0 = (fx-self.fc)**2 < (self.fc/2/self.factor)**2

        filter = np.cos((fx-self.fc)*np.pi/2/(self.fc/2/self.factor))

        # fourier domain of lineout
        F1 = np.fft.fft(self.lineout)

        # fourier domain with everything but peak masked out
        x_fft = F1*mask0*filter

        # plt.figure()
        # plt.plot(np.abs(x_fft)/np.max(np.abs(x_fft)))
        # plt.plot(np.abs(F1)/np.max(np.abs(x_fft)))
        # plt.plot(filter)

        # Find peak in Fourier domain
        x_peak = np.argmax(np.abs(x_fft)*mask0)

        # integer peak position (1/pixel)
        x_int = x_peak*dfx

        # crop width in Fourier domain
        crop_width = int(x_peak / 2 / self.factor)

        # pad width to make cropped peak the same size as x_fft
        pad_width = int(N / 2 - crop_width)

        # crop out peak and pad with zeros
        x_pad = np.pad(x_fft[int(x_peak - crop_width):int(x_peak + crop_width)], pad_width)

        # shift peak to zero
        x_shift = np.fft.fftshift(x_pad)

        # Nx1 = np.size(x_fft)
        #
        # x_shift = np.fft.fftshift(
        #     np.pad(x_fft[int(x_peak - int(x_peak / 2 / self.factor)):int(x_peak + int(x_peak / 2 / self.factor))],
        #            int(Nx1 / 2), 'constant'))

        # get size of x_shift just in case it's off by 2 or something from x_fft
        Nx = np.size(x_shift)

        # calculate dx of new coordinates in case it's slightly off
        dx_prime = 1. / (Nx * dfx)
        print('dxprime: '+str(dx_prime))

        # define coordinates for residual phase
        x_prime = np.linspace(-Nx/2, Nx/2-1, Nx, dtype=float) * dx_prime

        # plt.figure()
        # plt.plot(np.abs(x_shift))

        # inverse FFT to get residual phase
        x_filt = np.fft.ifft(x_shift)

        # remove edges just in case there's an issue there
        x_filt = x_filt[int(N/8):-int(N/8)]
        x_prime = x_prime[int(N/8):-int(N/8)]

        # polynomial fit for residual unwrapped phase
        px = np.polyfit(x_prime, np.unwrap(np.angle(x_filt)), 1)
        # px = np.polynomial.legendre.legfit(x_prime, np.unwrap(np.angle(x_filt)), 3)
        # px[2:] = 0

        # residual after subtracting any remaining linear phase and/or offset
        residual = np.unwrap(np.angle(x_filt)) - np.polyval(px, x_prime)
        # residual = np.unwrap(np.angle(x_filt)) + np.polynomial.legendre.legval(x_prime, px)

        # plt.figure()
        # plt.plot(x_prime, residual)
        # plt.plot(x_prime, np.polyval(px, x_prime))
        # plt.plot(x_prime, np.unwrap(np.angle(x_filt)))

        # update centroid with fractional part from linear phase
        x_centroid = x_int + px[0] / 2 / np.pi
        # x_centroid = x_int + px[1] / 2 / np.pi

        # period is 1/peak in Fourier domain (pixels)
        x_pitch = 1./x_centroid

        print(x_pitch)

        # set some attributes
        self.x_pitch = x_pitch
        self.residual = residual
        self.x_prime = x_prime
        self.dx_prime = dx_prime

    def calc_pitch_vis(self):
        """
        Method to both calculate pitch and visibility. This method might be out of date.
        :return: None
        """
        # get lineout length
        N = np.size(self.lineout)

        # calculate spatial frequencies
        dfx = 1./N

        fx = np.linspace(0,N-1,N,dtype=float)*dfx

        mask0 = (fx-self.fc)**2<(self.fc/2)**2

        F1 = np.fft.fft(self.lineout)

        zeromask = fx<self.fc/2
        zeromask = np.logical_or(zeromask, fx>1-self.fc/2)


        zero_order = F1*zeromask

        x_fft = F1*mask0

        #plt.figure()
        #plt.plot(np.abs(x_fft))
        #plt.plot(np.abs(zero_order))
        #plt.show()

        x_vis = np.max(np.abs(x_fft))/np.max(np.abs(zero_order))*2

        vis2 = (np.max(self.lineout)-np.min(self.lineout))/(np.max(self.lineout)+np.min(self.lineout))

        #print(np.sum(np.abs(mask0)))

        x_peak = np.argmax(np.abs(x_fft)*mask0)

        x_int = x_peak*dfx

        #print(x_peak)

        Nx1 = np.size(x_fft)

        # shift peak to zero
        #x_shift = np.fft.fftshift(np.pad(x_fft[int(x_peak-int(x_peak/2/factor)):int(x_peak+int(x_peak/2/factor))],int(Nx1/2),'constant'))
        x_shift = np.fft.fftshift(x_fft[int(x_peak-int(x_peak/2/self.factor)):int(x_peak+int(x_peak/2/self.factor))])

        Nx = np.size(x_shift)

        dx_prime = 1./Nx/dfx

        x_prime = np.linspace(-Nx/2,Nx/2-1,Nx,dtype=float)*dx_prime

        #x_filt = np.fft.ifft(x_shift)[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

        x_filt = np.fft.ifft(x_shift)

        x_filt = x_filt[2:-2]
        x_prime = x_prime[2:-2]
        px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

        #x_prime = x_prime[int(Nx/2-3*Nx/8)+1:int(Nx/2+3*Nx/8)-1]

        #px = np.polyfit(x_prime,np.unwrap(np.angle(x_filt)),1)

        residual = np.unwrap(np.angle(x_filt)) - px[1] - px[0]*x_prime

        x_centroid = x_int+px[0]/2/np.pi

        x_pitch = 1./x_centroid

        self.x_pitch = x_pitch
        self.residual = residual
        self.x_prime = x_prime
        self.x_vis = x_vis
        self.vis2 = vis2
