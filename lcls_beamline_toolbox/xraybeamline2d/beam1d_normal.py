"""
beam1d_normal module

Part of the xraybeamline2d package.

This module keeps track of the linear phase separately but includes quadratic phase with higher orders. Plan to
eventually integrate this into beam1d, with an option to include the quadratic phase or not.

Currently implements the following classes:
Beam: handles 2 x 1D beam propagation
GaussianSource: helper class for initializing a Beam object

All distances/lengths are in meters, spatial frequencies in 1/meters, angles in radians, energies in eV,
unless otherwise indicated.
"""
import numpy as np
# import matplotlib.pyplot as plt
from .util import Util


class Beam:
    """
    Class for handling 1D beam propagation.

    Attributes
    ----------
    wavex: (M,) ndarray
        complex-valued array containing beam's amplitude and phase information for horizontal direction
    wavey: (N,) ndarray
        complex-valued array containing beam's amplitude and phase information for vertical direction
    x: (M,) ndarray
        array of x coordinates at current beam location
    y: (N,) ndarray
        array of y coordinates at current beam location
    fx: (M,) ndarray
        array of spatial frequency coordinates at current beam location
    fy: (N,) ndarray
        array of spatial frequency coordinates at current beam location
    dx: float
        horizontal pixel size at current beam location
    dy: float
        vertical pixel size at current beam location
    cx: float
        horizontal beam center at current beam location
    cy: float
        vertical beam center at current beam location
    ax: float
        horizontal beam propagation angle (relative to z-axis)
    ay: float
        vertical beam propagation angle (relative to z-axis)
    zx: float
        horizontal radius of curvature
    zy: float
        vertical radius of curvature
    zRx: float
        size of horizontal focal range (half-width)
    zRy: float
        size of vertical focal range (half-width)
    photonEnergy: float
        beam photon energy
    lambda0: float
        beam wavelength
    k0: float
        beam wavenumber (2pi/lambda0)
    N: int
        number of pixels in vertical direction (in wave array)
    M: int
        number of pixels in horizontal direction (in wave array)
    focused_x: bool
        True if beam is currently within the horizontal focal range
    focused_y: bool
        True if beam is currently within the vertical focal range
    rangeFactor: float
        Multiplier to Rayleigh range for calculating zRx and zRy. zRx = Rayleigh_x * rangeFactor and ditto for y.
    scaleFactor: float
        Scale of how large the grid should be relative to the beam size in out of focus planes. Default is 8.
    """

    def __init__(self, initial_beam_x=None, initial_beam_y=None, beam_params=None):
        """
        Method to initialize a Beam object.
        :param initial_beam_x: (N,) ndarray
            complex-valued array to initialize the amplitude/phase of the beam in horizontal direction (optional)
        :param initial_beam_y: (M,) ndarray
            complex-valued array to initialize the amplitude/phase of the beam in vertical direction (optional)
        :param beam_params: dict
            Following is a list of the keys in this dictionary:
            cx: float
                initial horizontal beam center at initial plane (optional, default 0)
            cy: float
                initial vertical beam center at initial plane (optional, default 0)
            ax: float
                initial horizontal beam propagation angle (optional, default 0)
            ay: float
                initial vertical beam propagation angle (optional, default 0)
            dx: float
                initial horizontal pixel size (required when initial_beam is provided, otherwise optional)
            dy: float
                initial vertical pixel size (optional, defaults to horizontal pixel size)
            photonEnergy: float
                beam photon energy (required)
            rangeFactor: float
                factor controlling where at which point to switch between propagation via Fresnel scaling,
                versus typical unscaled propagation. This is in units of Rayleigh lengths from the focus.
                (optional, default 10)
            N: int
                grid size in pixels (over-written if initial_beam is provided)
            sigma_x: float
                horizontal beam width (1/e radius in field strength) at waist
                (required if initial_beam is not provided)
            sigma_y: float
                vertical beam width (1/e radius in field strength) at waist (required if initial_beam is not provided)
            z0x: float
                initial distance from horizontal waist (positive if beam is diverging, negative if beam is converging)
            z0y: float
                initial distance from vertical waist (positive if beam is diverging, negative if beam is converging)
        """

        # beam_param keys to check
        key_list = ['cx', 'cy', 'ax', 'ay']

        # set some attributes from the beam_param dict
        for key in key_list:
            if key in beam_params.keys():
                setattr(self, key, beam_params[key])
            else:
                setattr(self, key, 0.0)

        # take in manual input of initial wavefront/amplitude
        if initial_beam_x and initial_beam_y:
            # initialize wave with initial_beam array
            self.wavex = np.copy(initial_beam_x).astype(complex)
            self.wavey = np.copy(initial_beam_y).astype(complex)
            # set pixel size
            self.dx = beam_params['dx']
            # check if dy was provided, otherwise default to dx
            if 'dy' in beam_params.keys():
                self.dy = beam_params['dy']
            else:
                self.dy = np.copy(self.dx)
            # check if z0x was provided, otherwise default to 0
            if 'z0x' in beam_params.keys():
                self.zx = beam_params['z0x']
            else:
                self.zx = 0.0
            # check if z0y was provided, otherwise default to 0
            if 'z0y' in beam_params.keys():
                self.zy = beam_params['z0y']
            else:
                self.zy = 0.0
            # set ranges to zero for now, see if this works.
            self.zRx = 0
            self.zRy = 0

        # if initial_beam not provided, create GaussianSource from beam parameters
        else:
            # pass in parameters to GaussianSource
            b1 = GaussianSource(beam_params)
            # initialize relevant parameters from GaussianSource b1
            self.wavex = b1.source_x.astype(complex)
            self.wavey = b1.source_y.astype(complex)
            self.zx = b1.z0x
            self.zy = b1.z0y
            self.dx = b1.dx
            self.dy = np.copy(self.dx)

        # set photon energy and calculate wavelength, wavenumber
        self.photonEnergy = beam_params['photonEnergy']
        self.lambda0 = 1239.8 / beam_params['photonEnergy'] * 1e-9
        self.k0 = 2.0 * np.pi / self.lambda0

        # get array shape
        self.M = self.wavex.size
        self.N = self.wavey.size

        # set up coordinates
        x = np.linspace(-self.M / 2.0 * self.dx, (self.M / 2.0 - 1) * self.dx, self.M, dtype=float)
        y = np.linspace(-self.N / 2.0 * self.dy, (self.N / 2.0 - 1) * self.dy, self.N, dtype=float)
        self.x = x
        self.y = y

        # offset coordinates by beam center
        self.x = self.x + self.cx
        self.y = self.y + self.cy

        # calculate spatial frequencies at initial plane
        fx_max = 1.0 / (2.0 * self.dx)
        fy_max = 1.0 / (2.0 * self.dy)
        dfx = fx_max / self.M
        fx = np.linspace(-fx_max, fx_max - dfx, self.M)
        dfy = fy_max / self.N
        fy = np.linspace(-fy_max, fy_max - dfy, self.N)
        self.fx = fx
        self.fy = fy

        # if we're already inside the focal range, we need to multiply by quadratic phase in order for
        # propagation to work properly.
        self.wavex *= np.exp(1j * np.pi / self.lambda0 / self.zx * (self.x - self.cx)**2)
        self.wavey *= np.exp(1j * np.pi / self.lambda0 / self.zy * (self.y - self.cy)**2)

    def update_parameters(self, dz):
        """
        Method to update beam geometric parameters
        :param dz: float
            Distance to be propagated
        :return: None
        """

        # update beam center
        self.cx += self.ax * dz
        self.cy += self.ay * dz
        # move x/y arrays by change in beam center
        self.x = self.x + self.ax * dz
        self.y = self.y + self.ay * dz
        # update horizontal and vertical radii of curvature by propagation distance
        self.zx = self.zx + dz
        self.zy = self.zy + dz

    def rescale_x_noshift(self, factor):
        """
        Method to rescale x axis without scaling the beam center position
        :param factor: float
            scaling factor
        :return: None
        """

        # remove beam center
        self.x -= self.cx
        # scale coordinates centered around zero
        self.rescale_x(factor)
        # add beam center back to rescaled coordinates
        self.x += self.cx

    def rescale_y_noshift(self, factor):
        """
        Method to rescale y axis without scaling the beam center position
        :param factor: float
            scaling factor
        :return: None
        """

        # remove beam center
        self.y -= self.cy
        # scale coordinates centered around zero
        self.rescale_y(factor)
        # add beam center back to rescaled coordinates
        self.y += self.cy

    def propagation(self, dz_real):
        """
        Method to propagate the beam using two-step Fourier transform method. Compatible with Fresnel scaling theorem
        or normal unscaled propagation based on inputs.
        :param dz_real: float
            Actual distance to propagate.
        :param dz_x: float
            Effective horizontal propagation.
        :param dz_y: float
            Effective vertical propagation.
        :return: None
        """

        # phase to multiply by in Fourier plane
        phi_prop_x = (self.k0 * dz_real - self.k0 / 2 *
                    (self.lambda0 * self.fx) ** 2 * dz_real)

        phi_prop_y = (self.k0 * dz_real - self.k0 / 2 *
                    (self.lambda0 * self.fy) ** 2 * dz_real)

        # calculate Fourier plane of beam
        gx = Util.nfft1(self.wavex)
        # multiply by propagation phase
        gx *= np.exp(1j * phi_prop_x)
        # Inverse Fourier transform to calculate beam at new plane
        self.wavex = Util.infft1(gx)

        # calculate Fourier plane of beam
        gy = Util.nfft1(self.wavey)
        # multiply by propagation phase
        gy *= np.exp(1j * phi_prop_y)
        # Inverse Fourier transform to calculate beam at new plane
        self.wavey = Util.infft1(gy)

    def beam_prop(self, dz):
        """
        Method that handles beam propagation.
        :param dz: float
            Distance to propagate
        :param dz_progress: float
            Distance already propagated. This is an internal parameter used for recursive calls to the method.
        :param index: float
            Number of calls to the method. This is an internal parameter used for recursive calls to the method.
        :return beam.wavex: (M,) ndarray
            Returns the complex-valued beam array.
        :return beam.wavey: (M,) ndarray
            Returns the complex-valued beam array.
        """

        self.propagation(dz)
        # update the parameters and we're done
        self.update_parameters(dz)

        # return the wave
        return self.wavex, self.wavey

    def rescale_x(self, factor):
        """
        Method to rescale x coordinates and recalculate spatial frequencies
        :param factor: float
            scaling factor
        :return: None
        """

        # scale coordinates
        self.x = self.x * factor
        # recalculate spatial frequencies
        self.new_fx()

    def rescale_y(self, factor):
        """
        Method to rescale y coordinates and recalculate spatial frequencies
        :param factor: float
            scaling factor
        :return: None
        """

        # scale coordinates
        self.y = self.y * factor
        # recalculate spatial frequencies
        self.new_fx()

    def multiply_screen(self, screen):
        """
        Method to multiply beam by a complex-valued screen.
        :param screen: (N,M) ndarray
            Multiply the beam by this (possibly complex-valued) screen. Must be same shape as self.wave.
        :return self.wave: (N,M) ndarray
            Return the complex-valued beam amplitude/phase
        """

        # do the multiplication
        self.wave = self.wave * screen
        return self.wave

    def new_fx(self):
        """
        Method to recalculate spatial frequencies.
        :return: None
        """

        # check if x coordinates are reversed before calculating pixel size
        if self.x[0] > self.x[1]:
            # calculate pixel size based on first two entries in x
            self.dx = self.x[0] - self.x[1]
        else:
            # calculate pixel size based on first two entries in x
            self.dx = self.x[1]-self.x[0]
        # check if y coordinates are reversed before calculating pixel size
        if self.y[0] > self.y[1]:
            # calculate pixel size based on first two entries in y
            self.dy = self.y[0] - self.y[1]
        else:
            # calculate pixel size based on first two entries in y
            self.dy = self.y[1] - self.y[0]

        # recalculate maximum spatial frequencies based on new pixel sizes
        fx_max = 1.0 / (2 * self.dx)
        fy_max = 1.0 / (2 * self.dy)
        # calculate spatial frequency pixel size
        dfx = fx_max / self.M
        dfy = fy_max / self.N
        # spatial frequency coordinates
        self.fx = np.linspace(-fx_max, fx_max - dfx, self.M)
        self.fy = np.linspace(-fy_max, fy_max - dfy, self.N)

        
class GaussianSource:
    """
    Class for representing generating a monochromatic Gaussian beam.

    Attributes
    ----------
    source: (N,N) ndarray
        beam amplitude at output plane
    x: (N,N) ndarray
        x coordinates at output plane
    y: (N,N) ndarray
        y coordinates at output plane
    sigma_x: float
        Horizontal sigma at beam waist
    sigma_y: float
        Vertical sigma at beam waist
    N: int
        beam array size (N x N)
    z0x: float
        distance from horizontal waist
    z0y: float
        distance from vertical waist
    dx: float
        Horizontal pixel size
    dy: float
        Vertical pixel size
    wavelength: float
        beam wavelength
    photonEnergy: float
        beam photon energy (eV)
    zRx: float
        horizontal Rayleigh range
    zRy: float
        vertical Rayleigh range
    wx: float
        horizontal beam width at output plane (1/e^2)
    wy: float
        vertical beam width at output plane (1/e^2)
    """
    
    def __init__(self, beam_params):
        """
        Initialize a GaussianSource object
        :param beam_params: dict
            Following is a list of the relevant keys in this dictionary:
            dx: float
                initial horizontal pixel size (optional)
            dy: float
                initial vertical pixel size (optional, defaults to horizontal pixel size)
            photonEnergy: float
                beam photon energy (required)
            rangeFactor: float
                factor controlling where at which point to switch between propagation via Fresnel scaling,
                versus typical unscaled propagation. This is in units of Rayleigh lengths from the focus.
                (required)
            N: int
                grid size in pixels (required)
            sigma_x: float
                horizontal beam width (1/e radius in field strength) at waist (required)
            sigma_y: float
                vertical beam width (1/e radius in field strength) at waist (required)
            z0x: float
                initial distance from horizontal waist (positive if beam is diverging, negative if beam is converging)
                (optional)
            z0y: float
                initial distance from vertical waist (positive if beam is diverging, negative if beam is converging)
                (optional)
        """

        # set some attributes
        self.sigma_x = beam_params['sigma_x']
        self.sigma_y = beam_params['sigma_y']
        self.N = int(beam_params['N'])
        if beam_params['z0x']:
            self.z0x = beam_params['z0x']
        else:
            self.z0x = 0.0
        if beam_params['z0y']:
            self.z0y = beam_params['z0y']
        else:
            self.z0y = 0.0
        self.photonEnergy = beam_params['photonEnergy']
        if 'dx' in beam_params.keys():
            self.dx = beam_params['dx']
            self.dy = np.copy(self.dx)
        else:
            self.dx = None
            self.dy = None
        
        # calculate wavelength (m)
        self.wavelength = 1239.8/self.photonEnergy*1e-9
        # calculate Rayleigh ranges (m)
        self.zRx = np.pi*self.sigma_x**2/self.wavelength
        self.zRy = np.pi*self.sigma_y**2/self.wavelength
        
        # calculate beam widths
        self.wx = self.sigma_x*np.sqrt(1+(self.z0x/self.zRx)**2)
        self.wy = self.sigma_y*np.sqrt(1+(self.z0y/self.zRy)**2)

        # calculate divergence
        divergence_x = self.wx / self.z0x
        divergence_y = self.wy / self.z0y

        # print beam width and divergence
        print('FWHM in x: '+str(1.18*self.wx*1e6)+' microns')
        print('FWHM in y: '+str(1.18*self.wy*1e6)+' microns')
        print('FWHM Divergence (x): %.1f \u03BCrad' % (divergence_x * 1e6 * 1.18))
        print('FWHM Divergence (y): %.1f \u03BCrad' % (divergence_y * 1e6 * 1.18))

        # factor to multiply by Rayleigh range to check if the beam is inside the focal range
        factor = beam_params['rangeFactor']*(2/1.18)**2

        scale = beam_params['scaleFactor']

        # check if we're inside this range
        focused_x = -self.zRx*factor <= self.z0x < self.zRx*factor
        focused_y = -self.zRy*factor <= self.z0y < self.zRy*factor
        print(self.zRx)
        print(self.zRy)

        # need to modify this in the case where beam starts out "in focus"
        if self.dx is None:
            # if not, define array to cover 4x FWHM
            fwhm_x = 1.18*self.wx
            fwhm_y = 1.18*self.wy

            # set field of view
            if focused_x:
                # set it so that it will be 8 times the FWHM at the boundary of the focal range
                FOV_x = np.abs(self.zRx * factor/self.z0x) * scale * fwhm_x
                print('x is focused')
            else:
                # if out of focus, just set to 8 times the FWHM
                FOV_x = scale*fwhm_x
            if focused_y:
                # set it so that it will be 8 times the FWHM at the boundary of the focal range
                print('y is focused')
                FOV_y = np.abs(self.zRy * factor/self.z0y) * scale * fwhm_y
            else:
                # if out of focus, just set to 8 times the FWHM
                FOV_y = scale*fwhm_y

            # calculate the resulting pixel size
            self.dx = FOV_x/self.N
            self.dy = FOV_y/self.N

        # coordinates
        self.x = np.linspace(-self.N / 2, self.N / 2 - 1, self.N) * self.dx
        self.y = np.linspace(-self.N / 2, self.N / 2 - 1, self.N) * self.dy

        # beam amplitude
        self.source_x = np.exp(-((self.x / self.wx) ** 2))
        self.source_y = np.exp(-((self.y / self.wy) ** 2))
