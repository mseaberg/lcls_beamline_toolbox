"""
beam module

Part of the xraybeamline2d package.

Currently implements the following classes:
Beam: handles 2D beam propagation
GaussianSource: helper class for initializing a Beam object

All distances/lengths are in meters, spatial frequencies in 1/meters, angles in radians, energies in eV,
unless otherwise indicated.
"""
import numpy as np
# import matplotlib.pyplot as plt
from .util import Util


class Beam:
    """
    Class for handling 2D beam propagation.

    Attributes
    ----------
    wave: (N,M) ndarray
        complex-valued array containing beam's amplitude and phase information
    x: (N,M) ndarray
        array of x coordinates at current beam location
    y: (N,M) ndarray
        array of y coordinates at current beam location
    fx: (N,M) ndarray
        array of spatial frequency coordinates at current beam location
    fy: (N,M) ndarray
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
    """

    def __init__(self, initial_beam=None, beam_params=None):
        """
        Method to initialize a Beam object.
        :param initial_beam: (N,M) ndarray
            complex-valued array to initialize the amplitude/phase of the beam (optional)
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

        # check if rangeFactor was provided, or default to 10.
        if 'rangeFactor' in beam_params.keys():
            self.rangeFactor = beam_params['rangeFactor']
        else:
            # default value is 10
            self.rangeFactor = 10
            # add this to the dictionary
            beam_params['rangeFactor'] = self.rangeFactor

        # take in manual input of initial wavefront/amplitude
        if initial_beam is not None:
            # initialize wave with initial_beam array
            self.wave = np.copy(initial_beam).astype(complex)
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
            self.wave = b1.source.astype(complex)
            self.zx = b1.z0x
            self.zy = b1.z0y
            self.dx = b1.dx
            self.dy = np.copy(self.dx)
            # multiply rayleigh range by rangeFactor, and multiply by factor to make it consistent with
            # other calculations
            self.zRx = b1.zRx * self.rangeFactor * (2/1.18)**2
            self.zRy = b1.zRy * self.rangeFactor * (2/1.18)**2

        # set photon energy and calculate wavelength, wavenumber
        self.photonEnergy = beam_params['photonEnergy']
        self.lambda0 = 1239.8 / beam_params['photonEnergy'] * 1e-9
        self.k0 = 2.0 * np.pi / self.lambda0

        # get array shape
        self.N, self.M = self.wave.shape

        # set up coordinates
        x = np.linspace(-self.M / 2.0 * self.dx, (self.M / 2.0 - 1) * self.dx, self.M, dtype=float)
        y = np.linspace(-self.N / 2.0 * self.dy, (self.N / 2.0 - 1) * self.dy, self.N, dtype=float)
        self.x, self.y = np.meshgrid(x, y)

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
        self.fx, self.fy = np.meshgrid(fx, fy)

        # check if we're inside this range and initialize focus attribute
        self.focused_x = -self.zRx <= self.zx < self.zRx
        self.focused_y = -self.zRy <= self.zy < self.zRy
        # print(self.zRx)
        # print(self.zRy)

        # if we're already inside the focal range, we need to multiply by quadratic phase in order for
        # propagation to work properly.
        if self.focused_x:
            self.wave *= np.exp(1j * np.pi / self.lambda0 / self.zx * (self.x - self.cx)**2)
        if self.focused_y:
            self.wave *= np.exp(1j * np.pi / self.lambda0 / self.zy * (self.y - self.cy)**2)

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

    def propagation(self, dz_real, dz_x, dz_y):
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
        phi_prop = (self.k0 * dz_real - self.k0 / 2 *
                    ((self.lambda0 * self.fx) ** 2 * dz_x +
                     (self.lambda0 * self.fy) ** 2 * dz_y))

        # calculate Fourier plane of beam
        g = Util.nfft(self.wave)
        # multiply by propagation phase
        g *= np.exp(1j * phi_prop)
        # Inverse Fourier transform to calculate beam at new plane
        self.wave = Util.infft(g)

    def change_z(self, new_zx=None, new_zy=None):
        """
        Method that is called by focusing elements to check if the beam needs to be re-classified as unfocused.
        Must be called before beam z is adjusted. Also changes z to new values.

        Parameters
        ----------
        new_zx: float
            new horizontal beam radius of curvature
        new_zy: float
            new vertical beam radius of curvature
        """

        if new_zx is None:
            new_zx = self.zx
        if new_zy is None:
            new_zy = self.zy

        # update Rayleigh range
        self.zRx = (8 ** 2 * self.lambda0 * new_zx** 2 / np.pi / (np.max(self.x - self.cx) ** 2) *
                    self.rangeFactor)
        self.zRy = (8 ** 2 * self.lambda0 * new_zy** 2 / np.pi / (np.max(self.y - self.cy) ** 2) *
                    self.rangeFactor)

        # check if beam should change state. This only needs to happen if beam is already focused because if unfocused
        # the beam_prop method will check anyway.
        x_focused = -self.zRx <= new_zx < self.zRx
        y_focused = -self.zRy <= new_zy < self.zRy

        # check if transitioning to unfocused
        if self.focused_x:
            # if it stays focused, we need to modify the phase directly
            if x_focused:
                self.wave *= np.exp(1j * np.pi / self.lambda0 * (self.x - self.cx)**2 * (1/new_zx - 1/self.zx))
            else:
                print('x becomes unfocused')
                self.wave *= np.exp(-1j * np.pi / self.lambda0 / self.zx * (self.x - self.cx) ** 2)
                self.focused_x = False
        if self.focused_y:
            if y_focused:
                self.wave *= np.exp(1j * np.pi / self.lambda0 * (self.y - self.cy) ** 2 * (1 / new_zy - 1 / self.zy))
            else:
                print('y becomes unfocused')
                self.wave *= np.exp(-1j * np.pi / self.lambda0 / self.zy * (self.y - self.cy) ** 2)
                self.focused_y = False

        # update beam z
        self.zx = new_zx
        self.zy = new_zy

    def beam_prop(self, dz, dz_progress=0, index=0):
        """
        Method that handles beam propagation.
        :param dz: float
            Distance to propagate
        :param dz_progress: float
            Distance already propagated. This is an internal parameter used for recursive calls to the method.
        :param index: float
            Number of calls to the method. This is an internal parameter used for recursive calls to the method.
        :return beam.wave: (N,M) ndarray
            Returns the complex-valued beam array.
        """

        # print the index for how many times the method has been called during this propagation
        # print(index)

        # check if we've made it all the way and return the beam amplitude/phase if we're done
        if np.abs(dz_progress - dz) < 100 * self.lambda0:
            return self.wave
        else:

            # if we're not focused and this is the first step, calculate current Rayleigh length estimate
            if not self.focused_x and index == 0:
                self.zRx = (8 ** 2 * self.lambda0 * (-self.zx) ** 2 / np.pi / (np.max(self.x - self.cx) ** 2) *
                            self.rangeFactor)
            # if we're not focused and this is the first step, calculate current Rayleigh length estimate
            if not self.focused_y and index == 0:
                self.zRy = (8 ** 2 * self.lambda0 * (-self.zy) ** 2 / np.pi / (np.max(self.y - self.cy) ** 2) *
                            self.rangeFactor)

            # print current focal ranges
            print('zRx: %.2f microns' % (self.zRx*1e6))
            print('zRy: %.2f microns' % (self.zRy*1e6))

            # calculate remaining distance and print it
            dz_remaining = dz - dz_progress
            print('remaining distance: %.2f microns' % (dz_remaining*1e6))

            # calculate what the radius of curvature will be when we're finished
            zx_goal = self.zx + dz_remaining
            zy_goal = self.zy + dz_remaining
            print('goal for zx: %.2f microns' % (zx_goal*1e6))
            print('current zx: %.2f microns' % (self.zx*1e6))
            print('goal for zy: %.2f microns' % (zy_goal*1e6))
            print('current zy: %.2f microns' % (self.zy*1e6))

            # check if we end up inside the focus range?
            x_focused = -self.zRx <= zx_goal < self.zRx
            y_focused = -self.zRy <= zy_goal < self.zRy

            # the two simplest cases first
            # ----------------------------
            # propagation inside focus region
            if self.focused_x and self.focused_y and x_focused and y_focused:
                # normal, unscaled propagation so all three parameters are the same
                self.propagation(dz_remaining, dz_remaining, dz_remaining)
                # staying in focus region, update the parameters and we're done
                self.update_parameters(dz_remaining)

                # return the wave
                return self.wave

            # propagation outside focus region
            elif ((not self.focused_x) and (not self.focused_y) and
                  (not x_focused) and (not y_focused)):

                # calculate Fresnel scaling magnification
                mag_x = (self.zx + dz_remaining) / self.zx
                mag_y = (self.zy + dz_remaining) / self.zy

                # calculate effective distance to propagate
                z_eff_x = dz_remaining / mag_x
                z_eff_y = dz_remaining / mag_y

                # scaled propagation
                self.propagation(dz_remaining, z_eff_x, z_eff_y)

                # rescale coordinates based on magnification
                self.rescale_x_noshift(mag_x)
                self.rescale_y_noshift(mag_y)

                # update beam center and radii of curvature
                self.update_parameters(dz_remaining)

                # return the wave
                return self.wave

            # cases where multiple propagation steps are needed
            else:

                # check if x is focused and whether it will stay focused
                # -------------
                # check x first
                if self.focused_x:
                    # will x stay focused?
                    if x_focused:
                        # if so, x doesn't limit propagation distance
                        x_prop_limit = dz_remaining
                    else:
                        # if not, the limit is the downstream boundary of the focal region
                        x_prop_limit = self.zRx - self.zx
                # x is not currently focused
                else:
                    # will x become focused?
                    if x_focused:
                        # if it will become focused, the limit is the upstream boundary of the focal region
                        x_prop_limit = -self.zx - self.zRx
                    # x will stay unfocused
                    else:
                        # if so, x doesn't limit propagation distance
                        x_prop_limit = dz_remaining

                # now check y
                if self.focused_y:
                    # will x stay focused?
                    if y_focused:
                        # if so, y doesn't limit propagation distance
                        y_prop_limit = dz_remaining
                    else:
                        # if not, the limit is the downstream boundary of the focal region
                        y_prop_limit = self.zRy - self.zy
                # y is not currently focused
                else:
                    # will y become focused?
                    if y_focused:
                        # if it will become focused, the limit is the upstream boundary of the focal region
                        y_prop_limit = -self.zy - self.zRy
                    # x will stay unfocused
                    else:
                        # if so, y doesn't limit propagation distance
                        y_prop_limit = dz_remaining

                # distance to propagate during this step. Pick the more restrictive case.
                prop_step = np.min([x_prop_limit, y_prop_limit])

                # print the current step size
                # print('current step size: %.2f microns' % (prop_step*1e6))

                # radii of curvature at the end of this propagation step
                zx_goal = self.zx + prop_step
                zy_goal = self.zy + prop_step

                # do we end up inside the focus range?
                x_focused = -self.zRx <= zx_goal < self.zRx
                y_focused = -self.zRy <= zy_goal < self.zRy

                # initialize magnification to one
                mag_x = 1
                mag_y = 1

                # initialize boolean flags to check if we're making a transition into the focal region
                transition_to_x_focus = False
                transition_to_y_focus = False

                # initialize boolean flags to check if we're making a transition out of the focal region
                transition_to_x_defocus = False
                transition_to_y_defocus = False

                # check if x will remain focused
                if self.focused_x:
                    # x remains focused
                    if x_focused:
                        # no transition
                        print('x remains focused')
                        # no magnification
                        z_eff_x = prop_step
                    # x becomes unfocused
                    else:
                        print('x becomes unfocused')
                        # no magnification
                        z_eff_x = prop_step
                        # after propagation x will transition out of the focal region
                        transition_to_x_defocus = True

                # x doesn't start out focused
                else:
                    # calculate magnification
                    mag_x = (self.zx + prop_step) / self.zx
                    # calculate effective propagation distance
                    z_eff_x = prop_step / mag_x
                    # check if there will be a transition
                    if x_focused:
                        # after propagation x will transition into the focal region
                        print('x becomes focused')
                        transition_to_x_focus = True
                    else:
                        # no transition
                        print('x stays unfocused')

                # check if y will remain focused
                if self.focused_y:
                    # y remains focused
                    if y_focused:
                        # no transition
                        print('y remains focused')
                        # no magnification
                        z_eff_y = prop_step
                    # y becomes unfocused
                    else:
                        print('y becomes unfocused')
                        # no magnification
                        z_eff_y = prop_step
                        # after propagation y will transition out of the focal region
                        transition_to_y_defocus = True

                # y doesn't start out focused
                else:
                    # calculate magnification
                    mag_y = (self.zy + prop_step) / self.zy
                    # calculate effective propagation distance
                    z_eff_y = prop_step / mag_y
                    # check if there will be a transition
                    if y_focused:
                        # after propagation y will transition into the focal region
                        print('y becomes focused')
                        transition_to_y_focus = True
                    else:
                        # no transition
                        print('y stays unfocused')

                # general propagation step, may or may not be Fresnel scaling
                self.propagation(prop_step, z_eff_x, z_eff_y)

                # rescale just in case. If propagation is unscaled mag_x and mag_y still equal one.
                self.rescale_x_noshift(mag_x)
                self.rescale_y_noshift(mag_y)

                # update beam geometric parameters based on propagation distance
                self.update_parameters(prop_step)

                # check if we need to add phase near focus, and alter the focus state
                if transition_to_x_focus:
                    self.wave *= np.exp(1j * np.pi / self.lambda0 / self.zx * (self.x-self.cx) ** 2)
                    self.focused_x = True
                if transition_to_x_defocus:
                    self.wave *= np.exp(-1j * np.pi / self.lambda0 / self.zx * (self.x-self.cx) ** 2)
                    self.focused_x = False
                # check if we need to add phase near focus, and alter the focus state
                if transition_to_y_focus:
                    self.wave *= np.exp(1j * np.pi / self.lambda0 / self.zy * (self.y-self.cy) ** 2)
                    self.focused_y = True
                if transition_to_y_defocus:
                    self.wave *= np.exp(-1j * np.pi / self.lambda0 / self.zy * (self.y-self.cy) ** 2)
                    self.focused_y = False

                # recursively call this method until we've reached the original goal (dz)
                self.beam_prop(dz, dz_progress=(dz_progress + prop_step), index=(index + 1))

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
        if self.x[0, 0] > self.x[0, 1]:
            # calculate pixel size based on first two entries in x
            self.dx = self.x[0, 0] - self.x[0, 1]
        else:
            # calculate pixel size based on first two entries in x
            self.dx = self.x[0, 1]-self.x[0, 0]
        # check if y coordinates are reversed before calculating pixel size
        if self.y[0, 0] > self.y[1, 0]:
            # calculate pixel size based on first two entries in y
            self.dy = self.y[0, 0] - self.y[1, 0]
        else:
            # calculate pixel size based on first two entries in y
            self.dy = self.y[1, 0] - self.y[0, 0]

        # recalculate maximum spatial frequencies based on new pixel sizes
        fx_max = 1.0 / (2 * self.dx)
        fy_max = 1.0 / (2 * self.dy)
        # calculate spatial frequency pixel size
        dfx = fx_max / self.M
        dfy = fy_max / self.N
        # spatial frequency coordinates
        fx = np.linspace(-fx_max, fx_max - dfx, self.M)
        fy = np.linspace(-fy_max, fy_max - dfy, self.N)
        # make the grids
        self.fx, self.fy = np.meshgrid(fx, fy)


class Pulse:
    """
    Class to represent a collection of beams within a pulse structure.
    """

    def __init__(self, beam_params=None, bandwidth=None, N=10):
        """
        Create a Pulse object
        :param beam_params: same parameters as given for Beam
        :param bandwidth: float
            pulse bandwidth in eV (FWHM)
        :param N: int
            number of independent photon energies to propagate
        """
        # set some attributes
        self.N = N
        self.bandwidth = bandwidth
        self.E0 = beam_params['photonEnergy']
        sigma = self.bandwidth / 2.355

        # define pulse envelope (twice the FWHM)
        self.energy = np.linspace(-self.bandwidth, self.bandwidth, N)
        self.envelope = np.exp(-(self.energy-self.E0)**2/(2 * sigma**2))

    def propagate(self, beamline=None, screens=None):
        pass

        
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

        # check if we're inside this range
        focused_x = -self.zRx*factor <= self.z0x < self.zRx*factor
        focused_y = -self.zRy*factor <= self.z0y < self.zRy*factor
        # print(self.zRx)
        # print(self.zRy)

        # need to modify this in the case where beam starts out "in focus"
        if self.dx is None:
            # if not, define array to cover 4x FWHM
            fwhm_x = 1.18*self.wx
            fwhm_y = 1.18*self.wy

            # set field of view
            if focused_x:
                # set it so that it will be 8 times the FWHM at the boundary of the focal range
                FOV_x = np.abs(self.zRx * factor/self.z0x) * 8 * fwhm_x
                print('x is focused')
            else:
                # if out of focus, just set to 8 times the FWHM
                FOV_x = 8*fwhm_x
            if focused_y:
                # set it so that it will be 8 times the FWHM at the boundary of the focal range
                print('y is focused')
                FOV_y = np.abs(self.zRy * factor/self.z0y) * 8 * fwhm_y
            else:
                # if out of focus, just set to 8 times the FWHM
                FOV_y = 8*fwhm_y

            # calculate the resulting pixel size
            self.dx = FOV_x/self.N
            self.dy = FOV_y/self.N

        # coordinates
        x = np.linspace(-self.N / 2, self.N / 2 - 1, self.N) * self.dx
        y = np.linspace(-self.N / 2, self.N / 2 - 1, self.N) * self.dy
        self.x, self.y = np.meshgrid(x, y)

        # beam amplitude
        self.source = np.exp(-((self.x / self.wx) ** 2))*np.exp(-((self.y / self.wy) ** 2))
