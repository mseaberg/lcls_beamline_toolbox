"""
beam1d module

Part of the xraybeamline2d package.

Currently implements the following classes:
Beam: handles 2 x 1D beam propagation
GaussianSource: helper class for initializing a Beam object

All distances/lengths are in meters, spatial frequencies in 1/meters, angles in radians, energies in eV,
unless otherwise indicated.
"""
import numpy as np
import matplotlib.pyplot as plt
from .util import Util
from skimage.restoration import unwrap_phase
import scipy.spatial.transform as transform
import scipy.optimize as optimization


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
            z_source: float
                global z coordinate of undulator exit
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

        # check if rangeFactor was provided, or default to 8.
        if 'scaleFactor' in beam_params.keys():
            self.scaleFactor = beam_params['scaleFactor']
        else:
            # default value is 8
            self.scaleFactor = 8
            # add this to the dictionary
            beam_params['scaleFactor'] = self.scaleFactor

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
            # multiply rayleigh range by rangeFactor, and multiply by factor to make it consistent with
            # other calculations
            self.zRx = b1.zRx * self.rangeFactor * (2/1.18)**2
            self.zRy = b1.zRy * self.rangeFactor * (2/1.18)**2

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

        self.global_x = np.copy(self.cx)
        self.global_y = np.copy(self.cy)
        # initialize global z
        self.z_source = beam_params['z_source']
        self.global_z = beam_params['z_source'] + (self.zx + self.zy) / 2

        # initialize global angles
        self.global_azimuth = np.copy(self.ax)
        self.global_elevation = np.copy(self.ay)

        # initialize group delay
        self.group_delay = 0.0


        # calculate spatial frequencies at initial plane
        fx_max = 1.0 / (2.0 * self.dx)
        fy_max = 1.0 / (2.0 * self.dy)
        dfx = fx_max / self.M
        fx = np.linspace(-fx_max, fx_max - dfx, self.M)
        dfy = fy_max / self.N
        fy = np.linspace(-fy_max, fy_max - dfy, self.N)
        self.fx = fx
        self.fy = fy

        # check if we're inside this range and initialize focus attribute
        self.focused_x = -self.zRx <= self.zx < self.zRx
        self.focused_y = -self.zRy <= self.zy < self.zRy
        # print(self.zRx)
        # print(self.zRy)

        # if we're already inside the focal range, we need to multiply by quadratic phase in order for
        # propagation to work properly.
        if self.focused_x:
            self.wavex *= np.exp(1j * np.pi / self.lambda0 / self.zx * (self.x - self.cx)**2)
        if self.focused_y:
            self.wavey *= np.exp(1j * np.pi / self.lambda0 / self.zy * (self.y - self.cy)**2)

        # set beam parameters as attribute
        self.beam_params = beam_params

        # define beam unit vectors in LCLS coordinates
        self.xhat = np.array([1,0,0])
        self.yhat = np.array([0,1,0])
        self.zhat = np.array([0,0,1])

        # define LCLS unit vectors
        self.x_nom = np.copy(self.xhat)
        self.y_nom = np.copy(self.yhat)
        self.z_nom = np.copy(self.zhat)


    def reinitialize(self, dz):
        self.beam_params['z0x'] = dz
        self.beam_params['z0y'] = dz
        self.__init__(beam_params=self.beam_params)

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

        print(self.zx)
        print(self.zy)
        self.zx = self.zx + dz
        self.zy = self.zy + dz

        print(self.zx)
        print(self.zy)

        # update global positions
        k_beam = self.get_k()
        # tan(alpha) = k_beam[0]/k_beam[2]
        # x = k_beam[0]/k_beam[2] * dz * k_beam[2]
        # z = dz * np.cos(alpha). cos(alpha) = k[2]
        # self.global_z += k_beam[2] * dz
        self.global_x += k_beam[0] * dz
        self.global_y += k_beam[1] * dz
        self.global_z += k_beam[2] * dz

    def beam_offset(self, x_offset=0, y_offset=0):
        self.global_x += x_offset
        self.global_y += y_offset

    def rotate_nominal(self, delta_elevation=0, delta_azimuth=0):

        # an "elevation" rotation corresponds to a rotation about the xhat unit vector
        r1 = transform.Rotation.from_rotvec(-self.xhat * delta_elevation)
        Rx = r1.as_matrix()
        self.xhat = np.matmul(Rx, self.xhat)
        self.yhat = np.matmul(Rx, self.yhat)
        self.zhat = np.matmul(Rx, self.zhat)

        # an azimuth rotation corresponds to a rotation about the yhat unit vector
        r2 = transform.Rotation.from_rotvec(self.yhat * delta_azimuth)
        Ry = r2.as_matrix()
        self.xhat = np.matmul(Ry, self.xhat)
        self.yhat = np.matmul(Ry, self.yhat)
        self.zhat = np.matmul(Ry, self.zhat)

        self.global_elevation += delta_elevation
        self.global_azimuth += delta_azimuth

    def rotate_beam(self, delta_ax=0, delta_ay=0):
        # first adjust "local" angles. Going to keep this the same as before
        self.ax += delta_ax
        self.ay += delta_ay

        self.rotate_nominal(delta_elevation=delta_ay, delta_azimuth=delta_ax)

    def get_k(self):
        # x = np.array([1, 0, 0], dtype=float)
        # y = np.array([0, 1, 0], dtype=float)
        # z = np.array([0, 0, 1], dtype=float)
        #
        # r1 = transform.Rotation.from_rotvec(-x * self.global_elevation)
        # Rx = r1.as_matrix()
        # x = np.matmul(Rx, x)
        # y = np.matmul(Rx, y)
        # z = np.matmul(Rx, z)
        #
        # r2 = transform.Rotation.from_rotvec(y * self.global_azimuth)
        # Ry = r2.as_matrix()
        # x = np.matmul(Ry, x)
        # y = np.matmul(Ry, y)
        # z = np.matmul(Ry, z)

        # beam points in z direction
        # k = z
        k = np.copy(self.zhat)
        return k

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

        ### putting in kind of a terrible hack where I account for half of the constant propagation phase
        ### for both x and y, so that they add up to the correct thing when horizontal and vertical phases
        ### are multiplied together. This causes 1D and 2D codes to agree...
        phi_prop_x = ( - self.k0 / 2 *
                    (self.lambda0 * self.fx) ** 2 * dz_x)

        phi_prop_y = ( - self.k0 / 2 *
                    (self.lambda0 * self.fy) ** 2 * dz_y)

        # update group delay
        self.group_delay += dz_real/3e8

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
        if new_zx is not None:
            # if self.focused_x:
            #     xWidth = np.abs(self.x[0] - self.x[-1])
            #
            #     # need to handle zx=0 appropriately
            #     beamRef = xWidth / self.scaleFactor
            #     w0 = np.sqrt(self.lambda0*self.zRx/np.pi)
            #     w = w0 * np.sqrt(1 + (self.zx/self.zRx)**2)
            #     xWidth *= np.abs(self.zx / (self.zRx * self.rangeFactor))
            # else:
            #     xWidth = np.abs(self.x[0] - self.x[-1])
            xWidth = np.abs(self.x[0] - self.x[-1])

            # self.zRx = (self.scaleFactor ** 2 * self.lambda0 * (-self.zx) ** 2 / np.pi / ((xWidth / 2) ** 2) *
            #             self.rangeFactor)
            self.zRx = (self.scaleFactor ** 2 * self.lambda0 * (new_zx) ** 2 / np.pi / ((xWidth / 2) ** 2) *
                                     self.rangeFactor)
            # self.zRx = (8 ** 2 * self.lambda0 * new_zx** 2 / np.pi / (np.max(self.x - self.cx) ** 2) *
            #         self.rangeFactor)
        if new_zy is not None:
            # if self.focused_y:
            #     yWidth = np.abs(self.y[0] - self.y[-1])
            #     yWidth *= np.abs(self.zy / (self.zRy * self.rangeFactor))
            # else:
            #     yWidth = np.abs(self.y[0] - self.y[-1])
            yWidth = np.abs(self.y[0] - self.y[-1])

            # self.zRy = (self.scaleFactor ** 2 * self.lambda0 * (-self.zy) ** 2 / np.pi / ((yWidth / 2) ** 2) *
            #             self.rangeFactor)
            self.zRy = (self.scaleFactor ** 2 * self.lambda0 * (new_zy) ** 2 / np.pi / ((yWidth / 2) ** 2) *
                                     self.rangeFactor)
            # self.zRy = (8 ** 2 * self.lambda0 * new_zy** 2 / np.pi / (np.max(self.y - self.cy) ** 2) *
            #         self.rangeFactor)

        # check if beam should change state. This only needs to happen if beam is already focused because if unfocused
        # the beam_prop method will check anyway.
        x_focused = -self.zRx <= new_zx < self.zRx
        y_focused = -self.zRy <= new_zy < self.zRy
        print('zRx: %.2e' % self.zRx)
        print('zRy: %.2e' % self.zRy)

        # check if transitioning to unfocused
        if self.focused_x:
            # if it stays focused, we need to modify the phase directly
            if x_focused:
                self.wavex *= np.exp(1j * np.pi / self.lambda0 * (self.x - self.cx)**2 * (1/new_zx - 1/self.zx))
            else:
                print('x becomes unfocused')
                self.wavex *= np.exp(-1j * np.pi / self.lambda0 / self.zx * (self.x - self.cx) ** 2)
                self.focused_x = False
        if self.focused_y:
            if y_focused:
                self.wavey *= np.exp(1j * np.pi / self.lambda0 * (self.y - self.cy) ** 2 * (1 / new_zy - 1 / self.zy))
            else:
                print('y becomes unfocused')
                self.wavey *= np.exp(-1j * np.pi / self.lambda0 / self.zy * (self.y - self.cy) ** 2)
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
        :return beam.wavex: (M,) ndarray
            Returns the complex-valued beam array.
        :return beam.wavey: (M,) ndarray
            Returns the complex-valued beam array.
        """

        # print the index for how many times the method has been called during this propagation
        # print(index)

        # check if we've made it all the way and return the beam amplitude/phase if we're done
        if np.abs(dz_progress - dz) < 100 * self.lambda0:
            return self.wavex, self.wavey
        else:

            # if we're not focused and this is the first step, calculate current Rayleigh length estimate
            if not self.focused_x and index == 0:

                xWidth = np.abs(self.x[0] - self.x[-1])
                self.zRx = (self.scaleFactor ** 2 * self.lambda0 * (-self.zx) ** 2 / np.pi / ((xWidth/2) ** 2) *
                            self.rangeFactor)
                # print('FOVx: ' + str(np.max(self.x - self.cx)))
            # if we're not focused and this is the first step, calculate current Rayleigh length estimate
            if not self.focused_y and index == 0:
                yWidth = np.abs(self.y[0] - self.y[-1])
                self.zRy = (self.scaleFactor ** 2 * self.lambda0 * (-self.zy) ** 2 / np.pi / ((yWidth/2) ** 2) *
                            self.rangeFactor)
                # print('FOVy: ' + str(np.max(self.y - self.cy)))

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
                print('both stay focused')
                # normal, unscaled propagation so all three parameters are the same
                self.propagation(dz_remaining, dz_remaining, dz_remaining)
                # staying in focus region, update the parameters and we're done
                self.update_parameters(dz_remaining)

                # return the wave
                return self.wavex, self.wavey

            # propagation outside focus region
            elif ((not self.focused_x) and (not self.focused_y) and
                  (not x_focused) and (not y_focused)):
                print('both stay unfocused')

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
                return self.wavex, self.wavey

            # cases where multiple propagation steps are needed
            else:
                print('multiple propagation steps needed')

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
                prop_cases = [x_prop_limit, y_prop_limit]
                prop_choice = np.argmin(np.abs(prop_cases))
                prop_step = prop_cases[prop_choice]
                # prop_step = np.min([np.abs(x_prop_limit), np.abs(y_prop_limit)])

                # print the current step size
                print('current step size: %.2f microns' % (prop_step*1e6))

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

                # print('mag_y: ' + str(mag_y))

                # general propagation step, may or may not be Fresnel scaling
                self.propagation(prop_step, z_eff_x, z_eff_y)

                # rescale just in case. If propagation is unscaled mag_x and mag_y still equal one.
                self.rescale_x_noshift(mag_x)
                self.rescale_y_noshift(mag_y)

                # update beam geometric parameters based on propagation distance
                self.update_parameters(prop_step)

                # check if we need to add phase near focus, and alter the focus state
                if transition_to_x_focus:
                    self.wavex *= np.exp(1j * np.pi / self.lambda0 / self.zx * (self.x-self.cx) ** 2)
                    self.focused_x = True
                if transition_to_x_defocus:
                    self.wavex *= np.exp(-1j * np.pi / self.lambda0 / self.zx * (self.x-self.cx) ** 2)
                    self.focused_x = False
                # check if we need to add phase near focus, and alter the focus state
                if transition_to_y_focus:
                    self.wavey *= np.exp(1j * np.pi / self.lambda0 / self.zy * (self.y-self.cy) ** 2)
                    self.focused_y = True
                if transition_to_y_defocus:
                    self.wavey *= np.exp(-1j * np.pi / self.lambda0 / self.zy * (self.y-self.cy) ** 2)
                    self.focused_y = False

                # recursively call this method until we've reached the original goal (dz)
                return self.beam_prop(dz, dz_progress=(dz_progress + prop_step), index=(index + 1))

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

    def asymmetry_x(self, factor):
        """
        Method that rescales coordinates and also distance to focus, for asymmetric reflections
        Parameters
        ----------
        factor: float
            scaling factor

        Returns
        -------

        """

        # scale coordinates
        self.rescale_x(factor)
        # rescale distance to focus
        self.zx *= factor**2

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

    def asymmetry_y(self, factor):
        """
        Method that rescales coordinates and also distance to focus, for asymmetric reflections
        Parameters
        ----------
        factor: float
            scaling factor

        Returns
        -------

        """
        # rescale coordinates
        self.rescale_y(factor)
        # rescale distance to focus
        self.zy *= factor**2

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


class Pulse:
    """
    Class to represent a collection of beams within a pulse structure.
    """

    def __init__(self, beam_params=None, tau=None, time_window=None, SASE=False, num_spikes=3, unit_spectrum=False,
                 spectral_width=0, N=0, GDD=0):
        """
        Create a Pulse object
        :param beam_params: same parameters as given for Beam
        :param tau: float
            pulse width (fs)
        :param time_window: float
            full width of time window in fs (related to energy sampling)
        """
        # set some attributes
        self.beam_params = beam_params
        self.tau = tau
        self.time_window = time_window
        self.num_spikes = num_spikes
        self.E0 = beam_params['photonEnergy']

        # ----- energy range
        # 1/e^2 in intensity bandwidth (radius) for transform-limited pulse
        # hbar in eV*fs
        hbar = 0.6582

        if unit_spectrum:
            E_range = spectral_width
            # total frequency range in petaHz (energy divided by Planck's constant (in eV * fs))
            f_range = E_range / 4.136

            # time resolution corresponding to full energy range (in fs)
            self.deltaT = 1 / f_range

            self.N = N
            # define pulse energies and envelope
            self.energy = np.linspace(-E_range/2, E_range/2, self.N) + self.E0
            self.envelope = np.ones(self.N)

            # frequencies
            self.f = self.energy / 4.136
            self.f0 = self.E0 / 4.136

        else:
            self.bandwidth = 2 * np.sqrt(2) * hbar * np.sqrt(np.log(2)) / self.tau

            # define energy range 6 times the bandwidth
            if SASE:
                E_range = 6 * self.bandwidth * self.num_spikes
            else:
                E_range = 6 * self.bandwidth

            # total frequency range in petaHz (energy divided by Planck's constant (in eV * fs))
            f_range = E_range / 4.136

            # time resolution corresponding to full energy range (in fs)
            self.deltaT = 1 / f_range

            # calculate number of samples needed
            self.N = int(self.time_window / self.deltaT)

            # define pulse energies and envelope
            self.energy = np.linspace(-E_range/2, E_range/2, self.N) + self.E0

            # frequencies
            self.f = self.energy / 4.136
            self.f0 = self.E0 / 4.136

            # add in optional spectral chirp
            self.spectral_phase = np.exp(1j * GDD / 2 * (2 * np.pi) ** 2 * (self.f - self.f0) ** 2)

            if SASE:
                self.envelope = self.generate_SASE()
            else:
                self.envelope = np.sqrt(np.exp(-(self.energy-self.E0) ** 2 * tau ** 2 / 4 / hbar ** 2 / np.log(2)))
            self.envelope = self.envelope.astype(complex) * self.spectral_phase

        self.pulse = np.fft.fftshift(np.fft.fft(np.fft.fftshift(self.envelope)))
        # calculate wavelengths
        self.wavelength = 1239.8/self.energy*1e-9

        # total energy range
        E_range = np.max(self.energy) - np.min(self.energy)
        self.dE = E_range / self.N

        # time axis in fs
        self.t_axis = np.linspace(-self.N/2, self.N/2-1, self.N) * self.deltaT

        # initialize energy stacks with dictionary. Keys are profile monitor names
        self.energy_stacks = {}

        # initialize time stacks with dictionary. Keys are profile monitor names
        self.time_stacks = {}

        # initialize list of screens
        self.screens = []

        # initialize coordinates for profile monitors
        self.x = {}
        self.y = {}

        # initialize 2D coordinates for profile monitors
        self.xx = {}
        self.yy = {}

        # initialize quadratic phase
        self.qx = {}
        self.qy = {}

        # initialize beam center
        self.cx = {}
        self.cy = {}

        # initialize group delay dictionary
        self.delay = {}

    def generate_SASE(self):
        spike_centers = (.5-np.random.rand(self.num_spikes))*self.num_spikes*self.bandwidth*2
        spike_intensity = np.random.rand(self.num_spikes)
        spike_linphase = (.5-np.random.rand(self.num_spikes))*2*self.tau
        spike_offsetphase = np.random.rand(self.num_spikes)*2*np.pi
        hbar = 0.6582

        envelope = np.zeros_like(self.energy, dtype=complex)
        sigma = 1/(self.tau ** 2 / 4 / hbar ** 2 / np.log(2))

        for i in range(self.num_spikes):
            phase = np.exp(1j*spike_offsetphase[i])*np.exp(1j*2*np.pi*spike_linphase[i]*self.f)
            envelope += (spike_intensity[i]*
                             np.sqrt(np.exp(-(self.energy - self.E0-spike_centers[i]) ** 2 / sigma))*phase)

        return envelope

    @staticmethod
    def beam_analysis(x, y, line_x, line_y, threshold=0.1):
        """
        Method for analyzing image of the beam.
        :param line_x: (N,) ndarray
            Horizontal lineout. Could be summed across full image or from an ROI.
        :param line_y: (N,) ndarray
            Vertical lineout. Could be summed across full image or from an ROI.
        :return cx: float
            Calculated horizontal centroid (m)
        :return cy: float
            Calculated vertical centroid (m)
        :return fwhm_x: float
            Calculated horizontal FWHM (m). Based on Gaussian fit (or calculated from second moment if fit fails).
        :return fwhm_y: float
            Calculated vertical FWHM (m). Based on Gaussian fit (or calculated from second moment if fit fails).
        :return fwx_guess: float
            Calculated horizontal FWHM (m) based on calculation of second moment.
        :return fwy_guess: float
            Calculated vertical FWHM (m) based on calculation of second moment.
        """

        amp_x = np.max(line_x) - np.min(line_x)
        amp_y = np.max(line_y) - np.min(line_y)

        # normalize lineouts
        if np.max(line_x) > 0:
            line_x -= np.min(line_x)
            line_x = line_x / np.max(line_x)

        if np.max(line_y) > 0:
            line_y -= np.min(line_y)
            line_y = line_y / np.max(line_y)

        # set 20% threshold
        thresh_x = np.max(line_x) * threshold
        thresh_y = np.max(line_y) * threshold
        # subtract threshold and set everything below to zero
        norm_x = line_x - thresh_x
        norm_x[norm_x < 0] = 0
        # re-normalize

        if np.max(norm_x) > 0:
            norm_x = norm_x / np.max(norm_x)

        # subtract threshold and set everything below to zero
        norm_y = line_y - thresh_y
        norm_y[norm_y < 0] = 0
        # re-normalize
        if np.max(norm_y) > 0:
            norm_y = norm_y / np.max(norm_y)

        # calculate centroids

        if np.sum(norm_x) > 0:
            cx = np.sum(norm_x * x) / np.sum(norm_x)
            # calculate second moments. Converted to microns to help with fitting
            sx = np.sqrt(np.sum(norm_x * (x - cx) ** 2) / np.sum(norm_x)) * 1e6

        else:
            cx = 0
            sx = 0
        if np.sum(norm_y) > 0:
            cy = np.sum(norm_y * y) / np.sum(norm_y)
            # calculate second moments. Converted to microns to help with fitting
            sy = np.sqrt(np.sum(norm_y * (y - cy) ** 2) / np.sum(norm_y)) * 1e6

        else:
            cy = 0
            sy = 0

        # conversion factor from sigma to fwhm
        fwx_guess = sx * 2.355
        fwy_guess = sy * 2.355

        # initial guess for Gaussian fit
        guessx = [cx * 1e6, sx]
        guessy = [cy * 1e6, sy]

        fit_validity = 1

        # Gaussian fitting. Using try/except to deal with any fitting errors
        try:
            # only fit in the region where we have signal
            mask = line_x > .1
            # Gaussian fit using Scipy curve_fit. Using only data that has >10% of the max
            px, pcovx = optimization.curve_fit(Util.fit_gaussian, x[mask] * 1e6, line_x[mask], p0=guessx)
            # set sx to sigma from the fit if successful.
            sx = px[1]
        except ValueError:
            fit_validity = 0
            print('Some of the data contained NaNs or options were incompatible. Using second moment for width.')
        except RuntimeError:
            fit_validity = 0
            print('Least squares minimization failed. Using second moment for width.')

        try:
            # only fit in the region where we have signal
            mask = line_y > .1
            # Gaussian fit using Scipy curve_fit. Using only data that has >10% of the max
            py, pcovy = optimization.curve_fit(Util.fit_gaussian, y[mask] * 1e6, line_y[mask], p0=guessy)
            # set sy to sigma from the fit if successful.
            sy = py[1]
        except ValueError:
            fit_validity = 0
            print('Some of the data contained NaNs or options were incompatible. Using second moment for width.')
        except RuntimeError:
            fit_validity = 0
            print('Least squares minimization failed. Using second moment for width.')

        # conversion factor from sigma to FWHM. Also convert back to meters.
        fwhm_x = sx * 2.355 / 1e6
        fwhm_y = sy * 2.355 / 1e6

        # check validity
        validity = ((amp_x > 0) and (amp_y > 0) and fit_validity and
                    (fwhm_x < np.max(2 * x)) and (fwhm_y < np.max(2 * y)))

        return cx, cy, fwhm_x, fwhm_y, fwx_guess, fwy_guess

    def propagate(self, beamline=None, screen_names=None):
        """
        Method for propagating a pulse through a beamline
        Parameters
        ----------
        beamline: Beamline
            Collection of beamline devices to propagate through
        screen_names: list of strings
            Locations to evaluate pulse. Must correspond to profile monitor names in the beamline
        Returns
        -------
        None
        """

        self.screens = screen_names

        # add screens to energy stacks
        for screen in screen_names:
            screen_obj = getattr(beamline, screen)
            Ns = screen_obj.N
            self.x[screen] = screen_obj.x
            self.y[screen] = screen_obj.y
            self.xx[screen], self.yy[screen] = np.meshgrid(self.x[screen], self.y[screen])
            self.energy_stacks[screen] = np.zeros((Ns, Ns, self.N), dtype=complex)
            self.qx[screen] = np.zeros(self.N)
            self.qy[screen] = np.zeros(self.N)
            self.cx[screen] = np.zeros(self.N)
            self.cy[screen] = np.zeros(self.N)
            self.delay[screen] = np.zeros(self.N)

        # loop through beams in the pulse
        for num, energy in enumerate(self.energy):
            # define beam for current energy
            self.beam_params['photonEnergy'] = energy
            b1 = Beam(beam_params=self.beam_params)
            beamline.propagate_beamline(b1)

            for screen in screen_names:
                # put current photon energy into energy stack, multiply by spectral envelope
                screen_obj = getattr(beamline, screen)
                energy_slice, delay, zx, zy, cx, cy = screen_obj.complex_beam()
                self.energy_stacks[screen][:, :, num] = energy_slice * self.envelope[num]
                if zx != 0:
                    self.qx[screen][num] = 1/zx
                if zy != 0:
                    self.qy[screen][num] = 1/zy
                self.cx[screen][num] = cx
                self.cy[screen][num] = cy
                self.delay[screen][num] = delay
                # self.energy_stacks[screen][:, :, num] = screen_obj.complex_beam() * self.envelope[num]

        # convert to time domain
        for screen in screen_names:
            # deal with quadratic phase
            # subtract mean
            qx_mean = np.mean(self.qx[screen])
            qy_mean = np.mean(self.qy[screen])

            for num in range(self.N):
                qx = self.qx[screen][num]
                qy = self.qy[screen][num]
                cx = self.cx[screen][num]
                cy = self.cy[screen][num]
                # subtract off mean quadratic phase
                # x_phase = np.pi/self.wavelength[num]*(qx - qx_mean)*(self.xx[screen]-cx)**2
                # y_phase = np.pi/self.wavelength[num]*(qy - qy_mean)*(self.yy[screen]-cy)**2
                x_phase = np.pi / self.wavelength[num] * (qx) * (self.xx[screen] - cx) ** 2
                x_phase -= np.pi / self.wavelength[num] * qx_mean * self.xx[screen]**2
                y_phase = np.pi/self.wavelength[num]*(qy)*(self.yy[screen]-cy)**2
                y_phase -= np.pi / self.wavelength[num] * qy_mean * self.yy[screen] ** 2
                self.energy_stacks[screen][:, :, num] *= np.exp(1j*(x_phase+y_phase))

            omega = 2*np.pi*self.f*1e15
            omega0 = 2*np.pi*self.f0*1e15
            delay = self.delay[screen]-np.mean(self.delay[screen])
            p_delay = np.polyfit(omega-omega0,delay,4)
            p_phase = np.polyint(p_delay)
            phase = np.polyval(p_phase,omega-omega0)

            self.energy_stacks[screen] *= np.exp(1j*phase)


            self.time_stacks[screen] = Pulse.energy_to_time(self.energy_stacks[screen])

    @staticmethod
    def energy_to_time(energy_stack):
        """
        Method to convert from energy to time domain
        Parameters
        ----------
        energy_stack: (N,M,P) complex-valued ndarray
            electric field in the photon energy domain
        Returns
        -------
        time_stack: (N,M,P) complex-valued ndarray
            electric field in the time domain
        """

        # calculate time domain of pulse from energy domain (Fourier transform along the energy axis)
        time_stack = np.fft.fftshift(np.fft.fft(np.fft.fftshift(energy_stack, axes=2), axis=2), axes=2)

        return time_stack

    def add_pulse(self, another_pulse, time_shift):
        """
        Method to combine two pulses. For the moment it is assumed that the two pulses have the same time/energy
        sampling, and have been evaluated at the same screens.
        Parameters
        ----------
        another_pulse: Pulse
            The pulse to add to the current one
        time_shift: float
            relative delay to shift pulses (in fs).

        Returns
        -------
        A new Pulse that is the coherent sum of the two pulses.
        """
        beam_params = self.beam_params
        tau = self.tau
        time_window = self.time_window
        new_pulse = Pulse(beam_params=beam_params, tau=tau, time_window=time_window)

        time_stacks = {}
        energy_stacks = {}
        x = {}
        y = {}
        new_pulse.screens = self.screens.copy()
        new_pulse.xx = self.xx.copy()
        new_pulse.yy = self.yy.copy()

        # print(another_pulse.energy_stacks.keys())

        time_pixels = time_shift/self.deltaT

        energy_phase = np.exp(1j*2*np.pi*time_shift*self.f)

        # convert to time domain
        for screen in self.screens:
            # deal with quadratic phase
            # subtract mean
            qx_mean1 = np.mean(self.qx[screen])
            qx_mean2 = np.mean(another_pulse.qx[screen])
            qy_mean1 = np.mean(self.qy[screen])
            qy_mean2 = np.mean(another_pulse.qy[screen])

            qx_mean = (np.mean(self.qx[screen]) + np.mean(another_pulse.qx[screen]))/2
            qy_mean = (np.mean(self.qy[screen]) + np.mean(another_pulse.qy[screen]))/2

            energy_stacks[screen] = np.zeros_like(self.energy_stacks[screen],dtype=complex)
            x[screen] = self.x[screen]
            y[screen] = self.y[screen]

            new_pulse.qx[screen] = np.zeros(self.N)
            new_pulse.qy[screen] = np.zeros(self.N)

            for num in range(self.N):
                qx = self.qx[screen][num]
                qy = self.qy[screen][num]
                # cx = self.cx[screen][num]
                # cy = self.cy[screen][num]
                # subtract off mean quadratic phase
                # x_phase = np.pi/self.wavelength[num]*(qx - qx_mean)*(self.xx[screen]-cx)**2
                # y_phase = np.pi/self.wavelength[num]*(qy - qy_mean)*(self.yy[screen]-cy)**2
                # x_phase1 = np.pi / self.wavelength[num] * (qx) * (self.xx[screen] - cx) ** 2
                x_phase1 = np.pi / self.wavelength[num] * (qx_mean - qx_mean1) * self.xx[screen] ** 2
                # y_phase1 = np.pi / self.wavelength[num] * (qy) * (self.yy[screen] - cy) ** 2
                y_phase1 = np.pi / self.wavelength[num] * (qy_mean - qy_mean1) * self.yy[screen] ** 2

                qx = another_pulse.qx[screen][num]
                qy = another_pulse.qy[screen][num]
                # cx = another_pulse.cx[screen][num]
                # cy = another_pulse.cy[screen][num]
                # subtract off mean quadratic phase
                # x_phase = np.pi/self.wavelength[num]*(qx - qx_mean)*(self.xx[screen]-cx)**2
                # y_phase = np.pi/self.wavelength[num]*(qy - qy_mean)*(self.yy[screen]-cy)**2
                # x_phase2 = np.pi / self.wavelength[num] * (qx) * (self.xx[screen] - cx) ** 2
                x_phase2 = np.pi / self.wavelength[num] * (qx_mean - qx_mean2) * self.xx[screen] ** 2
                # y_phase2 = np.pi / self.wavelength[num] * (qy) * (self.yy[screen] - cy) ** 2
                y_phase2 = np.pi / self.wavelength[num] * (qy_mean - qy_mean2) * self.yy[screen] ** 2
                energy_stacks[screen][:, :, num] = (self.energy_stacks[screen][:,:,num] *
                                                    np.exp(1j * (x_phase1 + y_phase1))*energy_phase[num] +
                                                    another_pulse.energy_stacks[screen][:,:,num] *
                                                    np.exp(1j*(x_phase2 + y_phase2)))

                new_pulse.qx[screen][num] = (self.qx[screen][num] + another_pulse.qx[screen][num])/2
                new_pulse.qy[screen][num] = (self.qy[screen][num] + another_pulse.qy[screen][num])/2

            time_stacks[screen] = Pulse.energy_to_time(energy_stacks[screen])

        new_pulse.time_stacks = time_stacks
        new_pulse.energy_stacks = energy_stacks
        new_pulse.x = x
        new_pulse.y = y

        return new_pulse

    def plot_1d_projection(self, image_name, dim='x'):
        """
        Method to show an image of the total integrated intensity
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        dim: str
            dimension for the lineout

        Returns
        -------

        """

        # generate the figure
        plt.figure()

        # generate the axes, in a grid
        ax = plt.subplot2grid((1,1), (0,0))

        # calculate the profile
        # profile = np.sum(np.abs(self.energy_stacks[image_name])**2, axis=2)
        profile = np.sum(np.abs(self.time_stacks[image_name])**2, axis=2)
        if dim == 'x':
            lineout = np.sum(profile, axis=0)
        else:
            lineout = np.sum(profile, axis=1)

        # show the horizontal lineout (distance in microns)
        ax.plot(getattr(self, dim)[image_name] * 1e6, lineout / np.max(lineout))
        ax.set_xlabel('%s coordinates (microns)' % dim)
        ax.set_ylabel('Intensity (normalized)')

        return ax, lineout

    def imshow_projection(self, image_name):
        """
        Method to show an image of the total integrated intensity
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show

        Returns
        -------

        """

        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # generate the figure
        plt.figure(figsize=(8, 8))

        # generate the axes, in a grid
        ax_profile = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        ax_y = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        ax_x = plt.subplot2grid((4, 4), (3, 0), colspan=3)

        # calculate the profile
        # profile = np.sum(np.abs(self.energy_stacks[image_name])**2, axis=2)
        profile = np.sum(np.abs(self.time_stacks[image_name])**2, axis=2)
        x_lineout = np.sum(profile, axis=0)
        y_lineout = np.sum(profile, axis=1)

        cx, cy, wx, wy, fwx_guess, fwy_guess = Pulse.beam_analysis(self.x[image_name], self.y[image_name],
                                                                   x_lineout, y_lineout)

        # show the 2D profile
        ax_profile.imshow(np.flipud(profile),
                          extent=(minx, maxx, miny, maxy), cmap=plt.get_cmap('gnuplot'),
                                clim=(0,np.max(profile)))
        # label coordinates
        ax_profile.set_xlabel('X coordinates (microns)')
        ax_profile.set_ylabel('Y coordinates (microns)')
        ax_profile.set_title('%s Spatial Projection' % image_name)
        # show the horizontal lineout (distance in microns)
        ax_x.plot(self.x[image_name] * 1e6, x_lineout / np.max(x_lineout))
        ax_x.plot(self.x[image_name] * 1e6, np.exp(-(self.x[image_name] - cx) ** 2 / 2 / (wx / 2.355) ** 2))
        # show the vertical lineout (distance in microns)
        ax_y.plot(y_lineout / np.max(y_lineout), self.y[image_name] * 1e6)
        ax_y.plot(np.exp(-(self.y[image_name] - cy) ** 2 / 2 / (wy / 2.355) ** 2), self.y[image_name] * 1e6)

        # add some annotations with beam centroid and FWHM
        ax_y.text(.6, .1 * np.max(self.y[image_name] * 1e6), 'centroid: %.2f %s' % (cy * 1e6, '\u03BCm'), rotation=-90)
        ax_y.text(.3, .1 * np.max(self.y[image_name] * 1e6), 'width: %.2f %s' % (wy * 1e6, '\u03BCm'), rotation=-90)
        ax_x.text(-.9 * np.max(self.x[image_name] * 1e6), .6, 'centroid: %.2f %s' % (cx * 1e6, '\u03BCm'))
        ax_x.text(-.9 * np.max(self.x[image_name] * 1e6), .3, 'width: %.2f %s' % (wx * 1e6, '\u03BCm'))

        plt.tight_layout()

        return ax_profile, ax_x, ax_y

    def imshow_energy_slice(self, image_name, dim='x', slice_pos=0, image_type='intensity'):
        """
        Method to show a slice along space and energy
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        dim: str
            spatial dimension for the slice ('x' or 'y')
        slice_pos: float
            spatial slice location (in y if dim='x' and vice versa). Units are microns.

        Returns
        -------

        """

        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)
        min_E = np.min(self.energy) - self.E0
        max_E = np.max(self.energy) - self.E0

        # generate the figure
        plt.figure(figsize=(6,6))

        # generate the axes, in a grid
        ax_profile = plt.subplot2grid((5,8),(0,0),colspan=7,rowspan=5)
        ax_colorbar = plt.subplot2grid((5,8),(1,7),colspan=1,rowspan=3)


        # horizontal slice
        if dim == 'x':
            # slice index
            N = self.x[image_name].size
            dx = (maxx - minx) / N
            index = int((slice_pos - minx) / dx)

            profile = np.abs(self.energy_stacks[image_name][index, :, :]) ** 2
            profile = profile / np.max(profile)
            if image_type == 'phase':
                mask = (profile > 0.01 * np.max(profile)).astype(float)
                profile = unwrap_phase(np.angle(self.energy_stacks[image_name][index, :, :])) * mask
                cmap = plt.get_cmap('jet')
                cbar_label = 'Phase (rad)'
            else:
                cmap = plt.get_cmap('gnuplot')
                cbar_label = 'Intensity (normalized)'
            extent = (min_E, max_E, minx, maxx)
            ylabel = 'X coordinates (microns)'
            aspect_ratio = (max_E - min_E) / (maxx - minx)
            title = u'%s Energy Slice: Y = %d \u03BCm' % (image_name, slice_pos)
        # vertical slice
        elif dim == 'y':
            # slice index
            N = self.y[image_name].size
            dx = (maxy-miny)/N
            index = int((slice_pos-miny)/dx)

            profile = np.abs(self.energy_stacks[image_name][:, index, :]) ** 2
            profile = profile / np.max(profile)
            if image_type == 'phase':
                mask = (profile > 0.01 * np.max(profile)).astype(float)
                profile = unwrap_phase(np.angle(self.energy_stacks[image_name][:, index, :])) * mask
                cmap = plt.get_cmap('jet')
                cbar_label = 'Phase (rad)'
            else:
                cmap = plt.get_cmap('gnuplot')
                cbar_label = 'Intensity (normalized)'
            extent = (min_E, max_E, miny, maxy)
            ylabel = 'Y coordinates (microns)'
            aspect_ratio = (max_E-min_E)/(maxy-miny)
            title = u'%s Energy Slice: X = %d \u03BCm' % (image_name, slice_pos)
        else:
            profile = np.zeros((256,256))
            extent = (0, 0, 0, 0)
            ylabel = ''
            aspect_ratio = 1
            title = ''

        # show the 2D profile
        im_profile = ax_profile.imshow(np.flipud(profile), aspect=aspect_ratio,
                          extent=extent, cmap=cmap)
        plt.colorbar(im_profile, cax=ax_colorbar, label=cbar_label)
        # label coordinates
        ax_profile.set_xlabel('Energy (eV)')
        ax_profile.set_ylabel(ylabel)
        # ax_profile.set_title('%s Energy Slice' % image_name)
        ax_profile.set_title(title)

    def imshow_time_slice(self, image_name, dim='x', slice_pos=0, shift=None):
        """
        Method to show a slice along space and time
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        dim: str
            spatial dimension for the slice ('x' or 'y')
        slice_pos: float
            spatial slice location (in y if dim='x' and vice versa). Units are microns.

        Returns
        -------

        """

        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)
        min_t = np.min(self.t_axis)
        max_t = np.max(self.t_axis)

        # generate the figure
        plt.figure(figsize=(6, 6))

        # generate the axes, in a grid
        # ax_profile = plt.subplot2grid((1, 1), (0, 0))
        ax_profile = plt.subplot2grid((5, 8), (0, 0), colspan=7, rowspan=5)
        ax_colorbar = plt.subplot2grid((5, 8), (1, 7), colspan=1, rowspan=3)

        # horizontal slice
        if dim == 'x':
            # slice index
            N = self.x[image_name].size
            dx = (maxx - minx) / N
            index = int((slice_pos - minx) / dx)
            profile = np.abs(self.time_stacks[image_name][index, :, :]) ** 2
            extent = (min_t, max_t, minx, maxx)
            ylabel = 'X coordinates (microns)'
            aspect_ratio = (max_t - min_t) / (maxx - minx)
            title = u'%s Time Slice: Y = %d \u03BCm' % (image_name, slice_pos)
        # vertical slice
        elif dim == 'y':
            # slice index
            N = self.y[image_name].size
            dx = (maxy - miny) / N
            index = int((slice_pos - miny) / dx)
            profile = np.abs(self.time_stacks[image_name][:, index, :]) ** 2
            extent = (min_t, max_t, miny, maxy)
            ylabel = 'Y coordinates (microns)'
            aspect_ratio = (max_t - min_t) / (maxy - miny)
            title = u'%s Time Slice: X = %d \u03BCm' % (image_name, slice_pos)
        else:
            profile = np.zeros((256, 256))
            extent = (0, 0, 0, 0)
            ylabel = ''
            aspect_ratio = 1

        # normalize
        profile = profile/np.max(profile)

        if shift is not None:
            profile = np.roll(profile, int(shift/self.deltaT), axis=1)

        # show the 2D profile
        im_profile = ax_profile.imshow(np.flipud(profile), aspect=aspect_ratio,
                          extent=extent, cmap=plt.get_cmap('gnuplot'))
        cbar_label = 'Intensity (normalized)'
        plt.colorbar(im_profile, cax=ax_colorbar, label=cbar_label)
        # label coordinates
        ax_profile.set_xlabel('Time (fs)')
        ax_profile.set_ylabel(ylabel)
        ax_profile.set_title(title)

        return ax_profile

    def imshow_spatial_slice(self, image_name, slice_type='energy', slice_pos=None):
        """
        Method to show a spatial slice at a given energy or time.
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        slice_type: str
            'energy' or 'time'
        slice_pos: float
            time or energy to take the slice from (units are eV if energy or fs if time)

        Returns
        -------

        """

        if slice_type == 'time':
            if slice_pos is None:
                index = int(self.N/2)
            else:
                index = int((slice_pos - np.min(self.t_axis)) / self.deltaT)
            profile = np.abs(self.time_stacks[image_name][:, :, index])**2
            title = u'%s Spatial Slice: T = %d fs' % (image_name, slice_pos)
        else:
            if slice_pos is None:
                index = int(self.N/2)
            else:
                index = int((slice_pos - np.min(self.energy)) / self.dE)
            profile = np.abs(self.energy_stacks[image_name][:, :, index])**2
            title = u'%s Spatial Slice: E = %.2f eV' % (image_name, slice_pos)

        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # lineouts
        x_lineout = np.sum(profile, axis=0)
        y_lineout = np.sum(profile, axis=1)

        # generate the figure
        plt.figure(figsize=(8, 8))

        # generate the axes, in a grid
        ax_profile = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        ax_y = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        ax_x = plt.subplot2grid((4, 4), (3, 0), colspan=3)

        extent = (minx, maxx, miny, maxy)
        aspect_ratio = (maxx - minx) / (maxy - miny)

        # show the 2D profile
        ax_profile.imshow(np.flipud(profile), aspect=aspect_ratio,
                          extent=extent, cmap=plt.get_cmap('gnuplot'))
        # label coordinates
        ax_profile.set_xlabel(u'X coordinates (\u03BCm)')
        ax_profile.set_ylabel(u'Y coordinates (\u03BCm)')
        ax_profile.set_title(title)
        # show the horizontal lineout (distance in microns)
        ax_x.plot(self.x[image_name] * 1e6, x_lineout / np.max(x_lineout))
        # show the vertical lineout (distance in microns)
        ax_y.plot(y_lineout / np.max(y_lineout), self.y[image_name] * 1e6)

    def plot_spectrum(self, image_name, x_pos=0, y_pos=0, integrated=False, log=False, voigt=False, show_fit=True):
        """
        Method to plot the spectrum at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        x_pos: float
            horizontal location (microns)
        y_pos: float
            vertical location (microns)
        integrated: bool
            whether to integrate the spectrum

        Returns
        -------

        """
        # get boundaries
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # get number of pixels
        M = self.x[image_name].size
        N = self.y[image_name].size

        # calculate pixel sizes (microns)
        dx = (maxx - minx) / M
        dy = (maxy - miny) / N

        # calculate indices for the desired location
        x_index = int((x_pos - minx) / dx)
        y_index = int((y_pos - miny) / dy)

        # calculate spectral intensity
        if integrated:
            y_data = np.sum(np.abs(self.energy_stacks[image_name])**2, axis=(0,1))
        else:
            y_data = np.abs(self.energy_stacks[image_name][y_index,x_index,:])**2

        # get gaussian stats
        centroid, sx = Util.gaussian_stats(self.energy, y_data)
        fwhm = sx * 2.355

        if voigt:
            # get voigt fit
            popt, pcov = optimization.curve_fit(Util.fit_lorentzian, self.energy, y_data, p0=[centroid, fwhm])
            gauss_plot = Util.fit_lorentzian(self.energy, popt[0], popt[1])
        else:
            # gaussian fit to plot
            gauss_plot = Util.fit_gaussian(self.energy, centroid, sx)

        # change label depending on bandwidth
        if fwhm >= 1:
            width_label = '%.1f eV FWHM' % fwhm
        elif fwhm > 1e-3:
            width_label = '%.1f meV FHWM' % (fwhm * 1e3)
        else:
            width_label = u'%.1f \u03BCeV FWHM' % (fwhm * 1e6)

        # plotting
        plt.figure()
        ax = plt.subplot2grid((1, 1), (0, 0))
        if log:
            ax.semilogy(self.energy - self.E0, y_data/np.max(y_data), label='Simulated')
            if show_fit:
                ax.semilogy(self.energy - self.E0, gauss_plot, label=width_label)
        else:
            ax.plot(self.energy - self.E0, y_data/np.max(y_data), label='Simulated')
            if show_fit:
                ax.plot(self.energy - self.E0, gauss_plot, label=width_label)
        ax.set_ylim(-.05,1.3)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity (normalized)')
        if integrated:
            ax.set_title(u'%s Integrated Spectrum' % (image_name))
        else:
            ax.set_title(u'%s Spectrum at X: %d \u03BCm, Y: %d \u03BCm' % (image_name, x_pos, y_pos))
        if show_fit:
            plt.legend()
        plt.grid()

        return ax

    def pulse_bandwidth(self, image_name, x_pos=0, y_pos=0):
        """
        Method to calculate the bandwidth of the pulse at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to check
        x_pos: float
            horizontal location (microns)
        y_pos: float
            vertical location (microns)

        Returns
        -------
        bandwidth: float
            pulse bandwidth (FWHM) in eV
        """
        # get boundaries
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # get number of pixels
        M = self.x[image_name].size
        N = self.y[image_name].size

        # calculate pixel sizes (microns)
        dx = (maxx - minx) / M
        dy = (maxy - miny) / N

        # calculate indices for the desired location
        x_index = int((x_pos - minx) / dx)
        y_index = int((y_pos - miny) / dy)

        # calculate spectral intensity
        y_data = np.abs(self.energy_stacks[image_name][y_index, x_index, :]) ** 2

        # get gaussian stats
        centroid, sx = Util.gaussian_stats(self.energy, y_data)
        fwhm = sx * 2.355

        return fwhm

    def central_energy(self, image_name, x_pos=0, y_pos=0):
        """
        Method to calculate the central energy of a pulse at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to check
        x_pos: float
            horizontal location (microns)
        y_pos: float
            vertical location (microns)

        Returns
        -------
        central_energy: float
            central energy of the pulse in eV
        """

        # get boundaries
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # get number of pixels
        M = self.x[image_name].size
        N = self.y[image_name].size

        # calculate pixel sizes (microns)
        dx = (maxx - minx) / M
        dy = (maxy - miny) / N

        # calculate indices for the desired location
        x_index = int((x_pos - minx) / dx)
        y_index = int((y_pos - miny) / dy)

        # calculate spectral intensity
        y_data = np.abs(self.energy_stacks[image_name][y_index, x_index, :]) ** 2

        # get gaussian stats
        centroid, sx = Util.gaussian_stats(self.energy, y_data)
        fwhm = sx * 2.355

        return centroid

    def throughput(self, image1_name, image2_name):
        """
        Method to calculate the throughput at image2 relative to image1
        Parameters
        ----------
        image1_name: str
            upstream profile monitor name
        image2_name: str
            downstream profile monitor name

        Returns
        -------
        throughput: float
            fraction of pulse energy arriving at image2 relative to image1
        """
        image1_data = self.energy_stacks[image1_name]
        image2_data = self.energy_stacks[image2_name]

        # total integrated intensity at image2 location normalized by integrated intensity at image1 location
        # for now these probably need to be the same pixel size since I'm not being careful with units
        throughput = np.sum(np.abs(image2_data)**2)/np.sum(np.abs(image1_data)**2)

        return throughput

    def pulsefront_tilt(self, image_name, dim='x', slice_pos=0, shift=None):
        """
        Method to calculate the pulse front tilt at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        dim: str
            spatial dimension for the slice ('x' or 'y')
        slice_pos: float
            spatial slice location (in y if dim='x' and vice versa). Units are microns.
        shift: float
            amount to shift pulse in fs. if None, calculated automatically
        Returns
        -------
        tilt: float
            pulse front tilt in units of fs/micron
        """
        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # horizontal slice
        if dim == 'x':
            # slice index
            N = self.x[image_name].size
            dx = (maxx - minx) / N
            index = int((slice_pos - minx) / dx)
            profile = np.abs(self.time_stacks[image_name][index, :, :]) ** 2
            # spatial coordinates (microns)
            x = np.copy(self.x[image_name])*1e6

        # vertical slice
        elif dim == 'y':
            # slice index
            N = self.y[image_name].size
            dx = (maxy - miny) / N
            index = int((slice_pos - miny) / dx)
            profile = np.abs(self.time_stacks[image_name][:, index, :]) ** 2
            # spatial coordinates (microns)
            x = np.copy(self.x[image_name])*1e6

        else:
            profile = np.zeros((256, 256))
            x = np.linspace(0, 255, 256)

        # find peak at central spatial position
        index = np.argmax(profile[int(N/2), :])

        # distance between array center and peak
        shift = int(np.size(profile[int(N/2), :]) / 2 - index)

        profile = np.roll(profile, shift, axis=1)

        # find peak (in time) at each position and put into fs units
        time_peaks = np.argmax(profile, axis=1) * self.deltaT

        # mask out anything outside the fwhm
        spatial_projection = np.sum(profile, axis=1)
        mask = spatial_projection>0.5*np.max(spatial_projection)

        time_peaks = time_peaks[mask]
        x = x[mask]

        # fit a line to the peaks
        p = np.polyfit(x, time_peaks, 1)
        # return slope (units are fs/micron)
        slope = p[0]

        return slope

    def spatial_chirp(self, image_name, dim='x', slice_pos=0, shift=None):
        """
        Method to calculate the spatial chirp at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        dim: str
            spatial dimension for the slice ('x' or 'y')
        slice_pos: float
            spatial slice location (in y if dim='x' and vice versa). Units are microns.
        shift: float
            amount to shift spectrum in eV. if None, calculated automatically

        Returns
        -------
        chirp: float
            spatial chirp in units of eV/micron
        """
        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # horizontal slice
        if dim == 'x':
            # slice index
            N = self.x[image_name].size
            dx = (maxx - minx) / N
            index = int((slice_pos - minx) / dx)
            profile = np.abs(self.energy_stacks[image_name][index, :, :]) ** 2
            # spatial coordinates (microns)
            x = np.copy(self.x[image_name]) * 1e6

        # vertical slice
        elif dim == 'y':
            # slice index
            N = self.y[image_name].size
            dx = (maxy - miny) / N
            index = int((slice_pos - miny) / dx)
            profile = np.abs(self.energy_stacks[image_name][:, index, :]) ** 2
            # spatial coordinates (microns)
            x = np.copy(self.x[image_name]) * 1e6

        else:
            profile = np.zeros((256, 256))
            x = np.linspace(0, 255, 256)

        # find peak at central spatial position
        index = np.argmax(profile[int(N / 2), :])

        # distance between array center and peak
        shift = int(np.size(profile[int(N / 2), :]) / 2 - index)

        profile = np.roll(profile, shift, axis=1)

        # find peak (in time) at each position and put into fs units
        energy_peaks = np.argmax(profile, axis=1) * self.dE

        # mask out anything outside the fwhm
        spatial_projection = np.sum(profile, axis=1)
        mask = spatial_projection > 0.5 * np.max(spatial_projection)

        energy_peaks = energy_peaks[mask]
        x = x[mask]

        # fit a line to the peaks
        p = np.polyfit(x, energy_peaks, 1)
        # return slope (units are fs/micron)
        slope = p[0]

        return slope

    def pulse_duration(self, image_name, x_pos=0, y_pos=0):
        """
        Method to calculate the temporal pulse structure at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to check
        x_pos: float
            horizontal location (microns)
        y_pos: float
            vertical location (microns)

        Returns
        -------
        pulse_width: float
            pulse width (FWHM) in fs
        """

        # get boundaries
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # get number of pixels
        M = self.x[image_name].size
        N = self.y[image_name].size

        # calculate pixel sizes (microns)
        dx = (maxx - minx) / M
        dy = (maxy - miny) / N

        # calculate indices for the desired location
        x_index = int((x_pos - minx) / dx)
        y_index = int((y_pos - miny) / dy)

        # calculate temporal intensity
        y_data = np.abs(self.time_stacks[image_name][y_index, x_index, :]) ** 2

        # find peak
        index = np.argmax(y_data)

        # distance between array center and peak
        shift = int(np.size(y_data)/2 - index)

        y_data = np.roll(y_data, shift)

        # get gaussian stats
        centroid, sx = Util.gaussian_stats(self.t_axis, y_data)
        fwhm = int(sx * 2.355)

        return centroid, fwhm

    def plot_pulse(self, image_name, x_pos=0, y_pos=0, shift=None):
        """
        Method to plot the temporal pulse structure at a given location
        Parameters
        ----------
        image_name: str
            name of the profile monitor to show
        x_pos: float
            horizontal location (microns)
        y_pos: float
            vertical location (microns)

        Returns
        -------

        """

        # get boundaries
        minx = np.round(np.min(self.x[image_name]) * 1e6)
        maxx = np.round(np.max(self.x[image_name]) * 1e6)
        miny = np.round(np.min(self.y[image_name]) * 1e6)
        maxy = np.round(np.max(self.y[image_name]) * 1e6)

        # get number of pixels
        M = self.x[image_name].size
        N = self.y[image_name].size

        # calculate pixel sizes (microns)
        dx = (maxx - minx) / M
        dy = (maxy - miny) / N

        # calculate indices for the desired location
        x_index = int((x_pos - minx) / dx)
        y_index = int((y_pos - miny) / dy)

        # calculate temporal intensity
        y_data = np.abs(self.time_stacks[image_name][y_index, x_index, :]) ** 2

        if shift is not None:
            y_data = np.roll(y_data, int(shift/self.deltaT))

        # get gaussian stats
        centroid, sx = Util.gaussian_stats(self.t_axis, y_data)
        fwhm = int(sx * 2.355)

        # gaussian fit to plot
        gauss_plot = Util.fit_gaussian(self.t_axis, centroid, sx)

        # plotting
        plt.figure()
        plt.plot(self.t_axis, y_data / np.max(y_data), label='Simulated')
        plt.plot(self.t_axis, gauss_plot, label=u'Gaussian Fit: %d fs FWHM' % fwhm)
        plt.ylim(-.05, 1.3)
        plt.xlabel('Time (fs)')
        plt.ylabel('Intensity (normalized)')
        plt.title(u'%s Pulse at X: %d \u03BCm, Y: %d \u03BCm' % (image_name, x_pos, y_pos))
        plt.legend()
        plt.grid()

        
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
        self.photonEnergy = beam_params['photonEnergy']
        # calculate wavelength (m)
        self.wavelength = 1239.8 / self.photonEnergy * 1e-9
        # calculate Rayleigh ranges (m)
        self.zRx = np.pi * self.sigma_x ** 2 / self.wavelength
        self.zRy = np.pi * self.sigma_y ** 2 / self.wavelength

        if 'z0x' in beam_params.keys():
            self.z0x = beam_params['z0x']
        else:
            self.z0x = self.zRx
        if 'z0y' in beam_params.keys():
            self.z0y = beam_params['z0y']
        else:
            if 'z0x' in beam_params.keys():
                self.z0y = np.copy(self.z0x)
                beam_params['z0y'] = self.z0y
            else:
                self.z0y = self.zRy
        if 'dx' in beam_params.keys():
            self.dx = beam_params['dx']
            self.dy = np.copy(self.dx)
        else:
            self.dx = None
            self.dy = None

        # calculate beam widths
        self.wx = self.sigma_x*np.sqrt(1+(self.z0x/self.zRx)**2)
        self.wy = self.sigma_y*np.sqrt(1+(self.z0y/self.zRy)**2)

        # calculate divergence
        divergence_x = self.wavelength / np.pi / self.sigma_x
        divergence_y = self.wavelength / np.pi / self.sigma_y
        # divergence_x = self.wx / self.z0x
        # divergence_y = self.wy / self.z0y

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
