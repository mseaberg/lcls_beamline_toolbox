"""
optics1d module

Part of the xraybeamline2d package.

Currently implements the following optical elements:
Mirror: parent mirror class
FlatMirror: flat transport mirror
CurvedMirror: elliptical KB mirror
Mono: implementation of the NEH2.2 monochromator
Grating: VLS grating up to 3rd order
Collimator: photon collimator (circular aperture)
Slit: rectangular aperture
Drift: path length between adjacent optical elements
CRL: compound refractive lens (parabolic)
PPM: power profile monitor, for viewing beam intensity
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolation
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.spatial.transform as transform
import scipy.integrate as integration
import os
from .util import Util, LegendreUtil
from .pitch import TalbotLineout
import scipy.interpolate as interpolate
import xrt.backends.raycing.materials as materials


class Mirror:
    """
    Class for a glancing incidence X-ray mirror

    Attributes
    ----------
    name: str
        Name of the mirror (e.g. MR1L0)
    length: float
        Length of the mirror in meters (long direction)
    width: float
        Width of the mirror in meters (short direction)
    alpha: float
        Nominal glancing angle of incidence (rad)
    x: float
        Horizontal position along beamline (meters)
    y: float
        Vertical position along beamline (meters)
    z: float
        Longitudinal position along beamline (meters)
    global_alpha: float
        mirror angle in global sense
    orientation: int
        Mirror deflection orientation: 0, 1, 2, 3 (see Figure 1 in documentation)
    shapeError: ndarray, with shape (M,) or (N,M)
        Shape error of the mirror. If 1D, assumed to be along the long dimension. If 2D, the 0th index corresponds
        to the short dimension, and the 1st index corresponds to the long dimension. Units are in nanometers.
        Can be any 1D or 2D shape, will be assumed to cover the extent of the mirror and will be interpolated onto
        beam coordinates.
    dx: float
        Shift of the mirror normal to the mirror surface (meters)
    dy: float
        Shift of the mirror parallel to the mirror Y-axis (meters)
    delta: float
        Adjustment to the glancing angle of incidence in radians (counter-clockwise rotation about mirror Y-axis)
    roll: float
        Adjustment of the roll angle (counter-clockwise rotation about mirror X-axis)
    yaw: float
        Adjustment of the yaw angle (counter-clockwise rotation about mirror Z-axis)
    motor_list: list of strings
        Mirror degrees of freedom available as motorized axes
    projectWidth: float
        Mirror length projected onto transverse beam plane (meters)
    """

    def __init__(self, name, **kwargs):
        """
        Initialization for Mirror class
        :param name: str
            mirror name (e.g. MR1L0)
        :param kwargs: various
            any of the following: length, width, alpha, z, orientation, shapeError, delta,
                                  dx, dy, roll, yaw, motor_list
            See class attributes for kwargs descriptions
        """

        # set mirror name
        self.name = name
        # set default parameters
        self.motor_list = ['dx', 'delta']
        self.length = 1.
        self.width = 25.e-3
        self.alpha = 1.e-3
        self.beta0 = 1.e-3
        self.z = None
        self.orientation = 0
        self.shapeError = None
        self.delta = 0.
        self.roll = 0.
        self.yaw = 0.
        self.dx = 0.
        self.dy = 0.
        self.global_x = 0
        self.global_y = 0
        self.global_alpha = 0
        self.azimuth = 0
        self.elevation = 0
        self.transverse = None
        self.sagittal = None
        self.normal = None

        # set allowed kwargs
        allowed_arguments = ['length', 'width', 'alpha', 'z', 'orientation', 'shapeError',
                             'delta', 'dx', 'dy', 'motor_list', 'roll', 'yaw']
        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        # set beta to alpha for mirrors
        self.beta0 = self.alpha

        # set some calculated attributes
        self.projectWidth = np.abs(self.length * (self.alpha + self.delta))

    def enable_motors(self, *axes):
        """
        Method to add additional motors
        :param axes: str
            Motor name(s)
        :return: None
        """
        # Loop through inputs
        for axis in axes:
            # add each axis to the motor list
            self.motor_list.append(axis)

    def disable_motors(self, *axes):
        """
        Method to remove an axis/axes from the list of available motors
        :param axes: str
            Motor name(s)
        :return: None
        """
        # Loop through inputs
        for axis in axes:
            # remove each axis from the motor list
            self.motor_list.remove(axis)

    def rotation(self, k_i):
        """
        Method to calculate resulting k-vector based on incident k-vector and mirror orientation
        :param k_i: (3,) ndarray
            incident k-vector
        :return delta_k: (3,) ndarray
            change in outgoing k-vector (k_f - k_f0)
        """

        # figure out mirror vectors:
        mirror_x = np.array([1, 0, 0], dtype=float)
        mirror_y = np.array([0, 1, 0], dtype=float)
        mirror_z = np.array([0, 0, 1], dtype=float)

        # rotate mirror pitch by delta first
        r1 = transform.Rotation.from_rotvec(mirror_y * self.delta)
        Ry = r1.as_matrix()
        mirror_x = np.matmul(Ry, mirror_x)
        mirror_y = np.matmul(Ry, mirror_y)
        mirror_z = np.matmul(Ry, mirror_z)

        # next is roll rotation
        r2 = transform.Rotation.from_rotvec(mirror_z * self.roll)
        Rz = r2.as_matrix()
        mirror_x = np.matmul(Rz, mirror_x)
        mirror_y = np.matmul(Rz, mirror_y)
        mirror_z = np.matmul(Rz, mirror_z)

        # finally yaw rotation
        r3 = transform.Rotation.from_rotvec(mirror_x * self.yaw)
        Rx = r3.as_matrix()
        mirror_x = np.matmul(Rx, mirror_x)
        mirror_y = np.matmul(Rx, mirror_y)
        mirror_z = np.matmul(Rx, mirror_z)

        # calculate outgoing k vector in absence of additional mirror rotations
        mirror_0 = np.array([1, 0, 0])
        k_f_normal = k_i - 2 * np.dot(k_i, mirror_0) * mirror_0

        # component of outgoing k vector along rotated mirror x-axis
        k_f_y = np.dot(k_i, mirror_y) * mirror_y
        # component of outgoing k vector along rotated mirror y-axis
        k_f_z = np.dot(k_i, mirror_z) * mirror_z
        # use conservation of momentum to calculate component of outgoing k-vector in the rotated mirror's
        # normal direction
        k_f_x = np.sqrt(1 - np.dot(k_f_y, k_f_y) - np.dot(k_f_z, k_f_z)) * mirror_x

        # full outgoing k-vector
        k_f = k_f_x + k_f_y + k_f_z

        # difference between outgoing k-vector and the k-vector in absence of mirror rotations
        delta_k = k_f - k_f_normal

        # return delta_k
        return delta_k

    def reflect(self, beam):
        """
        Method to reflect a beam from a flat mirror.
        :param beam: Beam
            Beam object to be reflected. Beam object is modified.
        :return: None
        """

        # initialize some arrays based on beam shape
        shapeError2 = np.zeros_like(beam.x)
        k_ix = 0
        k_iy = 0
        k_iz = 0
        zi = np.zeros_like(beam.x)
        yi = np.zeros_like(beam.x)
        zi_1d = np.zeros(0)
        yi_1d = np.zeros(0)

        # actual angle of incidence
        total_alpha = self.alpha + self.delta

        # figure out outgoing k-vector based on incident beam and mirror orientation
        if self.orientation == 0:

            # account for change to angle of incidence
            total_alpha += -beam.ax

            k_ix = -np.sin(self.alpha - beam.ax)
            k_iy = np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.x / np.sin(total_alpha)
            zi_1d = zi
            yi = beam.y
            yi_1d = yi

        elif self.orientation == 1:

            # account for change to angle of incidence
            total_alpha += -beam.ay

            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(total_alpha)
            zi_1d = zi
            yi = -beam.x
            yi_1d = yi

        elif self.orientation == 2:

            # account for change to angle of incidence
            total_alpha += beam.ax

            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(total_alpha)
            zi_1d = zi
            yi = -beam.y
            yi_1d = yi

        elif self.orientation == 3:

            # account for change to angle of incidence
            total_alpha += beam.ay

            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(total_alpha)
            zi_1d = zi
            yi = beam.x
            yi_1d = yi

        k_i = np.array([k_ix, k_iy, k_iz])
        delta_k = self.rotation(k_i)

        # mirror shape error interpolation onto beam coordinates (if applicable)
        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                # 1D interpolation onto beam coordinates
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)

                # just take central line for 1d shape error
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError[int(Ns/2),:])

        # figure out aperturing due to mirror's finite size
        z_mask = (np.abs(zi - self.dx / np.tan(total_alpha)) < self.length / 2).astype(float)

        # height error now in meters
        total_error = shapeError2 * 1e-9

        # convert to phase error (additional factor of 2 due to reflection
        phase = -total_error * 4 * np.pi * np.sin(total_alpha) / beam.lambda0

        # now change outgoing beam k-vector based on mirror orientation
        if self.orientation == 0:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * phase)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=2*self.alpha)
            delta_ax = -2*beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha))
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
            # beam.rotate_beam(delta_ax=)
            # beam.ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha))
            # beam.ay += np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            # beam.beam_offset(x_offset=2 * self.dx * np.cos(total_alpha))
            delta_cx = 2 * self.dx * np.cos(total_alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 1:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * phase)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=2 * self.alpha)
            delta_ay = -2 * beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha))
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += -np.arcsin(delta_k[1])
            # beam.ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha))

            # adjust beam position due to mirror de-centering
            # beam.beam_offset(y_offset=2 * self.dx * np.cos(total_alpha))
            delta_cy = 2 * self.dx * np.cos(total_alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        elif self.orientation == 2:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * phase)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=-2 * self.alpha)
            delta_ax = -2 * beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha))
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha))
            # beam.ay += -np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            # beam.beam_offset(x_offset=-2 * self.dx * np.cos(total_alpha))
            delta_cx = -2 * self.dx * np.cos(total_alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 3:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * phase)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=-2 * self.alpha)
            delta_ay = -2 * beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha))
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += np.arcsin(delta_k[1])
            # beam.ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha))

            # adjust beam position due to mirror de-centering
            # beam.beam_offset(y_offset=-2 * self.dx * np.cos(total_alpha))
            delta_cy = -2 * self.dx * np.cos(total_alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

    def propagate(self, beam):
        """
        Method used with beamline2d.Beamline class. For Mirror, calls reflect
        :param beam: Beam
            Beam object to be reflected. Beam object is modified.
        :return: None
        """
        self.reflect(beam)

    def adjust_motor(self, motor_name, adjustment):
        """
        Method to adjust mirror position using a motorized mirror axis.
        :param motor_name: str
            String corresponding to motor
        :param adjustment: float
            Amount to adjust motor (relative move). Changes mirror attribute.
        :return: None
        """
        # print a message about motion
        print("Moving %s by %.2f microns." % (motor_name, adjustment * 1e6))

        # check if this is an allowed motor
        if motor_name in self.motor_list:
            # get current position
            currentValue = getattr(self, motor_name)
            # calculate new position
            newValue = currentValue + adjustment
            # move motor to new position
            setattr(self, motor_name, newValue)
        else:
            # if this isn't a valid motor, say so
            print("Not a motorized axis")


class FlatMirror(Mirror):
    """
    Class for a glancing incidence flat X-ray mirror. This is a child of the Mirror class.

    Attributes
    ----------
    name: str
        Name of the mirror (e.g. MR1L0)
    length: float
        Length of the mirror in meters (long direction)
    width: float
        Width of the mirror in meters (short direction)
    alpha: float
        Nominal glancing angle of incidence (rad)
    z: float
        Longitudinal position along beamline (meters)
    orientation: int
        Mirror deflection orientation: 0, 1, 2, 3 (see Figure 1 in documentation)
    shapeError: ndarray, shape (M,) or (N,M)
        Shape error of the mirror. If 1D, assumed to be along the long dimension. If 2D, the 0th index corresponds
        to the short dimension, and the 1st index corresponds to the long dimension. Units are in nanometers.
        Can be any 1D or 2D shape, will be assumed to cover the extent of the mirror and will be interpolated onto
        beam coordinates.
    dx: float
        Shift of the mirror normal to the mirror surface (meters)
    dy: float
        Shift of the mirror parallel to the mirror Y-axis (meters)
    delta: float
        Adjustment to the glancing angle of incidence in radians (counter-clockwise rotation about mirror Y-axis)
    roll: float
        Adjustment of the roll angle (counter-clockwise rotation about mirror X-axis)
    yaw: float
        Adjustment of the yaw angle (counter-clockwise rotation about mirror Z-axis)
    motor_list: list of strings
        Mirror degrees of freedom available as motorized axes
    projectWidth: float
        Mirror length projected onto transverse beam plane (meters)
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)


class CurvedMirror(Mirror):
    """
    Class for a glancing incidence elliptical KB X-ray mirror. Child of the Mirror class.

    Attributes
    ----------
    name: str
        Name of the mirror (e.g. MR1L0)
    length: float
        Length of the mirror in meters (long direction)
    width: float
        Width of the mirror in meters (short direction)
    alpha: float
        Nominal glancing angle of incidence (rad)
    z: float
        Longitudinal position along beamline (meters)
    p: float
        ellipse source distance (m)
    q: float
        ellipse focus distance (m)
    dF1: float
        upstream bender actuation (m)
    dF2: float
        downstream bender actuation (m)
    high_order:
    orientation: int
        Mirror deflection orientation: 0, 1, 2, 3 (see Figure 1 in documentation)
    shapeError: ndarray, shape (M,) or (N,M)
        Shape error of the mirror. If 1D, assumed to be along the long dimension. If 2D, the 0th index corresponds
        to the short dimension, and the 1st index corresponds to the long dimension. Units are in nanometers.
        Can be any 1D or 2D shape, will be assumed to cover the extent of the mirror and will be interpolated onto
        beam coordinates.
    dx: float
        Shift of the mirror normal to the mirror surface (meters)
    dy: float
        Shift of the mirror parallel to the mirror Y-axis (meters)
    delta: float
        Adjustment to the glancing angle of incidence in radians (counter-clockwise rotation about mirror Y-axis)
    roll: float
        Adjustment of the roll angle (counter-clockwise rotation about mirror X-axis)
    yaw: float
        Adjustment of the yaw angle (counter-clockwise rotation about mirror Z-axis)
    motor_list: list of strings
        Mirror degrees of freedom available as motorized axes
    projectWidth: float
        Mirror length projected onto transverse beam plane (meters)
    """

    def __init__(self, name, p=100, q=1, dF1=0, dF2=0, **kwargs):
        """
        Create a CurvedMirror object
        :param name: str
            Mirror name (e.g. MR2K4)
        :param p: float
            ellipse source distance (m)
        :param q: float
            ellipse focus distance (m)
        :param dF1: float
            upstream bender actuation (m)
        :param dF2: float
            downstream bender actuation (m)
        :param kwargs: any of the following: length, width, alpha, z, orientation, shapeError, delta,
                                  dx, dy, roll, yaw, motor_list
            See class attributes for kwargs descriptions of the same name
        """

        super().__init__(name, **kwargs)
        self.p = p
        self.q = q
        self.dF1 = dF1
        self.dF2 = dF2
        self.total_alpha = self.alpha + self.delta

        # check if mirror is too long for distance to focus or source
        if self.length/2 > np.abs(self.p):
            print('Mirror is longer than distance to source. Adjusting length to be compatible.')
            self.length = 2 * self.p * .9
        if self.length/2 > np.abs(self.q):
            print('Mirror is longer than distance to focus. Adjusting length to be compatible.')
            self.length = 2 * self.q * .9

    def bend(self, cz):
        """
        Method to calculate polynomial coefficients due to bender influence
        :return pBend: List of floats
            Polynomial coefficients following np.polyfit order convention.
        """
        # calculate 3rd order due to benders
        p3 = (self.dF2 - self.dF1) / 6 / self.length
        # calculate 2nd order due to benders
        p2 = (self.dF1 + self.dF2) / 4

        pBend = [p3, p2, 0, 0]

        # offset
        # offset = cz - self.dx / np.tan(self.total_alpha)
        #
        # pBend = Util.recenter_coeff(p_coeffs, offset)

        # # calculate modification to coefficients due to de-centering
        # # 3rd order (centered)
        # p3rd = p3
        # # 2nd order: centered coefficient plus decentering contribution
        # p2nd = p2 + Util.decentering(p_coeffs, 2, offset)
        # # 1st order: centered coefficient is zero, plus decentering contribution
        # p1st = 0.0 + Util.decentering(p_coeffs, 1, offset)
        #
        # # put coefficients in a list
        # pBend = [p3rd, p2nd, p1st]

        return pBend

    def calc_ellipse(self, p, q, alpha):
        """
        Method to calculate the shape of an ellipse based on mirror specifications. See Ellipse reference documentation.
        :param p: float
            Nominal distance to source (m)
        :param q: float
            Nominal distance to focus (m)
        :param alpha: float
            Nominal angle of incidence (radians)
        :return z1: (N,) ndarray
            ellipse z-axis coordinates
        :return x1: (N,) ndarray
            mirror surface as function of z1
        :return z0: float
            z position at center of mirror (relative to ellipse center)
        :return x0: float
            x position at center of mirror (relative to ellipse center)
        :return delta: float
            angle at center of mirror relative to ellipse z-axis (radians)
        """

        # arbitrarily chosen array size
        N = 1024

        # concave elliptical mirror
        if q>=0 and p>=0:

            # calculated ellipse values
            L = np.sqrt(p ** 2 + q ** 2 + 2 * p * q * np.cos(2 * alpha))
            a2 = (p + q) ** 2 / 4  # a^2 for ellipse
            b2 = a2 - (L / 2) ** 2  # b^2 for ellipse

            # angle of incident beam
            beta = np.arcsin(np.sin(2 * alpha) * q / L)

            # mirror angle
            delta = alpha - beta

            # mirror offset from ellipse center in x
            x0 = -p * q / L * np.sin(2 * alpha)
            if p > q:
                z0 = np.sqrt(a2) * np.sqrt(1 - x0 ** 2 / b2)
            else:
                z0 = -np.sqrt(a2) * np.sqrt(1 - x0 ** 2 / b2)

            # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
            z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)
            # ellipse equation (using center of ellipse as origin)

            x1 = -np.sqrt(b2) * np.sqrt(1 - z1 ** 2 / a2) * np.sign(alpha)

            return z1, x1, z0, x0, delta

        # convex hyperbolic mirror
        elif q<0 and p>=0:
            print('hyperbolic')
            # calculated hyperbola values
            L = np.sqrt(p**2+q**2-2*np.abs(p)*np.abs(q)*np.cos(2*alpha))
            print('L %.2f' % L)
            # a2 = (p-q)**2/4
            a = -(np.abs(q) - np.abs(p))/2
            a2 = a**2
            c2 = (L/2)**2
            b2 = c2-a2
            print(b2)
            # angle of incident beam
            beta = np.arcsin(np.sin(2*alpha)*np.abs(q)/L)
            print('beta %.2e' % beta)

            # mirror angle
            delta = alpha + beta

            # mirror offset from hyperbola center in x
            x0 = -p*q/L*np.sin(2*alpha)
            if np.abs(p) > np.abs(q):
                z0 = np.sqrt(a2) * np.sqrt(1+x0**2/b2)
            else:
                z0 = -np.sqrt(a2) * np.sqrt(1+x0**2/b2)

            # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
            z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length /2 * np.cos(delta), N)

            # hyperbola equation (using center of hyperbola as origin)
            x1 = np.sqrt(b2) * np.sqrt(z1**2 / a2 - 1) * np.sign(alpha)

            return z1, x1, z0, x0, delta

        # concave hyperbolic mirror
        elif p<0 and q>=0:
            print('hyperbolic')
            # calculated hyperbola values
            L = np.sqrt(p ** 2 + q ** 2 - 2 * np.abs(p) * np.abs(q) * np.cos(2 * alpha))
            print('L %.2f' % L)
            # a2 = (p-q)**2/4
            a = -(np.abs(q) - np.abs(p)) / 2
            a2 = a ** 2
            c2 = (L / 2) ** 2
            b2 = c2 - a2
            print(b2)
            # angle of incident beam
            beta = np.arcsin(np.sin(2 * alpha) * np.abs(q) / L)
            print('beta %.2e' % beta)

            # mirror angle
            delta = alpha + beta

            # mirror offset from hyperbola center in x
            x0 = p * q / L * np.sin(2 * alpha)
            if np.abs(p) > np.abs(q):
                z0 = -np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)
            else:
                z0 = np.sqrt(a2) * np.sqrt(1 + x0 ** 2 / b2)

            # mirror x-coordinates (taking into account small mirror angle relative to x-axis)
            z1 = np.linspace(z0 - self.length / 2 * np.cos(delta), z0 + self.length / 2 * np.cos(delta), N)

            # hyperbola equation (using center of hyperbola as origin)
            x1 = -np.sqrt(b2) * np.sqrt(z1 ** 2 / a2 - 1) * np.sign(alpha)

            return z1, x1, z0, x0, delta



    def calc_misalignment(self, beam, cz):
        """
        Method to calculate the effect of angular misalignment in terms of aberrations.
        :param beam: Beam
            Beam object input
        :return p_res: (5,) ndarray
            4th degree np.polyfit output corresonding to polynomial coefficients of effective height aberration.
        """

        # z position of upstream focus
        zs = -self.p
        # initialize x position of upstream focus (beam x- or y-coordinates)
        xs = 0

        # figure out transverse offset from upstream ellipse focus
        # This is a combination of:
        #   - beam offset (beam.cx)
        #   - beam angle multiplied by lever arm (beam.ax*zs)
        #   - mirror offset (self.dx)

        if self.orientation == 0:
            # x position of upstream focus
            xs = beam.cx + beam.ax * zs - self.dx / np.cos(self.alpha + self.delta)
            beamz = beam.zx

            cz -= self.dx / np.tan(self.total_alpha)

            # effective beam z at center of mirror
            z_eff_c = beamz - cz*np.cos(self.total_alpha)
            # effective beam angle at center of mirror
            alpha_eff_c = -beam.ax - np.arctan(-cz*np.sin(self.total_alpha)/z_eff_c)

        elif self.orientation == 1:
            xs = beam.cy + beam.ay * zs - self.dx / np.cos(self.alpha + self.delta)
            beamz = beam.zy

            cz -= self.dx / np.tan(self.total_alpha)

            # effective beam z at center of mirror
            z_eff_c = beamz - cz * np.cos(self.total_alpha)
            # effective beam angle at center of mirror
            alpha_eff_c = -beam.ay - np.arctan(-cz * np.sin(self.total_alpha) / z_eff_c)

        elif self.orientation == 2:
            xs = -beam.cx - beam.ax * zs - self.dx / np.cos(self.alpha + self.delta)
            beamz = beam.zx

            cz -= self.dx / np.tan(self.total_alpha)

            # effective beam z at center of mirror
            z_eff_c = beamz - cz * np.cos(self.total_alpha)
            # effective beam angle at center of mirror
            alpha_eff_c = beam.ax - np.arctan(-cz * np.sin(self.total_alpha) / z_eff_c)

        elif self.orientation == 3:
            xs = -beam.cy - beam.ay * zs - self.dx / np.cos(self.alpha + self.delta)
            beamz = beam.zy

            cz -= self.dx / np.tan(self.total_alpha)

            # effective beam z at center of mirror
            z_eff_c = beamz - cz * np.cos(self.total_alpha)
            # effective beam angle at center of mirror
            alpha_eff_c = beam.ay - np.arctan(-cz * np.sin(self.total_alpha) / z_eff_c)

        # calculate ellipse based on design parameters
        z1, x1, z0, x0, delta1 = self.calc_ellipse(self.p, self.q, self.alpha)

        print('z_eff: %.2f' % z_eff_c)
        alpha_total = self.alpha + self.delta + alpha_eff_c
        print('a_eff: %.2f' % alpha_total)

        # calculate ideal ellipse for this angle of incidence
        zI, xI, z0I, x0I, deltaI = self.calc_ellipse(z_eff_c, self.q, alpha_total)

        # rotate actual ellipse into mirror coordinates
        x1m = -np.sin(delta1) * (z1 - z0) + np.cos(delta1) * (x1 - x0)# + x0

        # rotate ideal ellipse into mirror coordinates
        xIm = -np.sin(deltaI) * (zI - z0I) + np.cos(deltaI) * (xI - x0I)# + x0

        # effective height error
        height_error = x1m - xIm

        # fit to a polynomial
        p_res = np.polyfit(z1 - np.mean(z1), height_error, 4)
        # print(p_res)

        return p_res

    def reflect(self, beam):
        """
        Method to imprint a phase/amplitude onto the beam related to the effect of a (possibly misaligned) elliptical
        KB mirror.
        :param beam: Beam
            Beam object to be reflected. Object is modified by this function.
        :return: None
        """

        # initialize phase contributions
        high_order = np.zeros_like(beam.x)
        quadratic = 0
        linear = 0

        # initialize some other arrays
        zi = np.zeros_like(beam.x)
        yi = np.zeros_like(beam.x)
        zi_1d = np.zeros(0)
        yi_1d = np.zeros(0)
        k_ix = 0
        k_iy = 0
        k_iz = 0
        cz = 0
        cy = 0

        # actual angle of incidence
        self.total_alpha = self.alpha + self.delta

        shapeError2 = np.zeros_like(beam.x)

        # check distance to beam focus
        self.projectWidth = np.abs(self.length * (self.alpha + self.delta))

        # figure out outgoing k-vector based on incident beam and mirror orientation
        if self.orientation == 0:

            # small change to total angle of incidence
            self.total_alpha += -beam.ax

            k_ix = -np.sin(self.alpha - beam.ax)
            k_iy = np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.x / np.sin(self.total_alpha)
            zi_1d = zi
            cz = beam.cx / np.sin(self.total_alpha)
            print(cz*1e6)
            yi = beam.y
            yi_1d = yi
            cy = beam.cy
            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(self.total_alpha)
            alphaBeam = -beam.ax - np.arctan((zi_1d - cz) * np.sin(self.total_alpha) / zEff)
            beamz = beam.zx

        elif self.orientation == 1:

            # small change to total angle of incidence
            self.total_alpha += -beam.ay

            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(self.total_alpha)
            zi_1d = zi
            cz = beam.cy / np.sin(self.total_alpha)
            yi = -beam.x
            yi_1d = yi
            cy = -beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(self.total_alpha)
            alphaBeam = -beam.ay - np.arctan((zi_1d - cz) * np.sin(self.total_alpha) / zEff)
            beamz = beam.zy

        elif self.orientation == 2:

            # small change to total angle of incidence
            self.total_alpha += beam.ax

            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(self.total_alpha)
            zi_1d = zi
            cz = -beam.cx / np.sin(self.total_alpha)
            yi = -beam.y
            yi_1d = yi
            cy = -beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(self.total_alpha)
            alphaBeam = beam.ax - np.arctan((zi_1d - cz) * np.sin(self.total_alpha) / zEff)
            beamz = beam.zx

        elif self.orientation == 3:

            # small change to total angle of incidence
            self.total_alpha += beam.ay

            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(self.total_alpha)
            zi_1d = zi
            cz = -beam.cy / np.sin(self.total_alpha)
            yi = beam.x
            yi_1d = yi
            cy = beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(self.total_alpha)

            alphaBeam = beam.ay - np.arctan((zi_1d - cz) * np.sin(self.total_alpha) / zEff)
            beamz = beam.zy

        k_i = np.array([k_ix, k_iy, k_iz])
        delta_k = self.rotation(k_i)

        # mirror shape error interpolation onto beam coordinates (if applicable)
        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                # 1D interpolation onto beam coordinates
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(self.total_alpha), zs, self.shapeError)

            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)

                # 1D interpolation onto beam coordinates (just take central line)
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(self.total_alpha), zs, self.shapeError[int(Ns/2), :])

        # figure out aperturing due to mirror's finite size
        z_mask = (np.abs(zi - self.dx / np.tan(self.total_alpha)) < self.length / 2).astype(float)

        # calculate effect of ellipse misalignment
        p_misalign = self.calc_misalignment(beam, cz)

        # apply benders
        bend_coeff = self.bend(cz)

        # sum up coefficients from misalignment and bending
        coeff_total = Util.combine_coeff(p_misalign, bend_coeff)

        # offset along mirror z-axis
        offset = cz - self.dx / np.tan(self.total_alpha)

        # get coefficients centered about beam center instead of mirror center
        p_recentered = Util.recenter_coeff(coeff_total, offset)

        # get polynomial order
        M_poly = np.size(coeff_total) - 1

        # calculate contributions to high order error
        total_error = shapeError2 * 1e-9 + Util.polyval_high_order(p_recentered, zi - cz)

        # calculate effect on high order phase for glancing incidence mirror
        phase = -total_error * 4 * np.pi * np.sin(self.total_alpha) / beam.lambda0

        # add phase to high_order
        high_order += phase

        # scaling between mirror z-axis and new beam coordinates
        scale = np.sin(self.total_alpha)

        # scale the offset
        offset_scaled = offset * scale

        # for low orders change coordinates into reflected coordinates
        p_scaled = Util.poly_change_coords(coeff_total, scale)

        # multiply by -2 sin(alpha) to get total path length change
        p_scaled *= -2 * np.sin(self.total_alpha)

        # Add normal 2nd order phase to p_scaled
        # p_scaled[-3] += (-1 / (2 * (self.p + cz*np.cos(self.total_alpha)))
        #                  - 1 / (2 * (self.q - cz * np.cos(self.total_alpha))))
        # the difference between p and beamz is already accounted for in the "calc_misalignment" method now,
        # so the beam radius of curvature should be completely removed here. For the cases considered so far this
        # gave identical results to previously.
        p_scaled[-3] += (-1 / (2 * (beamz))
                         - 1 / (2 * (self.q - cz * np.cos(self.total_alpha))))


        # account for decentering
        p_scaled = Util.recenter_coeff(p_scaled, offset_scaled)

        # now add in normal focusing contribution to phase
        # factor out pi/lambda for quadratic term (so equal to 1/z)
        quadratic += 2 * p_scaled[-3]

        # factor out 2pi/lambda for linear term (so equal to change in propagation angle)
        linear += p_scaled[-2]

        # now change outgoing beam k-vector based on mirror orientation, and apply quadratic phase
        if self.orientation == 0:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=2 * self.alpha)
            delta_ax = -2 * beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha)) - linear
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha)) - linear
            # beam.ay += np.arcsin(delta_k[1])

            # adjust beam quadratic phase
            # beam.zx = 1 / (1 / beam.zx + quadratic)
            new_zx = 1 / (1 / beam.zx + quadratic)
            beam.change_z(new_zx=new_zx)

            # adjust beam position due to mirror de-centering
            delta_cx = 2 * self.dx * np.cos(self.total_alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 1:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=2 * self.alpha)
            delta_ay = -2 * beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha)) - linear
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += -np.arcsin(delta_k[1])
            # beam.ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha)) - linear

            # adjust beam quadratic phase
            # beam.zy = 1 / (1 / beam.zy + quadratic)
            new_zy = 1 / (1 / beam.zy + quadratic)
            beam.change_z(new_zy=new_zy)

            # adjust beam position due to mirror de-centering
            delta_cy = 2 * self.dx * np.cos(self.total_alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        elif self.orientation == 2:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=-2 * self.alpha)
            delta_ax = -2 * beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha)) + linear
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha)) + linear
            # beam.ay += -np.arcsin(delta_k[1])

            # adjust beam quadratic phase
            # beam.zx = 1 / (1 / beam.zx + quadratic)
            new_zx = 1 / (1 / beam.zx + quadratic)
            beam.change_z(new_zx=new_zx)

            # adjust beam position due to mirror de-centering
            delta_cx = -2 * self.dx * np.cos(self.total_alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 3:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=-2 * self.alpha)
            delta_ay = -2 * beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha)) + linear
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += np.arcsin(delta_k[1])
            # beam.ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha)) + linear

            # adjust beam quadratic phase
            # beam.zy = 1 / (1 / beam.zy + quadratic)
            new_zy = 1 / (1 / beam.zy + quadratic)
            beam.change_z(new_zy=new_zy)

            # adjust beam position due to mirror de-centering
            delta_cy = -2 * self.dx * np.cos(self.total_alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        # plt.figure()
        # plt.plot(np.abs(beam.wavex))
        # plt.figure()
        # plt.plot(np.angle(beam.wavex))

        return


class Mono:
    """
    Class for representing the NEH 2.2 monochromator. This is a CFF monochromator using a pre-mirror.

    Attributes
    ----------
    name: str
        Name of the monochromator
    m2: FlatMirror
        Monochromator pre-mirror
    grating: Grating
        Monochromator grating
    yag: PPM
        Monochromator profile monitor for centering the beam on the grating
    delta: float
        Angle to rotate grating, with corresponding pre-mirror rotation to maintain CFF (radians)
    f: float
        Monochromator "focal length" (distance from grating to exit slits in meters)
    cff: float
        constant fixed-focus parameter (ratio of sin(beta)/sin(alpha) where beta is the glancing diffraction angle
        and alpha is the glancing incidence angle).
    e0: float
        photon energy that monochromator is aligned for (eV)
    m_ref: float
        reference pre-mirror angle of incidence when beam is centered on pre-mirror
    alpha_ref: float
        reference grating angle of incidence
    beta_ref: float
        reference grating diffraction angle
    energy_ref: float
        reference energy for this CFF ratio
    delta_mirror: float
        angular adjustment of the pre-mirror to compensate for adjustment of the grating
    z: float
        position along the beamline of the monochromator (taken as pre-mirror position)
    """

    def __init__(self, name, M2=None, grating=None, YAG=None, CFF=3, delta=0, f=None, E0=1150):
        """
        Monochromator initialization
        :param name: str
            Name of the monochromator
        :param M2: FlatMirror
            Monochromator pre-mirror
        :param grating: Grating
            Monochromator grating
        :param YAG: PPM
            Monochromator profile monitor for centering the beam on the grating
        :param CFF: float
            constant fixed-focus parameter (ratio of sin(beta)/sin(alpha) where beta is the glancing diffraction angle
            and alpha is the glancing incidence angle).
        :param delta: float
            Angle to rotate grating, with corresponding pre-mirror rotation to maintain CFF (radians)
        :param f: float
            Monochromator "focal length" (distance from grating to exit slits in meters)
        :param E0: float
            photon energy that monochromator is aligned for (eV)
        """
        # set attributes
        self.name = name
        self.m2 = M2
        self.grating = grating
        self.yag = YAG
        self.delta = delta
        self.f = f
        self.cff = CFF
        self.grating.z = self.m2.z + .68
        self.e0 = E0

        # set grating focal length
        self.grating.f = self.f
        # set grating energy
        self.grating.lambda0 = 1239.8/E0*1e-9

        # calculate some reference angles
        self.m_ref = np.arctan(.012 / .68) / 2
        self.alpha_ref = (85.52e-3 - 2 * self.m_ref) / (self.cff + 1)
        self.beta_ref = self.cff * self.alpha_ref

        # calculate reference energy
        lambda1 = np.cos(self.alpha_ref) - np.cos(np.arcsin(self.cff * np.sin(self.alpha_ref))) / self.grating.n0
        self.energy_ref = 1239.8 / (lambda1 * 1e9)

        # set pre-mirror alpha (angle of incidence when beam is centered on pre-mirror)
        self.m2.alpha = self.m_ref

        # calculate grating angle of incidence and diffraction angle for energy E0
        alpha0 = self.calc_alpha()
        beta0 = self.calc_beta(alpha0)

        # actual pre-mirror angle of incidence based on grating orientation
        mirror0 = (85.52e-3 - alpha0 - beta0) / 2.

        # deviation of pre-mirror angle from reference angle
        delta_mirror = mirror0 - self.m_ref

        # further adjustment to pre-mirror if grating angle is adjusted
        self.delta_mirror = self.delta * (1 + 1 / self.cff) / 2

        # pre-mirror distance adjustment
        self.m2.z = self.m2.z + .006 * (self.delta_mirror + delta_mirror) - .68 * (
                np.cos(self.delta_mirror + delta_mirror) - 1)
        # pre-mirror x-axis position adjustment
        # self.m2.dx = self.m2.dx - .68 * (self.delta_mirror + delta_mirror) - .006 * (
        #         np.cos(self.delta_mirror + delta_mirror) - 1)

        # pre-mirror angle adjustment
        # change this to an adjustment of alpha
        self.m2.alpha += self.delta_mirror + delta_mirror
        self.m2.dz = (- .68 * (self.delta_mirror + delta_mirror) - .006 * (
                 np.cos(self.delta_mirror + delta_mirror) - 1)) / np.tan(self.m2.alpha)

        self.m2.beta0 = self.m2.alpha
        # self.m2.delta = self.delta_mirror + delta_mirror
        # grating angle of incidence
        self.grating.alpha = alpha0 + self.delta - self.delta_mirror * 2
        # grating diffraction angle
        self.grating.beta0 = beta0 - self.delta

        # set monochromator z-position to pre-mirror z-position
        self.z = self.m2.z

    def calc_alpha(self):
        """
        Method to calculate grating angle of incidence for a given photon energy and CFF
        :return alpha0: float
            grating angle of incidence (radians)
        """
        # generate an array of equally spaced angles covering the full range
        alpha = np.linspace(.001, .05, 1000)
        # calculate corresponding wavelength
        lambda1 = (np.cos(alpha) - np.cos(np.arcsin(self.cff * np.sin(alpha)))) / self.grating.n0
        # convert to photon energy (eV)
        energy1 = 1239.8 / (lambda1 * 1e9)

        # interpolate to find the proper angle for this energy
        alpha0 = Util.interp_flip(self.e0, energy1, alpha)
        return alpha0

    def calc_beta(self, alpha):
        """
        Method to calculate diffraction angle based on grating parameters and incidence angle
        :param alpha: float
            grating glancing angle of incidence (radians)
        :return beta: float
            grating glancing diffraction angle (radians)
        """
        # calculate wavelength
        lambda0 = 1239.8 / self.e0 * 1e-9
        # calculate diffraction angle based on grating equation
        beta = np.arccos(np.cos(alpha) - self.grating.n0 * lambda0)
        return beta

    def calc_reference(self):
        """
        Method to set some references
        :return: None
        """
        self.m_ref = np.arctan(.012 / .68) / 2
        self.alpha_ref = (85.52e-3 - 2 * self.m_ref) / (self.cff + 1)
        self.beta_ref = self.cff * self.alpha_ref

        lambda1 = np.cos(self.alpha_ref) - np.cos(np.arcsin(self.cff * np.sin(self.alpha_ref))) / self.grating.n0
        self.energy_ref = 1239.8 / (lambda1 * 1e9)

    def propagate(self, beam):
        """
        Method to propagate the beam through the monochromator
        :param beam: Beam
            Beam object to propagate through. Object is modified.
        :return: None
        """
        # ------ Something to check, seems like this was double-counting some distance since we already set
        # ------ monochromator z to adjusted pre-mirror z. Maybe this will fix discrepancy with M1K1 q value
        # ------ with Shadow....
        # propagate additional distance to pre-mirror
        # beam.beam_prop(.006 * self.delta_mirror)
        ##########################################

        # reflect beam from pre-mirror
        self.m2.reflect(beam)
        # self.YAG.propagate(beam)
        # propagate from pre-mirror to grating
        beam.beam_prop(self.grating.z - self.m2.z)

        # calculate profile on monochromator YAG
        self.yag.propagate(beam)
        # adjust beam angle to prepare for grating orientation
        beam.ay -= 2 * self.m2.delta
        # propagate beam through grating
        self.grating.diffract(beam)


class Grating(Mirror):
    """
    Class for representing planar VLS gratings operating at glancing incidence angle

    Attributes
    ----------
    name: str
        device name (e.g. MR3K1)
    length: float
        grating length along z-axis
    width: float
        grating width along y-axis
    N0: float
        grating periodicity (lines/m)
    N1: float
        grating second order (lines/m^2)
    N2: float
        grating third order (lines/m^3)
    alpha: float
        grating angle of incidence (radians)
    delta: float
        adjustment to grating angle of incidence (rotation about y-axis)
    roll: float
        rotation about z-axis
    yaw: float
        rotation about x-axis
    orientation: int
        grating orientation: 0, 1, 2, or 3. See Figure 1 in documentation
    z: float
        z position along beamline (m)
    dx: float
        Shift of the mirror normal to the mirror surface (meters)
    dy: float
        Shift of the mirror parallel to the mirror Y-axis (meters)
    beta0: float
        glancing diffraction angle (radians)
    order: int
        diffraction order: 0 or 1
    f: float
        grating "focal length" (distance from grating to desired focus)
    """

    def __init__(self, name, N0=300, N1=None, N2=None, beta0=None, f=None, order=1, lambda0=None, **kwargs):
        """
        Initialize grating object
        :param name: str
            device name (e.g. MR3K1)
        :param N0: float
            grating periodicity (lines/mm)
        :param N1: float
            grating first order (lines/mm^2)
        :param N2: float
            grating second order (lines/mm^3)
        :param beta0: float
            glancing diffraction angle (radians)
        :param f: float
            grating "focal length" (distance from grating to desired focus)
        :param lambda0: float
            wavelength grating is aligned for
        :param order: int
            diffraction order: 0 or 1
        :param kwargs: any of the following: length, width, alpha, z, orientation, shapeError, delta,
                                  dx, dy, roll, yaw, motor_list
            See class attributes for kwargs descriptions of the same name
        """
        super().__init__(name, **kwargs)

        # convert from mm to m
        self.n0 = N0 * 1e3
        self.n1 = N1 * 1e6
        self.n2 = N2 * 1e9

        # set some more attributes
        self.order = order
        self.beta0 = beta0
        self.f = f
        self.lambda0 = lambda0

    def rotation_grating(self, k_i, lambda1):
        """
        Method to calculate output k-vector based on grating orientation
        :param k_i: (3,) ndarray
            initial k-vector in grating coordinates
        :param lambda1: float
            beam wavelength
        :return delta_k: (3,) ndarray
            change in outgoing k-vector (k_f - k_f0)
        """

        # figure out mirror vectors:
        mirror_x0 = np.array([1, 0, 0], dtype=float)
        mirror_y0 = np.array([0, 1, 0], dtype=float)
        mirror_z0 = np.array([0, 0, 1], dtype=float)
        grating_vector = np.array([0, 0, 1], dtype=float)

        r1 = transform.Rotation.from_rotvec(mirror_y0 * self.delta)
        Ry = r1.as_matrix()
        mirror_x = np.matmul(Ry, mirror_x0)
        mirror_y = np.matmul(Ry, mirror_y0)
        mirror_z = np.matmul(Ry, mirror_z0)
        grating_vector = np.matmul(Ry, grating_vector)

        r2 = transform.Rotation.from_rotvec(mirror_z * self.roll)
        Rz = r2.as_matrix()
        mirror_x = np.matmul(Rz, mirror_x)
        mirror_y = np.matmul(Rz, mirror_y)
        mirror_z = np.matmul(Rz, mirror_z)
        grating_vector = np.matmul(Rz, grating_vector)

        r3 = transform.Rotation.from_rotvec(mirror_x * self.yaw)
        Rx = r3.as_matrix()
        mirror_x = np.matmul(Rx, mirror_x)
        mirror_y = np.matmul(Rx, mirror_y)
        mirror_z = np.matmul(Rx, mirror_z)
        grating_vector = np.matmul(Rx, grating_vector)

        # print(mirror_x)
        # print(mirror_y)
        # print(mirror_z)

        # normal case when incoming beam has correct incidence angle (at beam center)
        k_ix_norm = -np.sin(self.alpha)
        k_iy_norm = 0
        k_iz_norm = np.cos(self.alpha)
        k_i_norm = np.array([k_ix_norm, k_iy_norm, k_iz_norm])

        # figure out k_f in "normal case"
        k_f_y = np.dot(k_i_norm, mirror_y0) * mirror_y0  # should be 0
        k_f_z = np.dot(k_i_norm, mirror_z0) * mirror_z0 - self.n0 * self.lambda0 * mirror_z0
        k_f_x = np.sqrt(1 - np.dot(k_f_y, k_f_y) - np.dot(k_f_z, k_f_z)) * mirror_x0
        k_f_normal = k_f_x + k_f_y + k_f_z  # should be same as k_i except x component changed sign

        # get component of k_i in direction of grating vector
        k_i_grating = np.dot(k_i, grating_vector)

        # component of k_f in direction of grating vector
        cos_beta = k_i_grating - self.n0 * lambda1

        # figure out the rest of k_f
        # component of k_i that is perpendicular to grating vector (but in plane) stays the same
        # (this is in the direction of mirror_y)
        k_i_y = np.dot(k_i, mirror_y)

        # component of k_f in direction of grating axis (mirror z-axis)
        k_f_g = cos_beta * grating_vector
        # print(np.dot(k_f_g, k_f_g))
        # component of k_f in direction of mirror y-axis
        k_f_perp = k_i_y * mirror_y
        # print(np.dot(k_f_perp, k_f_perp))
        # component of k_f in direction of mirror x-axis (by conservation of momentum
        k_f_x = np.sqrt(1 - np.dot(k_f_g, k_f_g) - np.dot(k_f_perp, k_f_perp)) * mirror_x

        # add up all components
        k_f = k_f_g + k_f_perp + k_f_x

        # calculate difference between outgoing k-vector and the k-vector in absence of grating rotations
        delta_k = k_f - k_f_normal

        # print(k_i)
        # print(k_f)
        # print(delta_k)

        return delta_k

    def propagate(self, beam):
        """
        Method that overrides Mirror class propagation
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """
        # if we're operating in zero order, just acts like a mirror
        if self.order == 0:
            self.reflect(beam)
        # if we're in first order, calculate diffraction
        elif self.order == 1:
            self.diffract(beam)

    def diffract(self, beam):
        """
        Method to calculate diffraction from a grating, including VLS parameters.
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """

        total_alpha = self.alpha + self.delta

        # initialize some arrays
        k_ix = 0
        k_iy = 0
        k_iz = 0
        alphaBeam = np.zeros_like(beam.x)
        zi = np.zeros_like(beam.x)
        yi = np.zeros_like(beam.x)
        zi_1d = np.zeros(0)
        yi_1d = np.zeros(0)
        cz = 0
        cy = 0
        beamz = 0

        if self.orientation == 0:

            # account for change to angle of incidence
            total_alpha += -beam.ax

            k_ix = -np.sin(self.alpha - beam.ax)
            k_iy = np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.x / np.sin(total_alpha)
            zi_1d = zi
            yi = beam.y
            yi_1d = yi

            cz = beam.cx / np.sin(total_alpha)
            cy = beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = -beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)
            beamz = beam.zx

        elif self.orientation == 1:

            # account for change to angle of incidence
            total_alpha += -beam.ay

            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(self.alpha + self.delta)
            zi_1d = zi
            yi = -beam.x
            yi_1d = yi

            cz = beam.cy / np.sin(total_alpha)
            cy = -beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = -beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)
            beamz = beam.zy

        elif self.orientation == 2:

            # account for change to angle of incidence
            total_alpha += beam.ax

            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(self.alpha + self.delta)
            zi_1d = zi
            yi = -beam.y
            yi_1d = yi

            cz = -beam.cx / np.sin(total_alpha)
            cy = -beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)
            beamz = beam.zx

        elif self.orientation == 3:

            # account for change to angle of incidence
            total_alpha += beam.ay

            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(self.alpha + self.delta)
            zi_1d = zi
            yi = beam.x
            yi_1d = yi

            cz = -beam.cy / np.sin(total_alpha)
            cy = beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(total_alpha)

            alphaBeam = beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)
            beamz = beam.zy

        k_i = np.array([k_ix, k_iy, k_iz])
        delta_k = self.rotation_grating(k_i, beam.lambda0)

        # project beam angle onto grating axis
        # Also take into account grating shift in dx (+dx corresponds to dz = -dx/alpha)

        # grating coordinates (along z-axis)
        z_g = np.linspace(-self.length / 2, self.length / 2, 1024)

        # # deviation from average angle of incidence at each point along the grating
        # alphaBeamG = Util.interp_flip(z_g, zi_1d - self.dx / np.tan(total_alpha), alphaBeam)

        z_g = zi_1d - self.dx / np.tan(total_alpha)
        # plt.figure()
        # plt.plot(z_g)

        # account for all contributions to alpha
        # alpha_total = self.alpha + self.delta + alphaBeamG
        alpha_total = self.alpha + self.delta + alphaBeam

        # calculate diffraction angle at every point on the grating
        beta = np.arccos(np.cos(alpha_total) - beam.lambda0 * (self.n0 + self.n1 * z_g + self.n2 * z_g ** 2))

        # figure out distance to focus
        D1 = self.n1 / self.n0 ** 2
        D0 = 1 / self.n0
        grating_focal_length = 1 / (self.lambda0 * D1 / D0 ** 2 / np.sin(self.beta0) ** 2)
        print('f: %.4f' % grating_focal_length)
        object_distance = beamz * (np.sin(self.beta0) / np.sin(self.alpha)) ** 2
        f2 = 1 / (1 / grating_focal_length - 1 / object_distance)
        self.f = f2
        print('Calculated distance to focus: %.6f' % f2)

        # calculate desired slope at each point of the grating
        x1 = self.f * np.sin(self.beta0 - self.delta) - self.dx
        z1 = self.f * np.cos(self.beta0 - self.delta)

        # take into account angular grating change, and dx
        x0 = 0.0

        # calculate ideal slope to focus at f in the direction beta0
        m = (x1 - x0) / (z1 - z_g)

        # calculate slope error
        slope_error = -np.tan(beta - np.arctan(m))

        # calculate phase contribution by integrating slope error. This is kind of equivalent to a height error but
        # we don't need to double-count it.
        # (do this with a polynomial fit up to 3rd order for now)
        # limit this to size of grating
        mask = np.abs(z_g) <= self.length/2
        # mask = np.ones_like(z_g).astype(bool)
        p = np.polyfit(z_g[mask], slope_error[mask], 3)

        # integrate slope error
        p_int = np.polyint(p)

        # offset from center (along mirror z-axis)
        offset = cz - self.dx / np.tan(total_alpha)

        # account for decentering
        p_recentered = Util.recenter_coeff(p_int, offset)

        # high order phase. Multiplied by sin(beta) because integration should actually happen in beam coordinates.
        high_order = (2 * np.pi / beam.lambda0 * Util.polyval_high_order(p_recentered, zi - cz) *
                      np.sin(self.beta0 - self.delta))

        # scaling between grating z-axis and new beam coordinates
        scale = np.sin(self.beta0 - self.delta)

        # change coordinate systems to get proper low-order coefficients. Multiplied by sin(beta) because integration
        # should actually happen in beam coordinates.
        p_scaled = Util.poly_change_coords(p_int, scale) * np.sin(self.beta0 - self.delta)

        # Add 2nd order phase to p_scaled
        p_scaled[-3] += -1 / (2 * self.f)

        # scale the offset
        offset_scaled = offset * scale

        # account for any decentering
        p_centered = Util.recenter_coeff(p_scaled, offset_scaled)

        # 2nd order phase (factoring out pi/lambda)
        p2nd = 2 * p_centered[-3]
        # print('z: %.2f' % (1/p2nd))

        # 1st order phase (factoring out 2 pi/lambda)
        # (only add any 1st order phase due to de-centering since the rest is already accounted for in delta_k).
        p1st = p_centered[-2] - p_scaled[-2]
        # print(p1st)

        # figure out aperturing due to mirror's finite size
        z_mask = (np.abs(zi - self.dx / np.tan(total_alpha)) < self.length / 2).astype(float)
        y_mask = (np.abs(yi - self.dy) < self.width / 2).astype(float)

        # 2D mirror aperture (1's and 0's)
        # mirror = z_mask * y_mask

        # multiply beam by aperture and phase
        # beam.wave *= mirror * np.exp(1j * high_order)

        # handle beam re-pointing depending on the orientation
        if self.orientation == 0:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order)

            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.asymmetry_x(np.sin(self.beta0) / np.sin(self.alpha))
            beam.cx *= np.sin(self.beta0) / np.sin(self.alpha)
            beam.x += beam.cx

            # add quadratic phase
            # beam.zx = 1 / (1 / beam.zx + p2nd)
            # beam.zx = 1 / p2nd
            new_zx = 1 / p2nd
            beam.change_z(new_zx=new_zx)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=self.alpha + self.beta0)
            delta_ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = np.arcsin(delta_k[0] / np.cos(self.alpha)) + p1st
            # beam.ay += np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            delta_cx = 2 * self.dx * np.cos(self.alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 1:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order)

            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.asymmetry_y(np.sin(self.beta0) / np.sin(self.alpha))
            beam.cy *= np.sin(self.beta0) / np.sin(self.alpha)
            beam.y += beam.cy

            # add quadratic phase
            # beam.zy = 1 / (1 / beam.zy + p2nd)
            # beam.zy = 1 / p2nd
            new_zy = 1 / p2nd
            beam.change_z(new_zy=new_zy)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=self.alpha + self.beta0)
            delta_ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += -np.arcsin(delta_k[1])
            # beam.ay = np.arcsin(delta_k[0] / np.cos(self.alpha)) + p1st

            # adjust beam position due to mirror de-centering
            delta_cy = 2 * self.dx * np.cos(self.alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        elif self.orientation == 2:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order)

            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.asymmetry_x(np.sin(self.beta0) / np.sin(self.alpha))
            beam.cx *= np.sin(self.beta0) / np.sin(self.alpha)
            beam.x += beam.cx

            # add quadratic phase
            # beam.zx = 1 / (1 / beam.zx + p2nd)
            # beam.zx = 1 / p2nd
            new_zx = 1 / p2nd
            beam.change_z(new_zx=new_zx)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=-self.alpha - self.beta0)
            delta_ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st
            # beam.ay += -np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            delta_cx = -2 * self.dx * np.cos(self.alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 3:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order)

            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.asymmetry_y(np.sin(self.beta0) / np.sin(self.alpha))
            beam.cy *= np.sin(self.beta0) / np.sin(self.alpha)
            beam.y += beam.cy

            # add quadratic phase
            # beam.zy = 1 / (1 / beam.zy + p2nd)
            # beam.zy = 1 / p2nd
            new_zy = 1 / p2nd
            beam.change_z(new_zy=new_zy)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=-self.alpha - self.beta0)
            delta_ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            beam.ax += np.arcsin(delta_k[1])
            beam.ay = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st

            # adjust beam position due to mirror de-centering
            delta_cy = -2 * self.dx * np.cos(self.alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        return


class Crystal(Mirror):
    """
    Class for representing planar crystal optics

    Attributes
    ----------
    name: str
        device name (e.g. CR3L0)
    hkl: list
        Crystal reflection hkl indices. Default is [1,1,1].
    alphaAsym: float
        crystal plane asymmetry angle, in radians. Negative means crystal normal faces in the "upstream" direction.
    pol: str
        Incident polarization: 's' or 'p'. Default is 's'.
    material: str
        Crystal material. Currently 'Si' and 'diamond' are implemented. default is 'Si'.
    E0: float
        Central photon energy for crystal reflection. Used to define Bragg angle and crystal angle of incidence.
    lambda0: float
        Central wavelength. Calculated from E0.
    crystal: materials.Crystal** from xrt package
        xrt Crystal object used for obtaining bragg angles, and complex reflectivity.
    d: float
        lattice spacing (m)
    bragg: float
        Bragg angle for energy E0. This accounts for shift due to crystal asymmetry.
    b: float
        Asymmetric factor (related to CFF of grating).
    length: float
        crystal length along z-axis
    width: float
        crystal width along y-axis
    alpha: float
        nominal crystal angle of incidence (radians)
    delta: float
        adjustment to crystal angle of incidence (rotation about y-axis)
    roll: float
        rotation about z-axis
    yaw: float
        rotation about x-axis
    orientation: int
        crystal orientation: 0, 1, 2, or 3. See Figure 1 in documentation
    z: float
        z position along beamline (m)
    dx: float
        Shift of the crystal normal to the mirror surface (meters)
    dy: float
        Shift of the crystal parallel to the mirror Y-axis (meters)
    beta0: float
        nominal glancing diffraction angle (radians)
    order: int
        diffraction order: 1 or higher. Not implemented yet.
    """

    def __init__(self, name, E0=None, hkl=None, material='Si', alphaAsym=0, order=1, pol='s', **kwargs):
        """
        Initialize grating object
        :param name: str
            device name (e.g. MR3K1)
        :param E0: float
            Central photon energy for crystal reflection. Used to define Bragg angle and crystal angle of incidence.
            This is a required parameter.
        :param hkl: list of ints (length 3)
            hkl indices of crystal reflection. Default is [1,1,1].
        :param material: str
            Crystal material. Currently 'Si' and 'diamond' are implemented. default is 'Si'.
        :param alphaAsym: float
            crystal plane asymmetry angle, in radians. Negative means crystal normal faces in the "upstream" direction.
        :param order: int
            Crystal diffraction order. Not implemented yet. Default 1.
        :param pol: str
            Incident polarization: 's' or 'p'. Default is 's'.
        :param kwargs: any of the following: length, width, alpha, z, orientation, shapeError, delta,
                                  dx, dy, roll, yaw, motor_list
            See class attributes for kwargs descriptions of the same name
        """
        super().__init__(name, **kwargs)

        self.hkl = hkl
        if self.hkl is None:
            self.hkl = [1, 1, 1]
        self.alphaAsym = alphaAsym
        self.material = material
        self.pol = pol
        self.E0 = E0
        self.lambda0 = 1239.8 / E0 * 1e-9

        # define xrt crystal for reflectivity and crystal parameters
        if self.material == 'Si':
            self.crystal = materials.CrystalSi(hkl=self.hkl)
        elif self.material == 'diamond':
            self.crystal = materials.CrystalDiamond(hkl=self.hkl)

        # lattice spacing
        self.d = self.crystal.d * 1e-10

        # get bragg peak angle
        self.bragg = self.crystal.get_Bragg_angle(self.E0) - self.crystal.get_dtheta(self.E0, alpha=alphaAsym)

        # calculate proper angle of incidence for energy E0 (general case is asymmetric)
        self.alpha = self.bragg + self.alphaAsym

        # define nominal beam k_i
        k_ix = -np.sin(self.alpha)
        k_iy = 0
        k_iz = np.cos(self.alpha)
        k_i = [k_ix, k_iy, k_iz]

        # define crystal plane normal in crystal surface coordinates
        c_x = np.cos(self.alphaAsym)
        c_z = np.sin(self.alphaAsym)
        c_normal = [c_x, 0, c_z]

        # component of crystal normal in +z direction
        c_parallel = c_z * self.lambda0 / (self.crystal.d * 1e-10)

        # calculate nominal beam k_f
        k_fy = 0
        k_fz = k_iz + c_parallel
        k_fx = np.sqrt(1 - k_fy**2 - k_fz**2)
        k_f = [k_fx, k_fy, k_fz]

        # calculate nominal diffraction angle (relative to crystal surface)
        self.beta0 = np.arccos(k_fz)

        # calculate asymmetric factor
        self.b = (np.sin(self.alpha)/np.sin(self.beta0))
        print('b %.2f' % self.b)

        # set diffraction order (not implemented yet)
        self.order = order

    def rotation_crystal(self, k_i, lambda1):
        """
        Method to calculate output k-vector based on grating orientation
        :param k_i: (3,) ndarray
            initial k-vector in grating coordinates
        :param lambda1: float
            beam wavelength
        :return delta_k: (3,) ndarray
            change in outgoing k-vector (k_f - k_f0)
        """

        # figure out mirror vectors:
        mirror_x0 = np.array([1, 0, 0], dtype=float)
        mirror_y0 = np.array([0, 1, 0], dtype=float)
        mirror_z0 = np.array([0, 0, 1], dtype=float)

        # crystal plane normal vector
        c_x = np.cos(self.alphaAsym)
        c_z = np.sin(self.alphaAsym)

        # vector parallel to crystal plane, in xz plane
        crystal_x = np.sin(self.alphaAsym)
        crystal_z = np.cos(self.alphaAsym)

        c_normal0 = np.array([c_x, 0, c_z], dtype=float)

        crystal_vector0 = np.array([crystal_x, 0, crystal_z], dtype=float)

        r1 = transform.Rotation.from_rotvec(mirror_y0 * self.delta)
        Ry = r1.as_matrix()
        mirror_x = np.matmul(Ry, mirror_x0)
        mirror_y = np.matmul(Ry, mirror_y0)
        mirror_z = np.matmul(Ry, mirror_z0)
        c_normal = np.matmul(Ry, c_normal0)
        crystal_vector = np.matmul(Ry, crystal_vector0)

        r2 = transform.Rotation.from_rotvec(mirror_z * self.roll)
        Rz = r2.as_matrix()
        mirror_x = np.matmul(Rz, mirror_x)
        mirror_y = np.matmul(Rz, mirror_y)
        mirror_z = np.matmul(Rz, mirror_z)
        c_normal = np.matmul(Rz, c_normal)
        crystal_vector = np.matmul(Rz, crystal_vector)

        r3 = transform.Rotation.from_rotvec(mirror_x * self.yaw)
        Rx = r3.as_matrix()
        mirror_x = np.matmul(Rx, mirror_x)
        mirror_y = np.matmul(Rx, mirror_y)
        mirror_z = np.matmul(Rx, mirror_z)
        c_normal = np.matmul(Rx, c_normal)
        crystal_vector = np.matmul(Rx, crystal_vector)

        # print(mirror_x)
        # print(mirror_y)
        # print(mirror_z)

        # ---- "normal case" when crystal is aligned perfectly to beamline
        # get component of crystal normal along crystal surface
        g_parallel = np.dot(c_normal0, mirror_z0) * mirror_z0 * self.lambda0/self.d

        k_ix_norm = -np.sin(self.alpha)
        k_iy_norm = 0
        k_iz_norm = np.cos(self.alpha)
        k_i_norm = np.array([k_ix_norm, k_iy_norm, k_iz_norm])

        # figure out k_f in "normal case"
        k_f_y = np.dot(k_i_norm, mirror_y0) * mirror_y0 # should be 0
        k_f_z = np.dot(k_i_norm, mirror_z0) * mirror_z0 + g_parallel
        k_f_x = np.sqrt(1 - np.dot(k_f_y, k_f_y) - np.dot(k_f_z, k_f_z)) * mirror_x0
        k_f_normal = k_f_x + k_f_y + k_f_z # should be same as k_i except x component changed sign

        # print('k_i ' + str(k_i))
        # print('k_f ' + str(k_f_normal))

        # ---- now figure out case when crystal is misaligned
        g_parallel = np.dot(c_normal, mirror_z) * mirror_z * lambda1/self.d

        k_f_y = np.dot(k_i, mirror_y) * mirror_y
        k_f_z = np.dot(k_i, mirror_z) * mirror_z + g_parallel
        k_f_x = np.sqrt(1 - np.dot(k_f_y, k_f_y) - np.dot(k_f_z, k_f_z)) * mirror_x
        # add up all components
        k_f = k_f_x + k_f_y + k_f_z

        # calculate difference between outgoing k-vector and the k-vector in absence of grating rotations
        delta_k = k_f - k_f_normal

        # print(k_i)
        # print(k_f)
        # print(delta_k)

        return delta_k, k_f

    def propagate(self, beam):
        """
        Method that overrides Mirror class propagation
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """
        # if we're operating in zero order, just acts like a mirror
        if self.order == 0:
            self.reflect(beam)
        # if we're in first order, calculate diffraction
        elif self.order == 1:
            self.diffract(beam)

    def calc_kf(self, z_s, k_iy, alpha_in, slope_error, lambda0):
        # calculate diffraction angle at every point on the grating
        # beta = np.arccos(np.cos(alpha_total) - beam.lambda0 * (self.n0 + self.n1 * z_g + self.n2 * z_g ** 2))
        m_x = np.array([1, 0, 0], dtype=float)
        m_y = np.array([0, 1, 0], dtype=float)
        m_z = np.array([0, 0, 1], dtype=float)

        # define k_i at each point along beam
        k_ix = np.outer(-np.sin(alpha_in), m_x)
        k_iy = np.outer(np.ones_like(z_s) * k_iy, m_y)
        # k_iz = np.outer(np.cos(alpha_total), m_z)
        k_iz = np.outer(
            np.sqrt(np.ones_like(z_s) - np.sum(k_ix * k_ix, axis=1) - np.sum(k_iy * k_iy, axis=1)) * np.sign(
                np.cos(alpha_in)), m_z)
        k_i = k_ix + k_iy + k_iz

        # define crystal plane at every coordinate including slope error
        c_x = np.outer(np.cos(self.alphaAsym - slope_error), m_x)
        c_z = np.outer(np.sin(self.alphaAsym - slope_error), m_z)
        c_normal = c_x + c_z

        c_parallel = np.outer(np.sum(c_normal * m_z, axis=1), m_z) * lambda0 / (self.crystal.d * 1e-10)
        k_fy = k_iy
        k_fz = k_iz + c_parallel
        k_fx = np.outer(np.sqrt(np.ones_like(z_s) - np.sum(k_fy * k_fy, axis=1) - np.sum(k_fz * k_fz, axis=1)), m_x)

        k_f = k_fy + k_fz + k_fx

        return k_i, k_f, m_x, c_normal

    def diffract(self, beam):
        """
        Method to calculate diffraction from a grating, including VLS parameters.
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """

        total_alpha = self.alpha + self.delta

        # initialize some arrays
        shapeError2 = np.zeros_like(beam.x)
        k_ix = 0
        k_iy = 0
        k_iz = 0
        alphaBeam = np.zeros_like(beam.x)
        zi = np.zeros_like(beam.x)
        yi = np.zeros_like(beam.x)
        zi_1d = np.zeros(0)
        yi_1d = np.zeros(0)
        cz = 0
        cy = 0

        wavefront = np.zeros_like(beam.x)


        if self.orientation == 0:
            # account for change to angle of incidence
            total_alpha -= beam.ax

            # k_ix = -np.sin(total_alpha)
            k_ix = -np.sin(self.alpha - beam.ax)
            k_iy = np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha - beam.ax))
            # k_iz = np.cos(total_alpha)

            # coordinate mapping for interpolation
            zi = beam.x / np.sin(total_alpha)
            zi_1d = zi
            yi = beam.y
            yi_1d = yi

            cz = beam.cx / np.sin(total_alpha)
            cy = beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = -beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            self.f = -beam.zx * (np.abs(np.sin(self.beta0)/np.sin(self.alpha))**2)
            # self.f = -beam.zx
            beamz = beam.zx

            wavefront = np.copy(beam.wavex)
            if beam.focused_x:
                print('subtracting second order')
                wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zx * (beam.x - beam.cx) ** 2)

        elif self.orientation == 1:
            # account for change to angle of incidence
            total_alpha -= beam.ay

            # k_ix = -np.sin(total_alpha)
            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha - beam.ay))
            # k_iz = np.cos(total_alpha)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(total_alpha)
            zi_1d = zi
            yi = -beam.x
            yi_1d = yi

            cz = beam.cy / np.sin(total_alpha)
            cy = -beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = -beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            self.f = -beam.zy * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
            # self.f = -beam.zy
            beamz = beam.zy

            wavefront = np.copy(beam.wavey)

            if beam.focused_y:
                wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zy * (beam.y - beam.cy) ** 2)

        elif self.orientation == 2:
            # account for change to angle of incidence
            total_alpha += beam.ax

            # k_ix = -np.sin(total_alpha)
            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha + beam.ax))
            # k_iz = np.cos(total_alpha)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(total_alpha)
            zi_1d = zi
            yi = -beam.y
            yi_1d = yi

            cz = -beam.cx / np.sin(total_alpha)
            cy = -beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            self.f = -beam.zx * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
            # self.f = -beam.zx
            beamz = beam.zx

            wavefront = np.copy(beam.wavex)

            if beam.focused_x:
                print('subtracting second order')
                wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zx * (beam.x - beam.cx) ** 2)

        elif self.orientation == 3:
            # account fo change to angle of incidence
            total_alpha += beam.ay

            # k_ix = -np.sin(total_alpha)
            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha + beam.ay))
            # k_iz = np.cos(total_alpha)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(total_alpha)
            zi_1d = zi
            yi = beam.x
            yi_1d = yi

            cz = -beam.cy / np.sin(total_alpha)
            cy = beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(total_alpha)

            alphaBeam = beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            self.f = -beam.zy * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
            # self.f = -beam.zy
            beamz = beam.zy

            wavefront = np.copy(beam.wavey)

            if beam.focused_y:
                wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zy * (beam.y - beam.cy) ** 2)

        # mirror shape error interpolation onto beam coordinates (if applicable)
        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                # 1D interpolation onto beam coordinates
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)

                # just take central line for 1d shape error
                shapeError2 = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs,
                                        self.shapeError[int(Ns / 2), :])


        # zi_1d is centered around cz, and beam is centered on cz

        # to make the mask, we need coordinates that are centered on the crystal,
        # with no offset this is zi_1d, with offset this is z_c = zi_1d - self.dx / np.tan(total_alpha)
        z_c = zi_1d - self.dx / np.tan(total_alpha)

        # beam-centered coordinates are zi_1d - cz (meaning zero at beam center) - we will call this z_b
        # This implies that z_b = z_c + self.dx / np.tan(total_alpha) - cz, which is consistent with the
        # offset applied below
        z_b = zi_1d - cz

        # in the end we need a polynomial that's centered on the beam (meaning on z_b)

        # limit fit to size of crystal
        mask = np.abs(z_c) <= self.length / 2
        print(np.sum(mask)/np.size(z_c))

        # perform a Legendre fit on the shape error, limited to the size of the crystal
        shapePoly = LegendreUtil(z_c[mask], shapeError2[mask], 16)

        # get second order term of Legendre fit for curved crystal calculation
        second_order = shapePoly.quad_coeff()

        # take derivative to get slope error
        shapePoly.legder(1)
        slope_error = shapePoly.legval()*1e-9

        # extend size of slope error array to size of beam array
        slope_error2 = np.zeros_like(z_c)
        slope_error2[mask] = slope_error
        slope_error = slope_error2

        # calculate nominal reflected k vector
        k_i = np.array([k_ix, k_iy, k_iz])
        delta_k, k_f = self.rotation_crystal(k_i, beam.lambda0)
        print(delta_k)

        # beta at beam center
        beta1 = np.arccos(k_f[2])

        # project beam angle onto grating axis
        # Also take into account grating shift in dx (+dx corresponds to dz = -dx/alpha)

        # calculate high order component of beam slope error. This is multiplied by
        ##############################
        beam_slope_error = np.gradient(np.unwrap(np.angle(wavefront)),z_b*np.sin(total_alpha))*beam.lambda0/2/np.pi
        ##############################

        # plt.figure()
        # plt.plot(z_b, beam_slope_error)

        dz = np.abs(z_b[1]-z_b[0])

        # plt.figure()
        # plt.plot(z_b, np.cumsum(beam_slope_error)*2*np.pi/beam.lambda0*dz)
        # plt.plot(z_b, np.unwrap(np.angle(wavefront)))
        # plt.plot(z_b, np.pi / beam.lambda0 / beam.zx * (beam.x - beam.cx) ** 2)

        # account for all contributions to alpha
        alpha_total = self.alpha + self.delta + alphaBeam
        # alpha_total[mask_beam] -= beam_slope_error
        alpha_full = np.copy(alpha_total)
        # alpha_full[mask_beam] -= beam_slope_error

        ##############################
        alpha_full -= beam_slope_error
        ##############################

        k_i_full, k_f_full, m_x, c_normal = self.calc_kf(zi_1d, k_iy, alpha_full, slope_error, beam.lambda0)

        k_i, k_f, temp1, temp2 = self.calc_kf(zi_1d, k_iy, alpha_total, slope_error, beam.lambda0)

        beta = np.arccos(k_f[:, 2])

        # calculate phase contribution by integrating slope error. This is kind of equivalent to a height error but
        # we don't need to double-count it.
        # (do this with a polynomial fit up to 3rd order for now)
        z_c = zi_1d - self.dx / np.tan(total_alpha)

        ##!! need to calculate effective focal distance while taking into account crystal curvature, similar to
        ##!! what was needed for the grating

        R = 1 / (2 * second_order*1e-9)
        print(R)

        # use equation for curved grating imaging condition. Works great!
        f2 = np.sin(self.beta0) ** 2 / (
                    (np.sin(self.alpha) + np.sin(self.beta0)) / R - np.sin(self.alpha) ** 2 / beamz)
        # f2 = -object_distance
        self.f = f2
        print('Calculated distance to focus: %.6f' % f2)

        # calculate desired slope at each point of the grating
        x1 = self.f * np.sin(self.beta0 - self.delta) - self.dx
        z1 = self.f * np.cos(self.beta0 - self.delta)

        # take into account angular grating change, and dx
        x0 = 0.0

        # calculate ideal slope to focus at f in the direction beta0
        m = (x1 - x0) / (z1 - z_c)

        # calculate slope error
        slope_error = -np.tan(beta - np.arctan(m))

        # limit fit to size of crystal
        mask = np.abs(z_c) <= self.length/2

        # fit legendre centered on beam
        shapePoly = LegendreUtil(z_c[mask], slope_error[mask], 16)
        # integrate slope error
        shapePoly.legint(1)

        # now subtract off second order Legendre polynomial.
        # residual = shapePoly.legval() - shapePoly.legval(2)

        # plt.figure()
        # plt.plot(shapePoly.x_norm, shapePoly.legval() - np.cumsum(slope_error[mask])*shapePoly.dx)


        if np.sum(mask) > 0:
            p = np.polyfit(z_c[mask], slope_error[mask], 16)
        else:
            p = np.zeros(16)

        # plt.figure()
        # plt.plot(z_c[mask],slope_error[mask])
        # plt.plot(z_c[mask],np.polyval(p,z_c[mask]))

        # integrate slope error
        p_int = np.polyint(p)

        # plt.figure()
        # plt.plot(z_c[mask],np.cumsum(slope_error[mask])*shapePoly.dx)
        # plt.plot(z_c[mask],np.polyval(p_int,z_c[mask]))

        # c2 = shapePoly.c[2]*3/2/(shapePoly.dx*shapePoly.N/2)**2
        c2 = shapePoly.quad_coeff()

        R = 1 / (2 * c2)
        # print('radius of curvature: %.2e' % R)

        # offset from center of crystal (along crystal z-axis)
        offset = cz - self.dx / np.tan(total_alpha)

        # account for decentering
        # print(p_int)
        p_recentered = Util.recenter_coeff(p_int, offset)
        print('offset %.6f' % offset)
        # print(p_int)
        # print(p_recentered)

        # high_order_temp = np.polyval(p_int, z_c)
        high_order_temp = integration.cumtrapz(slope_error, z_c, initial=0)
        high_order_temp[mask] -= shapePoly.legval(2)

        # subtract phase at beam center. This is already taken care of with the group delay
        beam_center_phase = np.interp(cz, zi_1d, high_order_temp)
        high_order_temp -= beam_center_phase

        # trade out polyfit coefficients for the coefficients found from Legendre polynomials
        # This helps keep most of the quadratic phase in the analytic term
        p_int[-3] += shapePoly.quad_coeff() - p_int[-3]
        p_int[-2] += shapePoly.linear_coeff() - p_int[-2]
        p_int[-1] += shapePoly.c[0] - shapePoly.c[2]/2 - p_int[-1]

        # high order phase. Multiplied by sin(beta) because integration should actually happen in beam coordinates.
        high_order = (2 * np.pi / beam.lambda0 * high_order_temp *
                      np.sin(beta1 - self.delta))

        # plt.figure()
        # plt.plot(high_order)

        # scaling between grating z-axis and new beam coordinates
        scale = np.sin(beta1 - self.delta)

        # change coordinate systems to get proper low-order coefficients. Multiplied by sin(beta) because integration
        # should actually happen in beam coordinates.
        p_scaled = Util.poly_change_coords(p_int, scale) * np.sin(beta1 - self.delta)

        # Add 2nd order phase to p_scaled
        p_scaled[-3] += -1 / (2 * self.f)

        # scale the offset
        offset_scaled = offset * scale

        # account for any decentering
        p_centered = Util.recenter_coeff(p_scaled, offset_scaled)

        # 2nd order phase (factoring out pi/lambda)
        p2nd = 2 * p_centered[-3]
        # print('z: %.2f' % (1/p2nd))

        # 1st order phase (factoring out 2 pi/lambda)
        # (only add any 1st order phase due to de-centering since the rest is already accounted for in delta_k).
        p1st = p_centered[-2] - p_scaled[-2]
        # print(p1st)

        # figure out aperturing due to mirror's finite size
        z_mask = (np.abs(zi - self.dx / np.tan(total_alpha)) < self.length / 2).astype(float)
        y_mask = (np.abs(yi - self.dy) < self.width / 2).astype(float)

        beamInDotNormal = np.sum(k_i_full * m_x, axis=1)
        beamOutDotNormal = np.sum(k_f_full * m_x, axis=1)
        beamInDotHNormal = np.sum(k_i_full * c_normal, axis=1)

        C1, C2 = np.array(self.crystal.get_amplitude(beam.photonEnergy,
                                                     beamInDotNormal, beamOutDotNormal, beamInDotHNormal))

        if self.pol == 's':
            C = C1
        else:
            C = C2

        # handle beam re-pointing depending on the orientation
        if self.orientation == 0:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order) * C

            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.asymmetry_x(np.abs(np.sin(beta1) / np.sin(total_alpha)))
            beam.cx *= np.abs(np.sin(beta1) / np.sin(total_alpha))
            beam.x += beam.cx

            # add quadratic phase
            # beam.zx = 1 / (1 / beam.zx + p2nd)
            # beam.zx = 1 / p2nd
            new_zx = 1 / p2nd
            print('z: %.2f' % new_zx)
            beam.change_z(new_zx=new_zx)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1
            # beam.wavex = np.flipud(beam.wavex)

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=self.alpha+self.beta0)
            delta_ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = np.arcsin(delta_k[0] / np.cos(total_alpha)) + p1st
            # beam.ay += np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            delta_cx = 2 * self.dx * np.cos(self.alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 1:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order) * C

            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.asymmetry_y(np.abs(np.sin(beta1) / np.sin(total_alpha)))
            beam.cy *= np.abs(np.sin(beta1) / np.sin(total_alpha))
            beam.y += beam.cy

            # add quadratic phase
            # beam.zy = 1 / (1 / beam.zy + p2nd)
            # beam.zy = 1 / p2nd
            new_zy = 1 / p2nd
            beam.change_z(new_zy=new_zy)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1
            beam.wavey = np.flipud(beam.wavey)

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=self.alpha + self.beta0)
            delta_ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = -np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += -np.arcsin(delta_k[1])
            # beam.ay = np.arcsin(delta_k[0] / np.cos(total_alpha)) + p1st

            # adjust beam position due to mirror de-centering
            delta_cy = 2 * self.dx * np.cos(self.alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        elif self.orientation == 2:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavex *= z_mask * np.exp(1j * high_order) * C

            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.asymmetry_x(np.abs(np.sin(beta1) / np.sin(total_alpha)))
            beam.cx *= np.abs(np.sin(beta1) / np.sin(total_alpha))
            beam.x += beam.cx

            # add quadratic phase
            # beam.zx = 1 / (1 / beam.zx + p2nd)
            # beam.zx = 1 / p2nd
            new_zx = 1 / p2nd
            print('z: %.2f' % new_zx)
            beam.change_z(new_zx=new_zx)

            # take into account mirror reflection causing beam to invert
            beam.x *= -1
            # beam.wavex = np.flipud(beam.wavex)

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_azimuth=-self.alpha - self.beta0)
            # the following might be problematic...
            delta_ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st
            # let's see if this fixes it... NOPE
            # delta_ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])

            delta_ay = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st
            # beam.ay += -np.arcsin(delta_k[1])

            # adjust beam position due to mirror de-centering
            delta_cx = -2 * self.dx * np.cos(self.alpha)
            beam.cx = -beam.cx + delta_cx
            beam.x = beam.x + delta_cx

        elif self.orientation == 3:

            # modify beam's wave attribute by mirror aperture and phase error
            beam.wavey *= z_mask * np.exp(1j * high_order) * C

            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.asymmetry_y(np.abs(np.sin(beta1) / np.sin(total_alpha)))
            beam.cy *= np.abs(np.sin(beta1) / np.sin(total_alpha))
            beam.y += beam.cy

            # add quadratic phase
            # beam.zy = 1 / (1 / beam.zy + p2nd)
            # beam.zy = 1 / p2nd
            new_zy = 1 / p2nd
            beam.change_z(new_zy=new_zy)

            # take into account mirror reflection causing beam to invert
            beam.y *= -1
            # beam.wavey = np.flipud(beam.wavey)

            # adjust beam direction relative to properly aligned axis
            beam.rotate_nominal(delta_elevation=-self.alpha - self.beta0)
            delta_ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st
            # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
            delta_ax = np.arcsin(delta_k[1])
            beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

            # adjust beam direction relative to properly aligned axis
            # beam.ax += np.arcsin(delta_k[1])
            # beam.ay = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st

            # adjust beam position due to mirror de-centering
            delta_cy = -2 * self.dx * np.cos(self.alpha)
            beam.cy = -beam.cy + delta_cy
            beam.y = beam.y + delta_cy

        # plt.figure()
        # plt.plot(np.abs(beam.wavex))
        # plt.figure()
        # plt.plot(np.angle(beam.wavex))

        return


class Collimator:
    """
    Class for representing photon collimators. Basically just a circular aperture

    Attributes
    ----------
    name: str
        device name (e.g. PC1K0)
    diameter: float
        collimator diameter (meters)
    z: float
        z-location along beamline
    dx: float
        misalignment along golden trajectory x-axis
    dy: float
        misalignment along golden trajectory y-axis
    """

    def __init__(self, name, diameter=10e-3, z=None, dx=0, dy=0):
        """
        Method to initialize Collimator object
        :param name: str
            device name (e.g. PC1K0)
        :param diameter: float
            collimator diameter (meters)
        :param z: float
            z-location along beamline
        :param dx: float
            misalignment along golden trajectory x-axis
        :param dy: float
            misalignment along golden trajectory y-axis
        """
        # just set attributes
        self.name = name
        self.diameter = diameter
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.dx = dx
        self.dy = dy
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

    def multiply(self, beam):
        """
        Method to multiply the beam by the collimator aperture.
        :param beam: Beam
            Beam object to propagate through the collimator. Beam is modified by this method.
        :return: None
        """
        # define aperture in beam coordinates
        # aperture = (np.abs((beam.x - self.dx) ** 2 +
        # (beam.y - self.dy) ** 2) < (self.diameter / 2) ** 2).astype(float)
        # # multiply beam by aperture
        # beam.wave *= aperture
        print('not implemented in 1D')

    def propagate(self, beam):
        """
        Method that all optics need to have, just calls multiply here.
        :param beam: Beam
            Beam object to propagate through the collimator. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class Slit:
    """
    Class to represent a JJ slit. Just a rectangular aperture.

    Attributes
    ----------
    name: str
        device name (e.g. SL1K0)
    x_width: float
        full width of slits along beam x-axis
    y_width: float
        full width of slits along beam y-axis
    dx: float
        slit offset along x-axis
    dy: float
        slit offset along y-axis
    z: float
        z location along beamline
    """

    def __init__(self, name, x_width=5e-3, y_width=5e-3, dx=0, dy=0, z=None):
        """
        Method to create a Slit object.
        :param name: str
            device name (e.g. SL1K0)
        :param x_width: float
            full width of slits along beam x-axis
        :param y_width: float
            full width of slits along beam y-axis
        :param dx: float
            slit offset along x-axis
        :param dy: float
            slit offset along y-axis
        :param z: float
            z location along beamline
        """
        self.name = name
        self.x_width = x_width
        self.y_width = y_width
        self.dx = dx
        self.dy = dy
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

    def multiply(self, beam):
        """
        Method to propagate beam through the slit.
        :param beam: Beam
            Beam object to propagate through slits. Beam is modified by this method.
        :return: None
        """
        # define slit aperture in beam coordinates
        aperture_x = (np.abs(beam.x - self.dx) < self.x_width / 2).astype(float)
        aperture_y = (np.abs(beam.y - self.dy) < self.y_width / 2).astype(float)

        # multiply beam by aperture
        beam.wavex *= aperture_x
        beam.wavey *= aperture_y

    def propagate(self, beam):
        """
        Method to propagate beam through aperture. Calls multiply.
        :param beam: Beam
            Beam object to propagate through slits. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)

    def set_x_width(self, width):
        print('set width')
        self.x_width = width

    def set_y_width(self, width):
        print('set width')
        self.y_width = width


class Drift:
    """
    Class to represent zeros space between beamline components. This is generally where beam propagation happens.

    Attributes
    ----------
    name: str
        Drift object name. Generally these are just numbered based on their z location along a beamline (e.g. drift0).
    upstream_component: object
        Can be any object defined in this (optics) module. Sets the upstream boundary of the drift section.
        Only requirement is that this object must have an attribute with name "z".
    downstream_component: object
        Can be any object defined in this (optics) module. Sets the downstream boundary of the drift section.
        Only requirement is that this object must have an attribute with name "z".
    dz: float
        Distance between upstream_component and downstream_component
    z: float
        Location of drift along beamline. This is set to be the average position between upstream_component and
        downstream_component.
    x: float
        Location of drift along beamline. This is set to be the average position between upstream_component and
        downstream_component.
    y: float
        Location of drift along beamline. This is set to be the average position between upstream_component and
        downstream_component.
    """

    def __init__(self, name, upstream_component=None, downstream_component=None):
        """
        Method to create a Drift object.
        :param name: str
            Drift object name. Generally these are just numbered based on their z location along a beamline
            (e.g. drift0).
        :param upstream_component: object
            Can be any object defined in this (optics) module. Sets the upstream boundary of the drift section.
            Only requirement is that this object must have an attribute with name "z".
        :param downstream_component: object
            Can be any object defined in this (optics) module. Sets the downstream boundary of the drift section.
            Only requirement is that this object must have an attribute with name "z".
        """
        # set some attributes
        self.name = name
        self.upstream_component = upstream_component
        self.downstream_component = downstream_component
        # calculate distance-related attributes
        dx = downstream_component.global_x - upstream_component.global_x
        dy = downstream_component.global_y - upstream_component.global_y
        dz = downstream_component.z - upstream_component.z
        self.dz = np.sqrt(dx**2 + dy**2 + dz**2)
        # self.dz = downstream_component.z - upstream_component.z
        self.z = (downstream_component.z + upstream_component.z) / 2.
        self.global_x = 0
        self.global_y = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

    def propagate(self, beam):
        """
        Method to propagate through a Drift section
        :param beam: Beam
            Beam object to propagate through the drift section. Beam is modified.
        :return: None
        """
        # propagate the beam along the full length of the Drift.

        # can put re-calculation of distance here
        # get beam k
        k = beam.get_k()

        # deal with the case that the beam is propagating perpendicular to the z direction
        # if k[2] == 0:
        #     if issubclass(type(self.downstream_component), Mirror):
        #         alpha = self.downstream_component.global_alpha
        #
        print('global_x %.2f' % beam.global_x)
        print('global_y %.2f' % beam.global_y)

        if issubclass(type(self.downstream_component), Mirror):
            # beam global coordinates are currently on surface of upstream component
            # get global alpha for mirror
            alpha = self.downstream_component.global_alpha
            z_m = self.downstream_component.z
            x_m = self.downstream_component.global_x
            y_m = self.downstream_component.global_y

            mirror_center = np.array([x_m, y_m, z_m])

            normal = self.downstream_component.normal
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]
            kx = k[0]
            ky = k[1]
            kz = k[2]

            if self.downstream_component.orientation==0 or self.downstream_component.orientation==1:
                mirror_center += normal*self.downstream_component.dx
            else:
                mirror_center -= normal*self.downstream_component.dx

            x_m = mirror_center[0]
            y_m = mirror_center[1]
            z_m = mirror_center[2]

            # find z location where two lines intersect
            # if self.downstream_component.orientation == 0:
            #     # z_intersect = ((-k[0]/k[2]*beam.global_z + beam.global_x + np.tan(alpha)*z_m - x_m)/
            #     #                (np.tan(alpha) - k[0]/k[2]))
            #
            #
            # elif self.downstream_component.orientation == 1:
            #     z_intersect = ((-k[1]/k[2]*beam.global_z + beam.global_y + np.tan(alpha)*z_m - y_m)/
            #                    (np.tan(alpha) - k[1]/k[2]))
            #
            # elif self.downstream_component.orientation == 2:
            #
            #     z_intersect = ((-k[0] / k[2] * beam.global_z + beam.global_x + np.tan(alpha) * z_m - x_m) /
            #                    (np.tan(alpha) - k[0] / k[2]))
            #
            # else:
            #
            #     z_intersect = ((-k[1] / k[2] * beam.global_z + beam.global_y + np.tan(alpha) * z_m - y_m) /
            #                    (np.tan(alpha) - k[1] / k[2]))
            z_intersect = ((nx*kx*beam.global_z - nx*kz*(beam.global_x-x_m) +
                           ny*ky*beam.global_z-ny*kz*(beam.global_y-y_m) + nz*kz*z_m)/
                           (nx*kx+ny*ky+nz*kz))

        else:
            z_m = self.downstream_component.z
            x_m = self.downstream_component.global_x
            y_m = self.downstream_component.global_y

            normal = self.downstream_component.zhat
            nx = normal[0]
            ny = normal[1]
            nz = normal[2]
            kx = k[0]
            ky = k[1]
            kz = k[2]
            z_intersect = ((nx * kx * beam.global_z - nx * kz * (beam.global_x - x_m) +
                            ny * ky * beam.global_z - ny * kz * (beam.global_y - y_m) + nz * kz * z_m) /
                           (nx * kx + ny * ky + nz * kz))

        x_intersect = k[0] / k[2] * (z_intersect - beam.global_z) + beam.global_x
        print('x intersect: %.4e' % x_intersect)
        print('component x: %.4e' % self.downstream_component.global_x)
        y_intersect = k[1] / k[2] * (z_intersect - beam.global_z) + beam.global_y
        print('y intersect: %.4e' % y_intersect)
        print('component y: %.4e' % self.downstream_component.global_y)
        print('z intersect: %.4e' % z_intersect)
        print('component z: %.4e' % self.downstream_component.z)
        dx = x_intersect - beam.global_x
        dy = y_intersect - beam.global_y
        dz = z_intersect - beam.global_z
        # re-calculate propagation distance
        old_z = np.copy(self.dz)

        self.dz = np.sqrt(dx**2 + dy**2 + dz**2)
        print('delta z: %.2f' % ((self.dz - old_z)*1e6))

        # beam.global_x = x_intersect
        # beam.global_y = y_intersect
        # beam.global_z =

        beam.beam_prop(self.dz)


class PPM:
    """
    Class to represent profile monitor output from PPMs.

    Attributes
    ----------
    name: str
        device name (e.g. IM1K4)
    FOV: float
        width of the (restricted to be square) field of view
    N: int
        number of pixels across the image. Image is NxN.
    dx: float
        PPM pixel size
    z: float
        z location along beamline
    blur: bool
        Blur beam intensity prior to interpolation if True, simulating blurring due to finite resolution of
        microscope. Mainly important for wavefront sensor profile monitors.
    distort: bool
        Distort image consistent with microscope distortion if True. Mainly important for wavefront sensor
        profile monitors.
    view_angle_x: float
        Set viewing angle (in degrees) relative to beam propagation axis. Defined as angle from glancing incidence,
        normal incidence is 90 degrees.
    view_angle_y: float
        Set viewing angle (in degrees) relative to beam propagation axis. Defined as angle from glancing incidence,
        normal incidence is 90 degrees.
    x: (N,) ndarray
        profile monitor x coordinates
    y: (N,) ndarray
        profile monitor y coordinates
    profile: (N,N) ndarray
        Calculated beam profile (normalized intensity) at profile monitor.
    x_lineout: (N,) ndarray
        Calculated horizontal lineout (normalized).
    y_lineout: (N,) ndarray
        Calculated vertical lineout (normalized).
    cx: float
        Horizontal beam centroid on PPM.
    cy: float
        Vertical beam centroid on PPM.
    wx: float
        Horizontal beam FWHM on PPM. Based on Gaussian fit (or calculated from second moment if fit fails).
    wy: float
        Vertical beam FWHM on PPM. Based on Gaussian fit (or calculated from second moment if fit fails).
    resolution: float
        PPM optical resolution. Used if blur is True.
    xline: TalbotLineout object
        See pitch module. Calculates pitch of Talbot pattern for WFS case.
    yline: TalbotLineout object
        See pitch module. Calculates pitch of Talbot pattern for WFS case.
    """

    def __init__(self, name, FOV=10e-3, z=None, N=2048, blur=False,
                 view_angle_x=90, view_angle_y=90, resolution=5e-6, distort=False, xoffset=0, yoffset=0):
        """
        Method to initialize a PPM.
        :param name: str
            device name (e.g. IM1K4)
        :param FOV: float
            width of the (restricted to be square) field of view (m)
        :param z: float
            z location along beamline
        :param N: int
            number of pixels across the image. Image is nxn.
        :param blur: bool
            Blur beam intensity prior to interpolation if True, simulating blurring due to finite resolution of
            microscope. Mainly important for wavefront sensor profile monitors.
        :param view_angle_x: float
            Set horizontal viewing angle (in degrees) relative to beam propagation axis.
            Defined as angle from glancing incidence, normal incidence is 90 degrees.
        :param view_angle_y: float
            Set vertical viewing angle (in degrees) relative to beam propagation axis.
            Defined as angle from glancing incidence, normal incidence is 90 degrees.
        :param resolution: float
            PPM optical resolution. Used if blur is True.
        :param distort: bool
            Distort image consistent with microscope distortion if True. Mainly important for wavefront sensor
            profile monitors.
        """

        # set some attributes
        self.N = N
        dx = FOV / N
        self.dx = dx
        self.xoffset=xoffset
        self.yoffset=yoffset
        self.FOV = FOV
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.name = name
        self.blur = blur
        self.distort = distort
        self.view_angle_x = view_angle_x
        self.view_angle_y = view_angle_y
        self.resolution = resolution
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

        # calculate PPM coordinates
        self.x = np.linspace(-N / 2, N / 2 - 1, N) * dx + xoffset
        # self.y = np.copy(self.x) + yoffset
        self.y = np.linspace(-N / 2, N / 2 -1, N) * dx + yoffset

        # initialize some attributes
        self.profile = np.zeros((N, N))
        self.x_phase = np.zeros(N)
        self.y_phase = np.zeros(N)
        self.zx = 0
        self.zy = 0
        self.cx_beam = 0
        self.cy_beam = 0
        self.x_lineout = np.zeros(N)
        self.y_lineout = np.zeros(N)
        self.xline = None
        self.yline = None
        self.cx = 0.0
        self.cy = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.lambda0 = 0.0
        self.group_delay = 0

    def reset(self):
        # initialize some attributes
        self.profile = np.zeros((self.N, self.N))
        self.x_lineout = np.zeros(self.N)
        self.y_lineout = np.zeros(self.N)
        self.xline = None
        self.yline = None
        self.cx = 0.0
        self.cy = 0.0
        self.wx = 0.0
        self.wy = 0.0
        self.lambda0 = 0.0

    def beam_analysis(self, line_x, line_y):
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

        # normalize lineouts
        line_x = line_x / np.max(line_x)
        line_y = line_y / np.max(line_y)

        # set 20% threshold
        thresh_x = np.max(line_x) * .2
        thresh_y = np.max(line_y) * .2
        # subtract threshold and set everything below to zero
        norm_x = line_x - thresh_x
        norm_x[norm_x < 0] = 0
        # re-normalize
        norm_x = norm_x / np.max(norm_x)

        # subtract threshold and set everything below to zero
        norm_y = line_y - thresh_y
        norm_y[norm_y < 0] = 0
        # re-normalize
        norm_y = norm_y / np.max(norm_y)

        # calculate centroids
        cx = np.sum(norm_x * self.x) / np.sum(norm_x)
        cy = np.sum(norm_y * self.y) / np.sum(norm_y)

        # calculate second moments. Converted to microns to help with fitting
        sx = np.sqrt(np.sum(norm_x * (self.x - cx) ** 2) / np.sum(norm_x)) * 1e6
        sy = np.sqrt(np.sum(norm_y * (self.y - cy) ** 2) / np.sum(norm_y)) * 1e6
        # conversion factor from sigma to fwhm
        fwx_guess = sx * 2.355
        fwy_guess = sy * 2.355

        # initial guess for Gaussian fit
        guessx = [cx * 1e6, sx]
        guessy = [cy * 1e6, sy]

        # Gaussian fitting. Using try/except to deal with any fitting errors
        try:
            # only fit in the region where we have signal
            mask = line_x > .1
            # Gaussian fit using Scipy curve_fit. Using only data that has >10% of the max
            px, pcovx = optimize.curve_fit(Util.fit_gaussian, self.x[mask] * 1e6, line_x[mask], p0=guessx)
            # set sx to sigma from the fit if successful.
            sx = px[1]
        except ValueError:
            print('Some of the data contained NaNs or options were incompatible. Using second moment for width.')
        except RuntimeError:
            print('Least squares minimization failed. Using second moment for width.')
        except TypeError:
            print('Some problem with input')

        try:
            # only fit in the region where we have signal
            mask = line_y > .1
            # Gaussian fit using Scipy curve_fit. Using only data that has >10% of the max
            py, pcovy = optimize.curve_fit(Util.fit_gaussian, self.y[mask] * 1e6, line_y[mask], p0=guessy)
            # set sy to sigma from the fit if successful.
            sy = py[1]
        except ValueError:
            print('Some of the data contained NaNs or options were incompatible. Using second moment for width.')
        except RuntimeError:
            print('Least squares minimization failed. Using second moment for width.')
        except TypeError:
            print('Some problem with input')

        # conversion factor from sigma to FWHM. Also convert back to meters.
        fwhm_x = sx * 2.355 / 1e6
        fwhm_y = sy * 2.355 / 1e6

        return cx, cy, fwhm_x, fwhm_y, fwx_guess, fwy_guess

    def add_profile(self, profile):
        self.profile += profile
        # calculate horizontal lineout
        self.x_lineout = np.sum(self.profile, axis=0)
        # calculate vertical lineout
        self.y_lineout = np.sum(self.profile, axis=1)

        # calculate centroids and beam widths
        self.cx, self.cy, self.wx, self.wy, wx2, xy2 = self.beam_analysis(self.x_lineout, self.y_lineout)


    def calc_profile(self, beam):
        """
        Method to calculate the beam profile at the PPM screen.
        :param beam: Beam
            Beam object for viewing at PPM location. The Beam object is not modified by this method.
        :return: None
        """

        # Calculate intensity from complex beam
        profilex = np.abs(beam.wavex) ** 2
        profiley = np.abs(beam.wavey) ** 2

        # check if either profile contains NANs
        # if np.sum(np.isnan(profilex))>0:
        #     profilex = np.zeros_like(profilex)
        # if np.sum(np.isnan(profiley))>0:
        #     profiley = np.zeros_like(profiley)

        # coordinate scaling due to off-axis viewing angle
        scaling_x = 1 / np.sin(self.view_angle_x * np.pi / 180)
        scaling_y = 1 / np.sin(self.view_angle_y * np.pi / 180)

        # if blurring is used, apply a gaussian filter
        if self.blur:
            # calculate blur widths in pixels, based on beam's pixel size
            x_width = self.resolution / beam.dx
            y_width = self.resolution / beam.dy
            # apply blurring using ndimage gaussian_filter
            profilex = ndimage.filters.gaussian_filter1d(profilex, x_width)
            profiley = ndimage.filters.gaussian_filter1d(profiley, y_width)

        # get beam coordinates for interpolation
        x = beam.x
        y = beam.y

        # interpolating function from np.interp (allowing for flipped coordinates)
        profilex_interp = Util.interp_flip(self.x, x * scaling_x, profilex)
        profiley_interp = Util.interp_flip(self.y, y * scaling_y, profiley)

        # beam phase
        x_phase = np.unwrap(np.angle(beam.wavex))
        y_phase = np.unwrap(np.angle(beam.wavey))

        # check for nans
        # if np.sum(np.isnan(x_phase))>0:
        #     x_phase = np.zeros_like(x_phase, dtype=complex)
        # if np.sum(np.isnan(y_phase))>0:
        #     y_phase = np.zeros_like(y_phase, dtype=complex)

        # interpolating function from np.interp (allowing for flipped coordinates)
        self.x_phase = Util.interp_flip(self.x, x * scaling_x, x_phase)
        self.y_phase = Util.interp_flip(self.y, y * scaling_y, y_phase)

        self.group_delay = beam.group_delay

        # add linear phase (centered on beam)
        self.x_phase += 2 * np.pi / beam.lambda0 * beam.ax * (self.x - beam.cx)
        self.y_phase += 2 * np.pi / beam.lambda0 * beam.ay * (self.y - beam.cy)

        # self.x_phase += 2 * np.pi / beam.lambda0 * beam.ax * (self.x)
        # self.y_phase += 2 * np.pi / beam.lambda0 * beam.ay * (self.y)

        # multiply two dimensions together to get the 2d profile
        self.profile = np.reshape(profiley_interp, (self.N, 1)) * np.reshape(profilex_interp, (1, self.N))

        # if beam is not focused get quadratic phase information
        if not beam.focused_x:
            self.zx = beam.zx
            self.cx_beam = beam.cx
        if not beam.focused_y:
            self.zy = beam.zy
            self.cy_beam = beam.cy

        # calculate horizontal lineout
        self.x_lineout = np.sum(self.profile, axis=0)
        # calculate vertical lineout
        self.y_lineout = np.sum(self.profile, axis=1)

        # calculate centroids and beam widths
        self.cx, self.cy, self.wx, self.wy, wx2, xy2 = self.beam_analysis(self.x_lineout, self.y_lineout)

        # get beam wavelength
        self.lambda0 = beam.lambda0

        # distortion of image, for simulating distortion effect for wavefront sensor
        if self.distort:
            # K is the distortion parameter. Distorted coordinates x_d are defined as x_d = x_u * (1 + K * r^2), where
            # x_u are the undistorted coordinates. Negative for barrel distortion. Use 1% distortion at edge of field.
            K = -.01/(self.FOV/2)**2
            # get a meshgrid for the coordinates
            xx,yy = np.meshgrid(self.x,self.y)
            # RectBivariateSpline takes flattened coordinates
            xx = xx.flatten()
            yy = yy.flatten()
            # calculate radius squared
            rr = xx**2 + yy**2
            # generate interpolating function
            f = interpolate.RectBivariateSpline(self.x, self.y, self.profile)
            # evaluate interpolating function to get distorted profile.
            self.profile = np.reshape(f.ev(xx * (1 + K * rr), yy * (1 + K * rr)), (self.N, self.N), order='F')

    def propagate(self, beam):
        """
        Method to propagate beam through PPM. Calls calc_profile.
        :param beam: Beam
            Beam object for viewing at PPM location. The Beam object is not modified by this method.
        :return: None
        """
        self.calc_profile(beam)

    def view_vertical(self, ax_y=None, normalized=True, log=False, show_fit=True, legend=False, label='Lineout'):
        """
        Method to view
        :param normalized: whether to normalize the lineout
        :return:
        """

        gaussian_fit = np.exp(-(self.y - self.cy) ** 2 / 2 / (self.wy / 2.355) ** 2)

        if ax_y is None:
            # generate the figure
            plt.figure()
            ax_y = plt.subplot2grid((1,1), (0, 0))
        if normalized:
            # show the vertical lineout (distance in microns)
            if log:
                ax_y.semilogy(self.y * 1e6, self.y_lineout / np.max(self.y_lineout), label=label)
            else:
                ax_y.plot(self.y * 1e6, self.y_lineout / np.max(self.y_lineout), label=label)
                ax_y.set_ylim(0, 1.05)
            ax_y.set_ylabel('Intensity (normalized)')
        else:
            # show the vertical lineout (distance in microns)
            if log:
                ax_y.semilogy(self.y * 1e6, self.y_lineout, label=label)
            else:
                ax_y.plot(self.y * 1e6, self.y_lineout, label=label)
            gaussian_fit *= np.max(self.y_lineout)
            ax_y.set_ylabel('Intensity (arbitrary units)')
        # also plot the Gaussian fit
        if show_fit:
            if log:
                ax_y.semilogy(self.y*1e6, gaussian_fit, label='fit')
            else:
                ax_y.plot(self.y * 1e6, gaussian_fit, label='fit')
        if legend:
            ax_y.legend()
        ax_y.set_xlabel('Y Coordinates (\u03BCm)')
        # show a grid
        ax_y.grid(True)
        # set limits


        return ax_y

    def view_beam(self):
        """
        Method to view beam after the fact. Will be zero intensity everywhere if calc_profile (or propagate)
        haven't been called yet.
        :return axes_handles: list of matplotlib axes handles
            Handles to the axes in the generated figure. Listed in the following order [profile,x_lineout,y_lineout].
        """

        # minima and maxima of the field of view (in microns) for imshow extent
        minx = np.round(np.min(self.x) * 1e6)
        maxx = np.round(np.max(self.x) * 1e6)
        miny = np.round(np.min(self.y) * 1e6)
        maxy = np.round(np.max(self.y) * 1e6)

        units = 'microns'
        mult = 1e6

        all_extrema = np.array([minx, maxx, miny, maxy])
        min_extrema = np.min(np.abs(all_extrema))
        if min_extrema < 1:
            minx = np.round(np.min(self.x) * 1e9)
            maxx = np.round(np.max(self.x) * 1e9)
            miny = np.round(np.min(self.y) * 1e9)
            maxy = np.round(np.max(self.y) * 1e9)
            units = 'nm'
            mult = 1e9

        # generate the figure
        plt.figure(figsize=(8, 8))

        # generate the axes, in a grid
        ax_profile = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=3)
        ax_y = plt.subplot2grid((4, 4), (0, 3), rowspan=3)
        ax_x = plt.subplot2grid((4, 4), (3, 0), colspan=3)

        # show the image, with positive y at the top of the figure
        ax_profile.imshow(np.flipud(self.profile), extent=(minx, maxx, miny, maxy), cmap=plt.get_cmap('gnuplot'))
        # label coordinates
        ax_profile.set_xlabel('X coordinates (%s)' % units)
        ax_profile.set_ylabel('Y coordinates (%s)' % units)
        ax_profile.set_title(self.name)

        # show the vertical lineout (distance in microns)
        ax_y.plot(self.y_lineout/np.max(self.y_lineout), self.y * mult)
        # also plot the Gaussian fit
        ax_y.plot(np.exp(-(self.y - self.cy) ** 2 / 2 / (self.wy / 2.355) ** 2), self.y * mult)
        # show a grid
        ax_y.grid(True)
        # set limits
        ax_y.set_xlim(0, 1.05)

        # show the horizontal lineout (distance in microns)
        ax_x.plot(self.x * mult, self.x_lineout/np.max(self.x_lineout))
        # also plot the Gaussian fit
        ax_x.plot(self.x * mult, np.exp(-(self.x - self.cx) ** 2 / 2 / (self.wx / 2.355) ** 2))
        # show a grid
        ax_x.grid(True)
        # set limits
        ax_x.set_ylim(0, 1.05)

        # add some annotations with beam centroid and FWHM
        ax_y.text(.6, .1 * np.max(self.y * mult), 'centroid: %.2f %s' % (self.cy * mult, units), rotation=-90)
        ax_y.text(.3, .1 * np.max(self.y * mult), 'width: %.2f %s' % (self.wy * mult, units), rotation=-90)
        ax_x.text(-.9 * np.max(self.x * mult), .6, 'centroid: %.2f %s' % (self.cx * mult, units))
        ax_x.text(-.9 * np.max(self.x * mult), .3, 'width: %.2f %s' % (self.wx * mult, units))

        # tight layout to make sure we're not cutting out anything
        plt.tight_layout()

        # bundle handles in a list
        axes_handles = [ax_profile, ax_x, ax_y]

        return axes_handles

    def get_x(self):
        """
        Returns x
        :return x: (N,) ndarray
            horizontal coordinates for PPM
        """
        return self.x

    def get_y(self):
        """
        Returns y
        :return y: (N,) ndarray
            vertical coordinates for PPM
        """
        return self.y

    def get_x_centroid(self):
        """
        Return calculated centroid
        :return cx: float
             Horizontal centroid
        """
        return self.cx

    def get_y_centroid(self):
        """
        Return calculated centroid
        :return cy: float
            Vertical centroid
        """
        return self.cy

    def get_x_width(self):
        """
        Return calculated horizontal FWHM
        :return wx: float
            Horizontal FWHM
        """
        return self.wx

    def get_y_width(self):
        """
        Return calculated vertical FWHM
        :return wy: float
            Vertical FWHM
        """
        return self.wy

    def get_profile_x(self):
        """
        Return horizontal lineout
        :return x_lineout: (N,) ndarray
            Horizontal lineout
        """
        return self.x_lineout

    def get_profile_y(self):
        """
        Return vertical lineout
        :return y_lineout: (N,) ndarray
            Vertical lineout
        """
        return self.y_lineout

    def get_2d_profile(self):
        """
        Return 2D profile
        :return profile: (N,N) ndarray
            2D beam profile at PPM
        """
        return self.profile

    def get_FOV(self):
        """
        Return PPM field of view
        :return FOV: float
            Width of square field of view
        """
        return self.FOV

    def retrieve_wavefront(self, wfs):
        """
        Method to calculate wavefront in the case where there is a wavefront sensor upstream of the PPM.
        :param wfs: WFS object
            Grating structure that generates Talbot interferometry patterns. Passed to this method to gain access
            to its attributes.
        :return wfs_data: dict
            Includes the following entries
            x_prime: (M,) ndarray
                Horizontal coordinates for retrieved high-order phase
            y_prime: (N,) ndarray
                Vertical coordinates for retrieved high-order phase
            x_res: (M,) ndarray
                Horizontal residual phase (>2nd order) at points in x_prime
            y_res: (N,) ndarray
                Vertical residual phase (>2nd order) at points in y_prime
            coeff_x: (k,) ndarray
                Legendre coefficients for horizontal phase lineout
            coeff_y: (k,) ndarray
                Legendre coefficients for vertical phase lineout
            z2x: float
                Distance to horizontal focus
            z2y: float
                Distance to vertical focus
        """

        print('retrieving wavefront')

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Distance from wavefront sensor to PPM
        zT = self.z - wfs.z

        # magnification of Talbot pattern
        mag = (zT + wfs.f0) / wfs.f0

        # number of pixels to sum across to get lineout
        lineout_width = int(wfs.pitch / self.dx * 5 * mag)

        # lineout boundaries in pixels (distance from center)
        x_lim = int(self.wx/self.dx)
        y_lim = int(self.wy/self.dx)

        # calculated beam center in pixels
        x_center = int(self.cx/self.dx) + self.N/2
        y_center = int(self.cy/self.dx) + self.N/2

        # get lineouts from 2d profile
        lineout_x = np.sum(self.profile[int(y_center - lineout_width / 2):int(y_center + lineout_width / 2),
                           int(x_center - x_lim):int(x_center+x_lim)], axis=0)
        lineout_y = np.sum(self.profile[int(y_center-y_lim):int(y_center+y_lim),
                           int(x_center - lineout_width / 2):int(x_center + lineout_width / 2)], axis=1)

        # expected spatial frequency of Talbot pattern (1/m)
        peak = 1. / mag / wfs.pitch * fraction

        # spatial frequency now in units of (1/pixels)
        fc = peak * self.dx

        # calculate pitch from lineouts. See pitch module.
        print('getting lineouts')
        self.xline = TalbotLineout(lineout_x, fc, fraction, pad=True)
        self.yline = TalbotLineout(lineout_y, fc, fraction, pad=True)

        # parameters for calculating Legendre coefficients
        param = {
                "dg": wfs.pitch,  # wavefront sensor pitch (m)
                "fraction": fraction,  # wavefront sensor fraction
                "dx": self.dx,  # PPM pixel size
                "zT": zT,  # distance between WFS and PPM
                "lambda0": self.lambda0  # beam wavelength
                }

        # calculate Legendre coefficients
        print('getting Legendre coefficients')
        z_x, coeff_x, x_prime, x_res = self.xline.get_legendre(param)
        z_y, coeff_y, y_prime, y_res = self.yline.get_legendre(param)
        print('found Legendre coefficients')

        # pixel size for retrieved wavefront
        dx_prime = x_prime[1] - x_prime[0]
        dy_prime = y_prime[1] - y_prime[0]

        # re-center residual phase coordinates on beam center
        x_prime += (x_center-self.N/2) * dx_prime
        y_prime -= (y_center-self.N/2) * dy_prime

        # convert coordinates to microns
        x_prime = x_prime * 1e6
        y_prime = y_prime * 1e6

        # print calculated distance to focus
        print('Distance to source: '+str(z_x))
        print('Distance to source: '+str(z_y))

        # output. See method docstring for descriptions.
        wfs_data = {
                'x_res': x_res,
                'x_prime': x_prime,
                'y_res': y_res,
                'y_prime': y_prime,
                'coeff_x': coeff_x,
                'coeff_y': coeff_y,
                'z2x': z_x,
                'z2y': z_y
                }

        return wfs_data

    def complex_beam(self):
        # multiply two dimensions together to get the 2d profile
        phase_2D = np.reshape(np.exp(1j * self.y_phase), (self.N, 1)) * np.reshape(np.exp(1j * self.x_phase),
                                                                                   (1, self.N))

        # reshape into 2 dimensional representation
        complex_beam = np.sqrt(self.profile) * phase_2D

        return complex_beam, self.group_delay, self.zx, self.zy, self.cx_beam, self.cy_beam


class CRL:
    """
    Class to represent parabolic compound refractive lenses (CRLs). This is a 1D implementation so the CRLs are square.

    Attributes
    ----------
    name: str
        Name of the device (e.g. CRL1)
    diameter: float
        Diameter beyond which the lenses absorb all photons. (meters)
    roc: float
        Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
    E0: float or None
        photon energy in eV for calculating radius of curvature for a given focal length
    f: float or None
        focal length in meters for calculating radius of curvature for a given energy
    material: str
        Lens material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
    dx: float
        Lens de-centering along beam's x-axis.
    dy: float
        Lens de-centering along beam's y-axis.
    z: float
        z location of lenses along beamline.
    energy: (N,) ndarray
        List of photon energies from CXRO file (eV).
    delta: (N,) ndarray
        Real part of index of refraction. n = 1 - delta + 1j * beta
    beta: (N,) ndarray
        Imaginary part of index of refraction. n = 1 - delta + 1j * beta
    """

    def __init__(self, name, diameter=300e-6, roc=50e-6, E0=None, f=None, material='Be', z=0, dx=0, orientation=0):
        """
        Method to create a CRL object.
        :param name: str
            Name of the device (e.g. CRL1)
        :param diameter: float
            Diameter beyond which the lenses absorb all photons. (meters)
        :param roc: float
            Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
        :param E0: float
            photon energy for calculating radius of curvature for a given focal length (eV)
        :param f: float
            focal length for calculating radius of curvature for a given energy (meters)
        :param material: str
            Lens material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
        :param z: float
            z location of lenses along beamline.
        :param dx: float
            Lens de-centering along beam's x-axis.
        :param orientation: int
            Whether or not this is a horizontal or vertical lens (0 for horizontal, 1 for vertical).
        """

        # set some attributes
        self.name = name
        self.diameter = diameter
        self.roc = roc
        self.E0 = E0
        self.f = f
        self.material = material
        self.dx = dx
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.orientation = orientation
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

        # get file name of CXRO data
        filename = os.path.join(os.path.dirname(__file__), 'cxro_data/%s.csv' % self.material)

        # load in CXRO data
        cxro_data = np.genfromtxt(filename, delimiter=',')
        self.energy = cxro_data[:, 0]
        self.delta = cxro_data[:, 1]
        self.beta = cxro_data[:, 2]

        # if these arguments are given then override default roc or even roc argument
        if self.f is not None and self.E0 is not None:
            # interpolate to find index of refraction at beam's energy
            delta = np.interp(self.E0, self.energy, self.delta)
            # calculate radius of curvature based on f and delta
            self.roc = 2 * delta * self.f

        elif self.E0 is not None:
            # interpolate to find index of refraction at beam's energy
            delta = np.interp(self.E0, self.energy, self.delta)
            self.f = self.roc/2/delta


    def multiply(self, beam):
        """
        Method to propagate beam through CRL
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """

        if self.orientation == 0:
            beamx = beam.x
            beamz = beam.zx
            beamc = beam.cx
        else:
            beamx = beam.y
            beamz = beam.zy
            beamc = beam.cy

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # CRL thickness (for now assuming perfect lenses but might add aberrations later)
        # thickness = 2 * self.roc * (1 / 2 * ((beam.x - self.dx) ** 2 + (beam.y - self.dy) ** 2) / self.roc ** 2)
        thickness = 2 * self.roc * (1 / 2 * ((beamx - self.dx) ** 2) / self.roc ** 2)

        # lens aperture
        mask = (((beamx - self.dx) ** 2) < (self.diameter / 2) ** 2).astype(float)

        # subtract 2nd order and linear terms
        phase = -beam.k0 * delta * (thickness - 2 / 2 / self.roc * ((beamx - self.dx) ** 2))

        # 2nd order
        p2 = -beam.k0 * delta * 2 / 2 / self.roc
        # 1st order
        p1_x = p2 * 2 * (beamc - self.dx)

        # lens transmission based on beta and thickness profile
        # phase shift at center of beam
        phase_shift = np.interp(beamc, beamx, thickness)*delta*2*np.pi/beam.lambda0

        transmission = np.exp(-beam.k0 * beta * thickness) * np.exp(1j * phase) * mask# * np.exp(1j*phase_shift)

        # adjust beam properties
        new_zx = 1 / (1 / beamz + p2 * beam.lambda0 / np.pi)

        if self.orientation == 0:
            beam.change_z(new_zx=new_zx)
            delta_ax = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ax=delta_ax)
            # beam.ax += p1_x * beam.lambda0 / 2 / np.pi
            # multiply beam by CRL transmission function and any high order phase
            beam.wavex *= transmission
        else:
            beam.change_z(new_zy=new_zx)
            delta_ay = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ay=delta_ay)
            # beam.ay += p1_x * beam.lambda0 / 2 / np.pi
            # multiply beam by CRL transmission function and any high order phase
            beam.wavey *= transmission

        print('focal length: %.2f' % (-1/(p2*beam.lambda0/np.pi)))

    def propagate(self, beam):
        """
        Method to propagate beam through CRL. Calls multiply.
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class Prism:
    """
    Class to represent a hard X-ray prism.

    Attributes
    ----------
    name: str
        Name of the device (e.g. PRM1)
    x_width: float
        horizontal size of prism
    y_width: float
        vertical size of prism
    slope: float
        thickness gradient (dimensionless)
    orientation: int
        defined as 0: thicker to +x; 1: thicker to +y; 2: thicker to -x; 3: thicker to -y
    material: str
        default to Beryllium
    z: float
        location along beamline
    dx: float
        horizontal de-centering relative to beam axis
    dy: float
        vertical de-centering relative to beam axis
    """

    def __init__(self, name, x_width=100e-6, y_width=100e-6, slope=100e-6, material='Be', z=0, dx=0, dy=0, orientation=0):
        """
        Method to create a prism
        Parameters
        ----------
        name: str
            Name of the device (e.g. PRM1)
        x_width: float
            horizontal size of prism
        y_width: float
            vertical size of prism
        slope: float
            thickness gradient (dimensionless)
        material: str
            default to Beryllium
        z: float
            location along beamline
        dx: float
            horizontal de-centering relative to beam axis
        dy: float
            vertical de-centering relative to beam axis
        orientation: int
            defined as 0: thicker to +x; 1: thicker to +y; 2: thicker to -x; 3: thicker to -y
        """

        # set some attributes
        self.name = name
        self.x_width = x_width
        self.y_width = y_width
        self.slope = slope
        self.material = material
        self.dx = dx
        self.dy = dy
        self.z = z
        self.orientation = orientation
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

        # get file name of CXRO data
        filename = os.path.join(os.path.dirname(__file__), 'cxro_data/%s.csv' % self.material)

        # load in CXRO data
        cxro_data = np.genfromtxt(filename, delimiter=',')
        self.energy = cxro_data[:, 0]
        self.delta = cxro_data[:, 1]
        self.beta = cxro_data[:, 2]

    def multiply(self, beam):

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # prism aperture
        aperture_x = (np.abs(beam.x - self.dx) < self.x_width / 2).astype(float)
        aperture_y = (np.abs(beam.y - self.dy) < self.y_width / 2).astype(float)

        thickness = np.zeros_like(beam.x)
        p1_x = 0
        p1_y = 0

        if self.orientation == 0:
            thickness = self.slope * (beam.x - self.dx + self.x_width / 2)
            p1_x = -delta * self.slope
        elif self.orientation == 1:
            thickness = self.slope * (beam.y - self.dy + self.y_width / 2)
            p1_y = -delta * self.slope
        elif self.orientation == 2:
            thickness = -self.slope * (beam.x - self.dx - self.x_width / 2)
            p1_x = delta * self.slope
        elif self.orientation == 3:
            thickness = -self.slope * (beam.y - self.dy - self.y_width / 2)
            p1_y = delta * self.slope

        # prism transmission based on beta and thickness profile
        transmission = np.exp(-beam.k0 * beta * thickness)

        # multiply by transmission
        # if self.orientation == 0:
        #     beam.wavex *= (transmission * aperture_x)
        #     beam.wavey *= aperture_y
        # elif self.orientation == 1:
        #     beam.wavey *= (transmission * aperture_y)
        #     beam.wavex *= aperture_x
        # elif self.orientation == 2:
        #     beam.wavex *= (transmission * aperture_x)
        #     beam.wavey *= aperture_y
        # elif self.orientation == 3:
        #     beam.wavey *= (transmission * aperture_y)
        #     beam.wavex *= aperture_x

        # adjust beam direction
        beam.rotate_beam(delta_ax=p1_x, delta_ay=p1_y)
        # beam.ax += p1_x
        # beam.ay += p1_y

    def propagate(self, beam):
        """
        Method to propagate beam through prism. Calls multiply.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class WFS:
    """
    Class to represent Talbot wavefront sensor gratings/pinhole arrays.

    Attributes
    ----------

    """

    def __init__(self, name, pitch=None, duty_cycle=0.1, z=None, f0=100, phase=False, enabled=True, fraction=1):
        """
        Method to initialize a wavefront sensor.
        :param name: str
            name of the device (e.g. PF1K4)
        :param pitch: float
            Period of the grating
        :param duty_cycle: float
            Duty cycle of the grating. Defaults to 0.1 (10%)
        :param z: float
            z-location along the beamline
        :param f0: float
            Nominal distance to focus
        :param phase: bool
            If True, make a phase grating instead of an amplitude grating.
        :param enabled: bool
            If True, wavefront sensor influences the beam, otherwise it is effectively "moved out" of the beam.
        :param fraction: int
            Set to 1, 2, or 3 based on which Talbot fractional plane is being used.
        """

        # set attributes
        self.name = name
        self.pitch = pitch
        self.duty_cycle = duty_cycle
        self.f0 = f0
        self.z = z
        self.global_x = 0
        self.global_y = 0
        self.phase = phase
        self.enabled = enabled
        self.fraction = fraction
        self.azimuth = 0
        self.elevation = 0
        # initialize some calculated attributes
        self.x_pitch = 0.
        self.y_pitch = 0.
        self.grating_x = np.zeros(0)
        self.grating_y = np.zeros(0)
        self.xhat = None
        self.yhat = None
        self.zhat = None

    def propagate(self,beam):
        """
        Method to send the beam through
        :param beam: Beam
            Beam to propagate through the device
        :return: None
        """
        # Only do something if enabled.
        if self.enabled:
            self.multiply(beam)
        else:
            pass

    def disable(self):
        """
        Method to disable the wavefront sensor
        :return: None
        """
        # disable
        self.enabled = False

    def enable(self):
        """
        Method to enable the wavefront sensor
        :return: None
        """
        # enable
        self.enabled = True

    def multiply(self,beam):
        """
        Method to multiply a Beam by the wavefront sensor array
        :param beam: Beam
            Beam to propagate through the device
        :return: None
        """

        # get array sizes
        N = np.size(beam.y)
        M = np.size(beam.x)

        # Number of pixels per grating period
        self.x_pitch = np.round(self.pitch/beam.dx)
        self.y_pitch = np.round(self.pitch/beam.dy)

        print(self.pitch/beam.dx)
        print(self.pitch/beam.dy)

        # re-initialize 1D gratings
        self.grating_x = np.zeros(M)
        self.grating_y = np.zeros(N)

        # calculate number of periods in the grating
        Mg = np.floor(M / self.x_pitch)
        Ng = np.floor(N / self.y_pitch)

        # width of feature based on duty cycle
        x_width = int(self.x_pitch/2*self.duty_cycle)
        y_width = int(self.y_pitch/2*self.duty_cycle)

        # loop through periods in the horizontal grating
        for i in range(int(Mg)):
            # each step defines one period
            self.grating_x[int(self.x_pitch) * (i + 1) - x_width:int(self.x_pitch) * (i + 1) + x_width] = 1

        # loop through features in the vertical grating
        for i in range(int(Ng)):
            # each step defines one period
            self.grating_y[int(self.y_pitch) * (i+1) - y_width:int(self.y_pitch) * (i + 1) + y_width] = 1

        # convert to checkerboard pi phase grating if desired
        if self.phase:

            self.grating_x = np.exp(1j*np.pi*self.grating_x)
            self.grating_y = np.exp(1j*np.pi*self.grating_y)

        # multiply beam by grating
        beam.wavex *= self.grating_x
        beam.wavey *= self.grating_y
