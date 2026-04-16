import numpy as np
from ..utility import util
import scipy.spatial.transform as transform


"""Motion primitives for translational/rotational device stacks.

The classes in this module provide lightweight software axes that update device
geometry in-place. Each axis tracks a scalar position in axis-native units and
optionally propagates its motion to axes mounted above it in a `MotionStack`.
"""


class MotionAxis:
    """
    Base class for a single scalar motion axis.

    Parameters
    ----------
    device_list : list
        Devices moved by this axis. Devices are expected to provide geometry
        accessors/mutators used by concrete axis implementations.
    name : str, optional
        User-facing axis name.
    initial_position : float, optional
        Initial scalar readback for this axis.
    low_limit : float, optional
        Lower software limit in axis-native units.
    high_limit : float, optional
        Upper software limit in axis-native units.

    Notes
    -----
    The base class stores axis metadata and soft limits only. Motion behavior is
    implemented by subclasses.
    """

    def __init__(self, device_list, name=None, initial_position=0, low_limit=-np.inf, high_limit=np.inf):
        # device that this axis belongs to
        self.device_list = device_list
        self.name = name
        # initialize position
        self.position = initial_position
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.coupled_axes = []

    def wm(self):
        """Return the current axis position (readback)."""
        return self.position

    def get_low_limit(self):
        return self.low_limit

    def get_high_limit(self):
        return self.high_limit

    def set_low_limit(self, limit):
        self.low_limit = limit

    def set_high_limit(self, limit):
        self.high_limit = limit

    def set_current_position(self, new_position):
        """Set the current readback and shift limits by the same offset.

        This is a bookkeeping operation equivalent to redefining the axis zero.
        It does not move underlying devices.
        """
        diff = new_position - self.position
        self.position = new_position
        self.set_low_limit(self.get_low_limit()+diff)
        self.set_high_limit(self.get_high_limit()+diff)


class TranslationAxis(MotionAxis):
    """
    Motion axis that translates devices along a fixed direction vector.

    Parameters
    ----------
    translation_vector : array_like, shape (3,)
        Unit vector defining direction used to convert scalar axis motion to device
        displacement in world coordinates.
    device_list : list
        Devices translated by this axis.
    **kwargs
        Forwarded to `MotionAxis`.
    """

    def __init__(self, translation_vector, device_list, **kwargs):
        super().__init__(device_list, **kwargs)
        # vector along which this axis translates
        self.translation_vector = np.copy(translation_vector)

    def mv(self, position):
        """
        Move to an absolute axis position.

        Parameters
        ----------
        position : float
            Target absolute position in axis-native units.

        Returns
        -------
        bool
            `True` when the requested move is within limits and applied,
            otherwise `False`.
        """
        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Move relatively along the translation axis.

        Parameters
        ----------
        adjustment : float
            Relative move increment in axis-native units.

        Returns
        -------
        bool
            `True` if the final position is within limits and applied,
            `False` if the move is rejected by soft limits.

        Notes
        -----
        For accepted motion, each device position is updated in-place and
        coupled rotation axes have their rotation centers translated by the
        same world-space displacement.
        """
        if self.high_limit >= self.position + adjustment >= self.low_limit:
            for device in self.device_list:
                current_position = device.get_pos()
                new_position = current_position + self.translation_vector * adjustment
                device.set_pos(new_position)

            self.position += adjustment

            for axis in self.coupled_axes:
                if isinstance(axis, RotationAxis):
                    axis.translate_center(self.translation_vector * adjustment)
            return True
        else:
            return False

    def rotate_axis(self, rotation_vector, rotation_center):
        """
        Rotate this axis direction when a lower-stage rotation is applied.

        Parameters
        ----------
        rotation_vector : array_like, shape (3,)
            Rotation vector (axis * angle, radians) describing lower-stage
            motion.
        rotation_center : array_like, shape (3,)
            Unused for translation axes; included for API compatibility with
            stack coupling.
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()

        self.translation_vector = np.matmul(Re, self.translation_vector)


class RotationAxis(MotionAxis):
    """
    Motion axis representing rotation about a world-space center.

    """

    def __init__(self, rotation_vector, device_list, rotation_center=None, units='rad', **kwargs):
        """
        Parameters
        ----------
        rotation_vector : array_like, shape (3,)
            Rotation axis in world coordinates. Must be a unit vector, which is
            multiplied by the commanded angle in `mvr`.
        device_list : list
            Devices rotated by this axis.
        rotation_center : array_like, shape (3,), optional
            Point about which rotation is applied. If omitted, uses
            `device_list[0].get_pos()` at initialization.
        units : {'rad', 'deg'}, optional
            Command/readback units for `position`, limits, and move requests.
            Internal geometry updates are always performed in radians.
        kwargs
            Forwarded to `MotionAxis`.
        """
        super().__init__(device_list, **kwargs)
        self.rotation_vector = np.copy(rotation_vector)
        self.units = units
        if rotation_center is None:
            # if rotation center is not specified, rotate about device center
            self.rotation_center = np.copy(device_list[0].get_pos())
        else:
            self.rotation_center = np.copy(rotation_center)

    def rotate_about_point(self, adjustment):
        """
        Apply a relative rotation to all attached devices.

        Parameters
        ----------
        adjustment : float
            Rotation angle in radians.

        Notes
        -----
        Device position and orientation bases are updated in-place. Devices with
        a `normal` attribute are treated as optic-like objects (`normal`,
        `sagittal`, `tangential`); otherwise `xhat`, `yhat`, `zhat` are rotated.
        """
        re = transform.Rotation.from_rotvec(self.rotation_vector*adjustment)
        Re = re.as_matrix()

        for device in self.device_list:
            device_pos = device.get_pos()
            new_pos = np.matmul(Re, device_pos - self.rotation_center) + self.rotation_center

            if hasattr(device,'normal'):
                device.normal = np.matmul(Re, device.normal)
                device.sagittal = np.matmul(Re, device.sagittal)
                device.tangential = np.matmul(Re, device.tangential)
            else:
                device.xhat = np.matmul(Re, device.xhat)
                device.yhat = np.matmul(Re, device.yhat)
                device.zhat = np.matmul(Re, device.zhat)

            device.set_pos(new_pos)

    def mv(self, position):
        """
        Move to an absolute axis position.

        Parameters
        ----------
        position : float
            Target absolute angle in axis units.

        Returns
        -------
        bool
            `True` when the requested move is within limits and applied,
            otherwise `False`.
        """

        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Move relatively about the rotation axis.

        Parameters
        ----------
        adjustment : float
            Relative angle in axis units (`rad` or `deg`).

        Returns
        -------
        bool
            `True` if the final position is within limits and applied,
            `False` if the move is rejected by soft limits.

        Notes
        -----
        Accepted moves propagate to coupled axes above this one: translation
        axes have their directions rotated; rotation axes have both their
        centers and axis vectors rotated consistently about this axis center.
        """
        if self.high_limit >= self.position + adjustment >= self.low_limit:

            # if we are working in degrees, convert to radians before actually moving things
            if self.units == 'deg':
                motion_adjustment = np.deg2rad(adjustment)
            else:
                motion_adjustment = np.copy(adjustment)
            self.rotate_about_point(motion_adjustment)
            for axis in self.coupled_axes:
                axis.rotate_axis(self.rotation_vector*motion_adjustment, self.rotation_center)
            # update motor position using working units
            self.position += adjustment
            return True
        else:
            return False

    def rotate_axis(self, rotation_vector, rotation_center):
        """
        Update this axis definition due to a lower-stage rotation.

        Parameters
        ----------
        rotation_vector : array_like, shape (3,)
            Rotation vector (axis * angle, radians) applied by a lower stage.
        rotation_center : array_like, shape (3,)
            Center of the lower-stage rotation.

        Notes
        -----
        Both the axis direction and rotation center are transformed so future
        moves remain mechanically consistent within the stack.
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()

        self.rotation_center = np.matmul(Re, self.rotation_center - rotation_center) + rotation_center

        self.rotation_vector = np.matmul(Re, self.rotation_vector)

    def translate_center(self, translation_vector):
        """Translate the stored rotation center by a world-space vector."""
        self.rotation_center += translation_vector


class MotionStack:
    """
    Ordered collection of axes that defines stage coupling.

    Axes are ordered bottom-to-top. Motion of a lower index axis is propagated
    to all higher index axes through each axis's `coupled_axes` list.
    """

    def __init__(self, axis_list):
        """
        Parameters
        ----------
        axis_list : list[MotionAxis]
            Stack-ordered axes, where index 0 is the lowest mechanical stage and
            the last index is the highest stage.
        """
        self.axis_list = axis_list
        for num, axis in enumerate(self.axis_list):
            axis.coupled_axes = self.axis_list[num+1:]
