import numpy as np
from ..utility import util
import scipy.spatial.transform as transform


class MotionAxis:
    """
    Base motion class, mostly defining attributes
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
        diff = new_position - self.position
        self.position = new_position
        self.set_low_limit(self.get_low_limit()+diff)
        self.set_high_limit(self.get_high_limit()+diff)


class TranslationAxis(MotionAxis):
    """
    Motion class for translations
    """

    def __init__(self, translation_vector, device_list, **kwargs):
        super().__init__(device_list, **kwargs)
        # vector along which this axis translates
        self.translation_vector = np.copy(translation_vector)

    def mv(self, position):
        """
        method to move to an absolute position, calls relative motion method
        """
        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Method for relative motion along a translation axis.
        Returns True if the new position is within limits, False if the new position is outside the limits.
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
        Method to rotate the translation axis (i.e. if it is above a rotation stage in a stack)
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()

        self.translation_vector = np.matmul(Re, self.translation_vector)


class RotationAxis(MotionAxis):
    """
    Class for rotational degrees of freedom.
    """

    def __init__(self, rotation_vector, device_list, rotation_center=None, **kwargs):
        super().__init__(device_list, **kwargs)
        self.rotation_vector = np.copy(rotation_vector)
        if rotation_center is None:
            # if rotation center is not specified, rotate about device center
            self.rotation_center = np.copy(device_list[0].get_pos())
        else:
            self.rotation_center = np.copy(rotation_center)

    def rotate_about_point(self, adjustment):
        """
        Method to rotate a device about an arbitrary rotation center
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
        method to move to an absolute position, calls relative motion method
        """
        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Method for relative motion about a rotation axis.
        Returns True if the new position is within limits, False if the new position is outside the limits.
        """
        if self.high_limit >= self.position + adjustment >= self.low_limit:

            self.rotate_about_point(adjustment)
            for axis in self.coupled_axes:
                axis.rotate_axis(self.rotation_vector*adjustment, self.rotation_center)
            self.position += adjustment
            return True
        else:
            return False

    def rotate_axis(self, rotation_vector, rotation_center):
        """
        Method to rotate the rotation axis and center (i.e. if it is above a rotation stage in a stack)
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()

        self.rotation_center = np.matmul(Re, self.rotation_center - rotation_center) + rotation_center

        self.rotation_vector = np.matmul(Re, self.rotation_vector)

    def translate_center(self, translation_vector):
        self.rotation_center += translation_vector


class MotionStack:
    """
    Class for a collection of motion axes, with relationships between them such as the order of the stack and
    coupled motion.
    """

    def __init__(self, axis_list):
        """
        Argument is a list of axes, with the bottom of the stack at the beginning (lowest index) of the list
        and top of the stack at the end (highest index) of the list.
        """
        self.axis_list = axis_list
        for num, axis in enumerate(self.axis_list):
            axis.coupled_axes = self.axis_list[num+1:]
