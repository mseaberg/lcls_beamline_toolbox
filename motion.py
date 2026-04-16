import numpy as np
from ..utility import util
import scipy.spatial.transform as transform


class MotionAxis:
    """
    Base motion class defining common attributes and methods for motion axes.
    
    This is the parent class for both translation and rotation axes in a motion stack.
    It handles basic position tracking and limit management. Subclasses should implement
    specific motion behaviors (translation or rotation).
    
    Attributes
    ----------
    device_list : list
        List of device objects that are affected by motion on this axis.
    name : str, optional
        Human-readable name for the axis (e.g., 'X', 'Theta').
    position : float
        Current position of the axis in working units.
    low_limit : float, default -np.inf
        Minimum allowed position for this axis.
    high_limit : float, default np.inf
        Maximum allowed position for this axis.
    coupled_axes : list
        List of axes that move together with this axis (e.g., axes stacked above).
    """

    def __init__(self, device_list, name=None, initial_position=0, low_limit=-np.inf, high_limit=np.inf):
        """
        Initialize a motion axis.
        
        Parameters
        ----------
        device_list : list
            List of device objects controlled by this axis.
        name : str, optional
            Name identifier for the axis.
        initial_position : float, default 0
            Starting position of the axis.
        low_limit : float, default -np.inf
            Lower position limit.
        high_limit : float, default np.inf
            Upper position limit.
        """
        self.device_list = device_list
        self.name = name
        self.position = initial_position
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.coupled_axes = []

    def wm(self):
        """
        Get the current position ('where am I?').
        
        Returns
        -------
        float
            Current position in working units.
        """
        return self.position

    def get_low_limit(self):
        """
        Get the lower position limit.
        
        Returns
        -------
        float
            Lower limit value.
        """
        return self.low_limit

    def get_high_limit(self):
        """
        Get the upper position limit.
        
        Returns
        -------
        float
            Upper limit value.
        """
        return self.high_limit

    def set_low_limit(self, limit):
        """
        Set the lower position limit.
        
        Parameters
        ----------
        limit : float
            New lower limit value.
        """
        self.low_limit = limit

    def set_high_limit(self, limit):
        """
        Set the upper position limit.
        
        Parameters
        ----------
        limit : float
            New upper limit value.
        """
        self.high_limit = limit

    def set_current_position(self, new_position):
        """
        Update the current position and shift limits accordingly.
        
        This is useful for re-zeroing an axis. The position is updated and the limits
        are shifted by the same amount to maintain relative constraints.
        
        Parameters
        ----------
        new_position : float
            The new position value for the axis.
        """
        diff = new_position - self.position
        self.position = new_position
        self.set_low_limit(self.get_low_limit() + diff)
        self.set_high_limit(self.get_high_limit() + diff)


class TranslationAxis(MotionAxis):
    """
    Motion axis class for linear translations.
    
    This axis moves devices along a specified translation vector in 3D space.
    It inherits position tracking and limit management from MotionAxis.
    
    Attributes
    ----------
    translation_vector : np.ndarray
        3-element array defining the direction and magnitude of translation per unit position.
    """

    def __init__(self, translation_vector, device_list, **kwargs):
        """
        Initialize a translation axis.
        
        Parameters
        ----------
        translation_vector : array-like
            3-element array defining the direction of translation.
            Should typically be a unit vector or normalized vector.
        device_list : list
            List of device objects to be translated by this axis.
        **kwargs
            Additional keyword arguments passed to MotionAxis (name, initial_position, limits).
        """
        super().__init__(device_list, **kwargs)
        self.translation_vector = np.copy(translation_vector)

    def mv(self, position):
        """
        Move to an absolute position.
        
        Parameters
        ----------
        position : float
            Target absolute position in working units.
            
        Returns
        -------
        bool
            True if move was successful and within limits, False otherwise.
        """
        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Move relative to the current position.
        
        Checks position limits before executing the move. Updates all devices in the
        device_list and any coupled axes that are above this one in the stack.
        
        Parameters
        ----------
        adjustment : float
            Relative displacement in working units.
            
        Returns
        -------
        bool
            True if move was successful and within limits, False if move would
            violate position limits.
        """
        if self.high_limit >= self.position + adjustment >= self.low_limit:
            for device in self.device_list:
                current_position = device.get_pos()
                new_position = current_position + self.translation_vector * adjustment
                device.set_pos(new_position)

            self.position += adjustment

            # Update any rotation axes stacked above this one
            for axis in self.coupled_axes:
                if isinstance(axis, RotationAxis):
                    axis.translate_center(self.translation_vector * adjustment)
            return True
        else:
            return False

    def rotate_axis(self, rotation_vector, rotation_center):
        """
        Rotate this translation axis about a point.
        
        Used when this axis is positioned above a rotation stage in a motion stack.
        Rotates the translation_vector to account for the rotation of the stage below.
        
        Parameters
        ----------
        rotation_vector : np.ndarray
            3-element rotation vector (axis-angle representation).
        rotation_center : np.ndarray
            3-element point about which rotation occurs.
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()
        self.translation_vector = np.matmul(Re, self.translation_vector)


class RotationAxis(MotionAxis):
    """
    Motion axis class for rotational degrees of freedom.
    
    This axis rotates devices about a specified center point and rotation vector.
    Supports both radian and degree units. Also handles coupled axes in a motion stack.
    
    Attributes
    ----------
    rotation_vector : np.ndarray
        3-element array defining the rotation axis direction.
    rotation_center : np.ndarray
        3-element point about which rotation occurs.
    units : str
        Working units, either 'rad' (radians) or 'deg' (degrees).
    """

    def __init__(self, rotation_vector, device_list, rotation_center=None, units='rad', **kwargs):
        """
        Initialize a rotation axis.
        
        Parameters
        ----------
        rotation_vector : array-like
            3-element array defining the direction of the rotation axis.
            Should typically be a unit vector.
        device_list : list
            List of device objects to be rotated by this axis.
        rotation_center : array-like, optional
            3-element point about which rotation occurs. If None, defaults to the
            position of the first device in device_list.
        units : str, default 'rad'
            Working units for positions: 'rad' for radians or 'deg' for degrees.
            Other values default to 'rad'.
        **kwargs
            Additional keyword arguments passed to MotionAxis (name, initial_position, limits).
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
        Rotate devices about the rotation center point.
        
        Applies rotation to both device positions and orientation vectors
        (normal/sagittal/tangential or xhat/yhat/zhat depending on device type).
        
        Parameters
        ----------
        adjustment : float
            Rotation angle in radians.
        """
        re = transform.Rotation.from_rotvec(self.rotation_vector * adjustment)
        Re = re.as_matrix()

        for device in self.device_list:
            device_pos = device.get_pos()
            new_pos = np.matmul(Re, device_pos - self.rotation_center) + self.rotation_center

            # Update device orientation vectors based on device type
            if hasattr(device, 'normal'):
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
        Move to an absolute angular position.
        
        Parameters
        ----------
        position : float
            Target absolute angular position in working units.
            
        Returns
        -------
            bool
            True if move was successful and within limits, False otherwise.
        """
        adjustment = position - self.position
        return self.mvr(adjustment)

    def mvr(self, adjustment):
        """
        Move relative to the current angular position.
        
        Checks position limits before executing the move. Converts from working units
        (degrees or radians) to radians for internal calculations. Updates all devices
        and any coupled axes stacked above.
        
        Parameters
        ----------
        adjustment : float
            Relative angular displacement in working units.
            
        Returns
        -------
            bool
            True if move was successful and within limits, False if move would
            violate position limits.
        """
        if self.high_limit >= self.position + adjustment >= self.low_limit:
            # Convert to radians if working in degrees
            if self.units == 'deg':
                motion_adjustment = np.deg2rad(adjustment)
            else:
                motion_adjustment = np.copy(adjustment)
                
            self.rotate_about_point(motion_adjustment)
            
            # Update coupled axes (axes above this one in the stack)
            for axis in self.coupled_axes:
                axis.rotate_axis(self.rotation_vector * motion_adjustment, self.rotation_center)
                
            # Update motor position using working units
            self.position += adjustment
            return True
        else:
            return False

    def rotate_axis(self, rotation_vector, rotation_center):
        """
        Rotate this rotation axis and its center about a point.
        
        Used when this axis is positioned above another rotation stage in a motion stack.
        Rotates both the rotation_vector and rotation_center to account for the rotation
        of the stage below.
        
        Parameters
        ----------
        rotation_vector : np.ndarray
            3-element rotation vector (axis-angle representation).
        rotation_center : np.ndarray
            3-element point about which rotation occurs.
        """
        re = transform.Rotation.from_rotvec(rotation_vector)
        Re = re.as_matrix()

        self.rotation_center = np.matmul(Re, self.rotation_center - rotation_center) + rotation_center
        self.rotation_vector = np.matmul(Re, self.rotation_vector)

    def translate_center(self, translation_vector):
        """
        Translate the rotation center by a given vector.
        
        Called when a translation axis below this rotation axis moves, to keep
        the rotation center in the correct location.
        
        Parameters
        ----------
        translation_vector : np.ndarray
            3-element translation vector.
        """
        self.rotation_center += translation_vector


class MotionStack:
    """
    Collection of motion axes with defined relationships and stacking order.
    
    A motion stack represents a series of axes (translations and rotations) stacked
    vertically, where the order determines how coupled motion works. Axes lower in
    the stack (earlier in axis_list) affect axes higher in the stack, and motion
    on lower axes can trigger updates to higher axes.
    
    Attributes
    ----------
    axis_list : list of MotionAxis
        List of axes in the stack, ordered from bottom (index 0) to top (highest index).
    """

    def __init__(self, axis_list):
        """
        Initialize a motion stack.
        
        Parameters
        ----------
        axis_list : list of MotionAxis
            List of motion axes in order from bottom of stack (index 0, affects all others)
            to top of stack (highest index, not affected by others). Both TranslationAxis
            and RotationAxis objects can be included.
            
        Notes
        -----
        Axes are automatically linked as coupled axes based on their position in the stack.
        Each axis's coupled_axes attribute is set to all axes above it in the stack.
        """
        self.axis_list = axis_list
        for num, axis in enumerate(self.axis_list):
            # Axes above this one in the stack are coupled to it
            axis.coupled_axes = self.axis_list[num + 1:]