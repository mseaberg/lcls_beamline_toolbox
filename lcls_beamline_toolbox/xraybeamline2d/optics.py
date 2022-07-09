"""
optics module

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
import json
from time import sleep
from .pitch import TalbotLineout, TalbotImage
import scipy.interpolate as interpolation
import scipy.ndimage as ndimage
import scipy.optimize as optimize
import scipy.spatial.transform as transform
from skimage.restoration import unwrap_phase
import os
import pickle
from ..polyprojection.legendre import LegendreFit2D
from lcls_beamline_toolbox.xrayinteraction import interaction
from .util import Util, LegendreUtil
from scipy.interpolate import interp2d, RectBivariateSpline
import xraydb
try:
    from epics import PV
    from pcdsdevices.areadetector.detectors import PCDSAreaDetector
    from ophyd import EpicsSignal
    from ophyd import EpicsSignalRO
    from ophyd import Component as Cpt
except ImportError:
    print("Can't find epics package. PPM_Imager class will not be supported")
try:
    import xrt.backends.raycing.materials as materials
except ImportError:
    print("Can't find xrt package. Crystal class will not be supported")


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
    z: float
        Longitudinal position along beamline (meters)
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
        self.correction = 0
        self.beam_cx = 0
        self.beam_cy = 0
        self.beam_ax = 0
        self.beam_ay = 0
        self.show_figures = False
        self.x_intersect = 0.0
        self.y_intersect = 0.0
        self.z_intersect = 0.0
        self.use_reflectivity = False
        self.material = 'B4C'

        # set allowed kwargs
        allowed_arguments = ['length', 'width', 'alpha', 'z', 'orientation', 'shapeError',
                             'delta', 'dx', 'dy', 'motor_list', 'roll', 'yaw', 'show_figures', 'use_reflectivity',
                             'material']
        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        self.beta0 = self.alpha

        # set some calculated attributes
        self.projectWidth = np.abs(self.length * (self.alpha + self.delta))

        # get some material properties
        mirror_material = interaction.Mirror(name=name, range='HXR', material=self.material)
        self.density = mirror_material.density

    def find_intersection(self, beam):

        ux = np.reshape(np.array([1, 0, 0]), (3, 1))
        uy = np.reshape(np.array([0, 1, 0]), (3, 1))
        uz = np.reshape(np.array([0, 0, 1]), (3, 1))

        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx

        # no need to rotate into a different plane than the mirror surface, so mirror unit vectors
        # are unchanged
        mirror_x = self.normal
        mirror_y = self.sagittal
        mirror_z = self.transverse

        central_ray = np.reshape(beam.zhat, (3,1))

        coords = np.reshape(beam_center, (3,1))

        coords -= np.reshape(mirror_center, (3, 1))

        # now write beam coordinates in ellipse coordinates
        transform_matrix = np.tensordot(np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_mirror = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # now write rays in mirror coordinates
        rays_mirror = np.tensordot(transform_matrix, central_ray, axes=(1, 0))

        z_intersect = coords_mirror[2, :] - rays_mirror[2, :] / rays_mirror[0, :] * coords_mirror[0, :]
        # by definition the x coordinate is zero for intersection
        x_intersect = np.reshape(np.array(0.0),(1,))
        y_intersect = rays_mirror[1, :] / rays_mirror[2, :] * (z_intersect - coords_mirror[2, :]) + coords_mirror[1,:]

        # put into an array
        intersect_point = np.reshape(np.array([x_intersect,y_intersect,z_intersect]), (3,1))

        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        intersect_global = np.tensordot(inv_transform, intersect_point, axes=(1, 0))

        intersect_global += np.reshape(mirror_center, (3, 1))

        return intersect_global

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

    def trace_surface(self, beam):

        figon = self.show_figures
        # global unit vectors
        ux = np.reshape(np.array([1,0,0]),(3,1))
        uy = np.reshape(np.array([0,1,0]),(3,1))
        uz = np.reshape(np.array([0,0,1]),(3,1))

        delta_z = self.length / 2 * 1.1

        print('ax: %.6e' % beam.ax)
        print('ay: %.6e' % beam.ay)

        # propagate beam to just upstream of mirror
        beam.beam_prop(-delta_z)

        # vector defining displacement from beam location to mirror center. This is in global coordinates
        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx
        beam_to_mirror = beam_center - mirror_center

        # unit vectors for ellipse coordinates, written in global coordinates
        mirror_x = self.normal
        mirror_y = self.sagittal
        mirror_z = self.transverse

        # define beam rays up to second order (assume that linear term is not needed and is
        # already captured in the k-vector
        rays_x = beam.x/beam.zx
        rays_y = beam.y/beam.zy

        # beam coordinates in global coordinates
        coords = np.multiply.outer(beam.xhat, beam.x) + np.multiply.outer(beam.yhat, beam.y)

        # reference to global origin by adding beam global center
        coords += np.reshape(beam_center, (3, 1, 1))

        # now subtract mirror center so that beam coordinates are in global coordinates,
        # but with origin at mirror center
        coords -= np.reshape(mirror_center, (3, 1, 1))

        # now write beam coordinates in mirror plane coordinates
        # (transforming from global coordinates to mirror coordinates)
        transform_matrix = np.tensordot(np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_mirror = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # mirror vectors written in mirror coordinates
        mirror_x_local = ux
        mirror_y_local = uy
        mirror_z_local = uz

        # calculate z component of rays (enforcing unit vector)
        rays_z = np.sqrt(np.ones_like(rays_x) - rays_x ** 2 - rays_y ** 2)
        # ray vectors at each point in the beam
        rays = (np.multiply.outer(beam.xhat, rays_x) + np.multiply.outer(beam.zhat, rays_z)
                + np.multiply.outer(beam.yhat, rays_y))

        # normalize rays (should be redundant)
        rays = rays / np.sqrt(np.sum(rays*rays, axis=0))

        # now write rays in mirror coordinates
        rays_mirror = np.tensordot(transform_matrix, rays, axes=(1,0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2,:,0],coords_mirror[0,:,0])
            plt.quiver(coords_mirror[2,:,0],coords_mirror[0,:,0],rays_mirror[2,:,0],rays_mirror[0,:,0])
            plt.ylim(-.5,.5)
            plt.grid()
            plt.title('incoming rays and mirror')

        z_intersect = coords_mirror[2, :, :] - rays_mirror[2, :, :] / rays_mirror[0, :, :] * coords_mirror[0, :, :]
        # by definition the x coordinate is zero for intersection
        Ni, Mi = np.shape(z_intersect)
        x_intersect = np.zeros_like(z_intersect)
        y_intersect = (rays_mirror[1, :, :] / rays_mirror[2, :, :]
                       * (z_intersect - coords_mirror[2, :, :]) + coords_mirror[1, :, :])

        # put mirror intersection coordinates in a 3xNxM array
        intersect_coords = np.zeros((3,Ni,Mi))
        intersect_coords[0,:,:] = x_intersect
        intersect_coords[1,:,:] = y_intersect
        intersect_coords[2,:,:] = z_intersect

        # vectors pointing from beam location to mirror intersection
        i_vector = intersect_coords - coords_mirror

        # length of each vector
        distance_1 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        # define normals along mirror surface
        mirror_normal = np.zeros_like(rays)
        mirror_normal[0,:,:] = np.ones_like(z_intersect)

        # calculate ray direction after interaction with ellipse using law of reflection
        rays_out = rays_mirror - 2 * np.sum(rays_mirror*mirror_normal,axis=0) * mirror_normal

        # now find intersection with exit plane
        # we can define this simply as having a normal vector in the direction of the central ray
        # and we will define the plane to be a distance length/2*1.1 from the intersection point of the central ray
        plane_normal = np.reshape(rays_out[:,int(beam.N/2), int(beam.M/2)],(3,1,1))
        central_point = np.reshape(intersect_coords[:,int(beam.N/2),int(beam.M/2)],(3,1,1)) + plane_normal*self.length/2*1.1

        # find z intersection with this plane
        d2 = np.sum((central_point - intersect_coords)*plane_normal,axis=0)/np.sum(rays_out*plane_normal,axis=0)
        plane_intersect = intersect_coords + rays_out*d2
        i_vector = plane_intersect - intersect_coords
        distance_2 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2, :, 0], coords_mirror[0, :, 0])
            plt.plot(z_intersect, x_intersect)
            plt.plot(plane_intersect[2,:, 0],plane_intersect[0,:, 0])
            # plt.ylim(-.5, .5)
            plt.grid()
            plt.title('entrance/exit planes, mirror intersection')

        # total distance for each beam ray
        total_distance = (distance_1+distance_2)
        #
        if figon:
            plt.figure()
            plt.plot(intersect_coords[2,:,0],distance_1)
            plt.plot(intersect_coords[2,:,0],distance_2)
            plt.plot(intersect_coords[2,:,0],distance_1+distance_2)
            plt.title('distances')

        # find location of central ray in exit plane
        origin = np.reshape(plane_intersect[:,int(beam.N/2),int(beam.M/2)],(3,1,1))

        # put beam center at origin
        shifted_plane = plane_intersect-origin

        # get final k-vector for central ray
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        # convert to global coordinates
        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))
        k_f_global = k_f_global[:,0,0]

        # first rotate by the "nominal" amount
        if self.orientation==0:
            beam.rotate_nominal(delta_azimuth=2*self.alpha)
        elif self.orientation==1:
            beam.rotate_nominal(delta_elevation=2*self.alpha)
        elif self.orientation==2:
            beam.rotate_nominal(delta_azimuth=-2*self.alpha)
        elif self.orientation==3:
            beam.rotate_nominal(delta_elevation=-2*self.alpha)

        # get initial k-vector for central ray in global coordinates
        k_i = np.copy(beam.zhat)

        # find the change in the k-vector in global coordinates
        delta_k = k_f_global - k_i

        print('xhat: {}'.format(beam.xhat))
        print('yhat: {}'.format(beam.yhat))
        print('zhat: {}'.format(beam.zhat))
        print('dk: {}'.format(delta_k))

        # now make minor adjustment to k-vector based on central ray at exit plane
        # might want to do one axis at a time or change the order. Or could change the rotation
        # to rotate about the "unrotated" axes.
        delta_ax = np.arcsin(delta_k[0])
        x_sign = np.sign(np.dot(np.cross(k_i,k_f_global),beam.yhat))
        delta_ay = -np.arcsin(delta_k[1])
        y_sign = np.sign(-np.dot(np.cross(k_i,k_f_global),beam.xhat))
        beam.rotate_beam(delta_ax=x_sign*np.abs(delta_ax), delta_ay=y_sign*np.abs(delta_ay))

        # check for consistency
        print('is beam in the correct direction?')
        # print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global)))

        # mask defining mirror acceptance
        # if self.q>=0:
        #     mask = np.logical_and(coords_ellipse[0,:,:]>intersect_coords[0,:,:],
        #                           plane_intersect[0,:,:]>intersect_coords[0,:,:])
        # else:
        #     mask = np.logical_and(coords_ellipse[0, :,:] < intersect_coords[0, :,:],
        #                           plane_intersect[0, :,:] > intersect_coords[0, :,:])
        #
        # # second order fit to ray distance
        # if self.q>=0:
        #     mask = np.logical_and(mask, coords_ellipse[0,:,:]<0)
        # else:
        #     mask = np.logical_and(mask, coords_ellipse[0, :,:] > 0)

        # can define mirror acceptance by taking dot product with mirror unit vectors (in ellipse coordinate system).
        # distance along length axis
        d_length = np.sum((intersect_coords)*np.reshape(mirror_z_local,(3,1,1)),axis=0)
        # distance along width axis
        d_width = np.sum((intersect_coords)*np.reshape(mirror_y_local,(3,1,1)),axis=0)

        mask = np.ones_like(d_length)

        # d_length and d_width are equivalent to the mirror coordinates, these can be used for interpolating
        # the shape error
        # mirror shape error interpolation onto beam coordinates (if applicable)

        shapeInterp = np.zeros((beam.N,beam.M))

        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                max_ys = self.width / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                ys = np.linspace(-1,1,100)*max_ys

                print(np.shape(np.tile(self.shapeError,(100,1))))
                f = RectBivariateSpline(ys,zs,np.tile(self.shapeError,(100,1)))
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N, beam.M))*1e-9

                plt.figure()
                plt.imshow(shapeInterp)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = RectBivariateSpline(ys,zs,self.shapeError)
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N,beam.M))*1e-9
                # f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                # shapeError2 = f(zi_1d - self.dx / np.tan(self.total_alpha), yi_1d - self.dy)

        # mask based on mirror length
        mask = np.logical_and(mask, np.abs(d_length)<self.length/2)
        # mask based on mirror width
        mask = np.logical_and(mask, np.abs(d_width)<self.width/2)

        # now write new beam coordinates in local beam coordinate system
        # (transforming from ellipse coordinates to local beam coordinates)
        transform_matrix2 = np.tensordot(np.reshape([beam.xhat,beam.yhat,beam.zhat], (3, 3)),
                                        np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)), axes=(1, 1))
        shifted_plane2 = np.tensordot(transform_matrix2, shifted_plane, axes=(1, 0))

        # beam's x and y coordinates in the exit beam local coordinate system
        # note these are not in general on a regularly spaced grid but are based on tracking
        # rays through the course of the reflection.
        x_eff = shifted_plane2[0,:,:]
        y_eff = shifted_plane2[1,:,:]

        # calculate desired pixel size due to expected change in beam size, and
        # define new coordinate grids to interpolate onto at beam exit plane.
        if self.orientation == 0 or self.orientation == 2:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)
        else:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)

        # 1D masks based on mirror acceptance, going through beam center.
        max_intensity = np.max(np.abs(beam.wave)**2)
        # mask2 = np.logical_and(mask,np.abs(beam.wave)**2>0.1*max_intensity)
        mask_x = mask[int(beam.N/2),:]>0
        mask_y = mask[:,int(beam.M/2)]>0
        # introducing weights for the polynomial fits based on beam intensity

        weight_x = (np.abs(beam.wave[int(beam.N/2),:])**2>0.1*max_intensity).astype(float)
        weight_y = (np.abs(beam.wave[:,int(beam.M/2)])**2>0.1*max_intensity).astype(float)

        # plt.figure()
        # plt.imshow((total_distance-np.mean(total_distance[mask]))*mask)
        #
        # plt.figure()
        # plt.imshow(np.abs(beam.wave)**2>0.1*max_intensity)

        incidence_angle = np.abs(np.arccos(np.sum(rays_out * mirror_normal, axis=0)))
        glancing = np.pi / 2 - incidence_angle

        reflectivity = xraydb.mirror_reflectivity(self.material, glancing, beam.photonEnergy, self.density)

        # plt.figure()
        # plt.imshow((glancing-np.mean(glancing[mask]))*mask)

        # subtract best fit parabola in x-direction and best fit line in y direction
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff)# + np.polyval(p_coeff_y[:-2],y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff)# + np.polyval(p_coeff_y[:-2], y_eff)
        #
        # TRY THIS OUT LATER
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff) + np.polyval(p_coeff_y,y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff) + np.polyval(p_coeff_x, x_eff)

        points = np.zeros((np.size(x_eff[mask]),2))
        points[:,0] = x_eff[mask]
        points[:,1] = y_eff[mask]
        # xi = np.zeros((np.size(x_out),2))
        # xi[:,0] = x_out
        # xi[:,1] = y_out
        xi_0, xi_1 = np.meshgrid(x_out, y_out)
        print('attempting interpolation')
        mask2 = interpolation.griddata(points, mask[mask], (xi_0, xi_1), method='nearest',fill_value=0)
        # mask2 = fmask(x_out,y_out)
        mask2[mask2<.9] = 0
        # mask2 = mask2.astype(int)
        mask2 = mask2 > 0.5

        # interpolate intensity onto new exit plane grid
        abs_out = interpolation.griddata(points, np.abs(beam.wave[mask]), (xi_0, xi_1), fill_value=0)

        # unwrap phase of beam at input
        angle_in = unwrap_phase(np.angle(beam.wave))

        # add quadratic phase if beam is not focused since this also needs to be interpolated
        quadratic = np.zeros_like(beam.x)

        if not beam.focused_x:
            quadratic += np.pi / beam.lambda0 / beam.zx * (beam.x) ** 2
        if not beam.focused_y:
            quadratic += np.pi / beam.lambda0 / beam.zy * (beam.y) ** 2

        # add the quadratic phase to the input phase
        angle_in += quadratic

        # add phase contribution from deviations in the distance traveled by each ray
        total_phase = (angle_in + 2 * np.pi / beam.lambda0 * total_distance
                       - shapeInterp * 4*np.pi*np.sin(self.alpha) / beam.lambda0)

        # get polynomial fits based on new coordinates
        p_coeff_x = np.polyfit(x_eff[int(beam.N/2),:][mask_x], total_phase[int(beam.N/2),:][mask_x], 2,
                               w=weight_x[mask_x])

        p_coeff_y = np.polyfit(y_eff[:,int(beam.M/2)][mask_y], total_phase[:,int(beam.M/2)][mask_y], 2,
                               w=weight_y[mask_y])

        # calculate effective distance to focus based on total phase
        z_2 = np.pi / beam.lambda0 / p_coeff_x[-3]
        z_2_y = np.pi / beam.lambda0 / p_coeff_y[-3]

        # add this to what was already subtracted (might be able to do this in one step)
        # z_total_x = 1 / (1 / z_out_x + 1 / z_2)
        # z_total_y = 1 / (1 / z_out_y + 1 / z_2_y)
        z_total_x = z_2
        z_total_y = z_2_y
        print('new z: %.6f' % z_total_x)
        print('new z y: %.6f' % z_total_y)

        # calculate residual linear phase terms
        # linear += p_coeff_x[-2] * beam.lambda0/2/np.pi
        # linear_y += p_coeff_y[-2] * beam.lambda0/2/np.pi
        linear = p_coeff_x[-2] * beam.lambda0 / 2 / np.pi
        linear_y = p_coeff_y[-2] * beam.lambda0/2/np.pi

        print(linear)
        print(linear_y)

        # subtract linear terms since this should be taken care of based on central ray direction
        # Now this is being done in both tangential and sagittal directions, and we compensate with
        # a change in beam k-vector direction.
        total_phase -= np.polyval(p_coeff_x[-2:], x_eff) + np.polyval(p_coeff_y[-2:], y_eff)

        # sutbtract off quadratic phase term if the beam is not focused
        if not beam.focused_x:
            total_phase -= np.polyval([p_coeff_x[-3], 0, 0], x_eff)
        if not beam.focused_y:
            total_phase -= np.polyval([p_coeff_y[-3], 0, 0], y_eff)

        # interpolate the phase onto the exit plane grid
        points = np.zeros((np.size(x_eff), 2))
        points[:, 0] = x_eff.flatten()
        points[:, 1] = y_eff.flatten()
        phase_interp = interpolation.griddata(points, total_phase.flatten(), (xi_0, xi_1), fill_value=0)

        # interpolate the reflectivity onto the exit plane grid
        reflectivity_interp = interpolation.griddata(points, reflectivity.flatten(), (xi_0, xi_1), fill_value=0)

        # update beam complex amplitude using what has been interpolated onto the new grid
        beam.wave = abs_out * np.exp(1j * phase_interp)
        # multiply by reflectivity if this is being used
        if self.use_reflectivity:
            beam.wave *= np.sqrt(reflectivity_interp)
        # multiply by mirror aperture that has been interpolated onto the exit plane grid
        beam.wave *= mask2

        if figon:
            plt.figure()
            plt.imshow(np.abs(beam.wave))

        ax0 = np.copy(beam.ax)
        ay0 = np.copy(beam.ay)

        # figure out where the beam is in global coordinates
        # change in angle
        k_i = rays_mirror[:, int(beam.N / 2), int(beam.M / 2)]
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))

        # compensate for removing linear phase by adjusting beam k-vector
        beam.rotate_beam(delta_ax=linear)
        beam.rotate_beam(delta_ay=linear_y)

        print(beam.ax)
        delta_cx = ax0 * self.length / 2 * 1.1
        delta_cx += beam.ax * self.length / 2 * 1.1
        delta_cx += 2 * np.dot(self.normal, beam.xhat) * self.dx
        print('change in beam center')
        print(delta_cx)
        # beam.cx = -beam.cx + delta_cx
        # print(beam.cx)

        x_out, y_out = np.meshgrid(x_out, y_out)
        beam.x = x_out
        beam.y = y_out

        beam.new_fx()

        print('is beam in the correct direction?')
        print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global[:, 0, 0])))
        print(k_f)
        print(k_f_global)

        # now figure out global coordinates
        # get back into global coordinates using inverse of transformation matrix, just looking at central ray
        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        origin_global = np.tensordot(inv_transform, origin, axes=(1,0))

        # now add the mirror center in global coordinates, so that this should be the beam location
        # in global coordinates
        origin_global += np.reshape(mirror_center, (3, 1, 1))
        # origin_global -= np.reshape(self.normal*dx,(3,1))
        print(origin_global)
        # now shift origin to ellipse origin

        beam.global_x = origin_global[0,0,0]
        beam.global_y = origin_global[1,0,0]
        beam.global_z = origin_global[2,0,0]

        # account for coordinate scaling and/or changes to Rayleigh range,
        # whether beam is classified as focused or not.
        beam.change_z(new_zx=z_total_x,new_zy=z_total_y)

        beam.new_fx()
        print('global_x: %.2f' % beam.global_x)
        print('global_y: %.2f' % beam.global_y)
        print('global_z: %.2f' % beam.global_z)

        print(np.shape(beam.x))
        print(np.shape(beam.y))

    def reflect(self, beam):
        """
        Method to reflect a beam from a flat mirror.
        :param beam: Beam
            Beam object to be reflected. Beam object is modified.
        :return: None
        """

        self.trace_surface(beam)
        beam.beam_prop(-self.length / 2 * 1.1)

        # # initialize some arrays based on beam shape
        # shapeError2 = np.zeros_like(beam.x)
        # k_ix = 0
        # k_iy = 0
        # k_iz = 0
        # zi = np.zeros_like(beam.x)
        # yi = np.zeros_like(beam.x)
        # zi_1d = np.zeros(0)
        # yi_1d = np.zeros(0)
        #
        # # actual angle of incidence
        # total_alpha = self.alpha + self.delta
        #
        # # figure out outgoing k-vector based on incident beam and mirror orientation
        # if self.orientation == 0:
        #
        #     # account for change to angle of incidence
        #     total_alpha += -beam.ax
        #
        #     k_ix = -np.sin(self.alpha - beam.ax)
        #     k_iy = np.sin(beam.ay)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)
        #
        #     # coordinate mapping for interpolation
        #     zi = beam.x / np.sin(total_alpha)
        #     zi_1d = zi[0, :]
        #     yi = beam.y
        #     yi_1d = yi[:, 0]
        #
        # elif self.orientation == 1:
        #
        #     # account for change to angle of incidence
        #     total_alpha += -beam.ay
        #
        #     k_ix = -np.sin(self.alpha - beam.ay)
        #     k_iy = -np.sin(beam.ax)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)
        #
        #     # coordinate mapping for interpolation
        #     zi = beam.y / np.sin(total_alpha)
        #     zi_1d = zi[:, 0]
        #     yi = -beam.x
        #     yi_1d = yi[0, :]
        #
        # elif self.orientation == 2:
        #
        #     # account for change to angle of incidence
        #     total_alpha += beam.ax
        #
        #     k_ix = -np.sin(self.alpha + beam.ax)
        #     k_iy = -np.sin(beam.ay)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)
        #
        #     # coordinate mapping for interpolation
        #     zi = -beam.x / np.sin(total_alpha)
        #     zi_1d = zi[0, :]
        #     yi = -beam.y
        #     yi_1d = yi[:, 0]
        #
        # elif self.orientation == 3:
        #
        #     # account for change to angle of incidence
        #     total_alpha += beam.ay
        #
        #     k_ix = -np.sin(self.alpha + beam.ay)
        #     k_iy = beam.ax
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)
        #
        #     # coordinate mapping for interpolation
        #     zi = -beam.y / np.sin(total_alpha)
        #     zi_1d = zi[:, 0]
        #     yi = beam.x
        #     yi_1d = yi[0, :]
        #
        # k_i = np.array([k_ix, k_iy, k_iz])
        # delta_k = self.rotation(k_i)
        #
        # # mirror shape error interpolation onto beam coordinates (if applicable)
        # if self.shapeError is not None:
        #     # get shape of shape error input
        #     mirror_shape = np.shape(self.shapeError)
        #
        #     # assume this is the central line shaper error along the long axis if only 1D
        #     if np.size(mirror_shape) == 1:
        #         # assume this is the central line and it's the same across the mirror width
        #         Ms = mirror_shape[0]
        #         # mirror coordinates (beam coordinates)
        #         max_zs = self.length / 2
        #         # mirror coordinates
        #         zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
        #         # 1D interpolation onto beam coordinates
        #         central_line = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError)
        #         # tile onto mirror short axis direction
        #         shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))
        #     # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
        #     else:
        #         # shape error array shape
        #         Ns = mirror_shape[0]
        #         Ms = mirror_shape[1]
        #         # mirror coordinates
        #         max_xs = self.length / 2
        #         # mirror coordinates
        #         zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
        #         max_ys = self.width / 2
        #         ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)
        #
        #         # 2D interpolation onto beam coordinates
        #         f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
        #         shapeError2 = f(zi_1d - self.dx / np.tan(total_alpha), yi_1d - self.dy)
        #
        # # figure out aperturing due to mirror's finite size
        # z_mask = (np.abs(zi - self.dx / np.tan(total_alpha)) < self.length / 2).astype(float)
        # y_mask = (np.abs(yi - self.dy) < self.width / 2).astype(float)
        #
        # # 2D mirror aperture (1's and 0's)
        # mirror = z_mask * y_mask
        #
        # # height error now in meters
        # total_error = shapeError2 * 1e-9
        #
        # # convert to phase error (additional factor of 2 due to reflection
        # phase = -total_error * 4 * np.pi * np.sin(total_alpha) / beam.lambda0
        #
        # # modify beam's wave attribute by mirror aperture and phase error
        # beam.wave *= mirror * np.exp(1j * phase)
        #
        # # now change outgoing beam k-vector based on mirror orientation
        # if self.orientation == 0:
        #     # take into account mirror reflection causing beam to invert
        #     beam.x *= -1
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_azimuth=2 * self.alpha)
        #     delta_ax = -2 * beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ay = np.arcsin(delta_k[1])
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #     # beam.rotate_beam(delta_ax=)
        #     # beam.ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # beam.ay += np.arcsin(delta_k[1])
        #
        #     # adjust beam position due to mirror de-centering
        #     # beam.beam_offset(x_offset=2 * self.dx * np.cos(total_alpha))
        #     delta_cx = 2 * self.dx * np.cos(total_alpha)
        #     beam.cx = -beam.cx + delta_cx
        #     beam.x = beam.x + delta_cx
        #
        # elif self.orientation == 1:
        #     # take into account mirror reflection causing beam to invert
        #     beam.y *= -1
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_elevation=2 * self.alpha)
        #     delta_ay = -2 * beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ax = -np.arcsin(delta_k[1])
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax += -np.arcsin(delta_k[1])
        #     # beam.ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.alpha))
        #
        #     # adjust beam position due to mirror de-centering
        #     # beam.beam_offset(y_offset=2 * self.dx * np.cos(total_alpha))
        #     delta_cy = 2 * self.dx * np.cos(total_alpha)
        #     beam.cy = -beam.cy + delta_cy
        #     beam.y = beam.y + delta_cy
        #
        # elif self.orientation == 2:
        #     # take into account mirror reflection causing beam to invert
        #     beam.x *= -1
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_azimuth=-2 * self.alpha)
        #     delta_ax = -2 * beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ay = -np.arcsin(delta_k[1])
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # beam.ay += -np.arcsin(delta_k[1])
        #
        #     # adjust beam position due to mirror de-centering
        #     # beam.beam_offset(x_offset=-2 * self.dx * np.cos(total_alpha))
        #     delta_cx = -2 * self.dx * np.cos(total_alpha)
        #     beam.cx = -beam.cx + delta_cx
        #     beam.x = beam.x + delta_cx
        #
        # elif self.orientation == 3:
        #     # take into account mirror reflection causing beam to invert
        #     beam.y *= -1
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_elevation=-2 * self.alpha)
        #     delta_ay = -2 * beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha))
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ax = np.arcsin(delta_k[1])
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax += np.arcsin(delta_k[1])
        #     # beam.ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.alpha))
        #
        #     # adjust beam position due to mirror de-centering
        #     # beam.beam_offset(y_offset=-2 * self.dx * np.cos(total_alpha))
        #     delta_cy = -2 * self.dx * np.cos(total_alpha)
        #     beam.cy = -beam.cy + delta_cy
        #     beam.y = beam.y + delta_cy

        return

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

    def trace_surface(self, beam):

        figon = self.show_figures
        # global unit vectors
        ux = np.reshape(np.array([1,0,0]),(3,1))
        uy = np.reshape(np.array([0,1,0]),(3,1))
        uz = np.reshape(np.array([0,0,1]),(3,1))

        delta_z = self.length / 2 * 1.1

        print('ax: %.6e' % beam.ax)
        print('ay: %.6e' % beam.ay)

        # propagate beam to just upstream of mirror
        beam.beam_prop(-delta_z)

        # vector defining displacement from beam location to mirror center. This is in global coordinates
        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx
        beam_to_mirror = beam_center - mirror_center

        # unit vectors for ellipse coordinates, written in global coordinates
        mirror_x = self.normal
        mirror_y = self.sagittal
        mirror_z = self.transverse

        # define beam rays up to second order (assume that linear term is not needed and is
        # already captured in the k-vector
        rays_x = beam.x/beam.zx
        rays_y = beam.y/beam.zy

        # beam coordinates in global coordinates
        coords = np.multiply.outer(beam.xhat, beam.x) + np.multiply.outer(beam.yhat, beam.y)

        # reference to global origin by adding beam global center
        coords += np.reshape(beam_center, (3, 1, 1))

        # now subtract mirror center so that beam coordinates are in global coordinates,
        # but with origin at mirror center
        coords -= np.reshape(mirror_center, (3, 1, 1))

        # now write beam coordinates in mirror plane coordinates
        # (transforming from global coordinates to mirror coordinates)
        transform_matrix = np.tensordot(np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_mirror = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # mirror vectors written in mirror coordinates
        mirror_x_local = ux
        mirror_y_local = uy
        mirror_z_local = uz

        # calculate z component of rays (enforcing unit vector)
        rays_z = np.sqrt(np.ones_like(rays_x) - rays_x ** 2 - rays_y ** 2)
        # ray vectors at each point in the beam
        rays = (np.multiply.outer(beam.xhat, rays_x) + np.multiply.outer(beam.zhat, rays_z)
                + np.multiply.outer(beam.yhat, rays_y))

        # normalize rays (should be redundant)
        rays = rays / np.sqrt(np.sum(rays*rays, axis=0))

        # now write rays in mirror coordinates
        rays_mirror = np.tensordot(transform_matrix, rays, axes=(1,0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2,:,0],coords_mirror[0,:,0])
            plt.quiver(coords_mirror[2,:,0],coords_mirror[0,:,0],rays_mirror[2,:,0],rays_mirror[0,:,0])
            plt.ylim(-.5,.5)
            plt.grid()
            plt.title('incoming rays and mirror')

        z_intersect = coords_mirror[2, :, :] - rays_mirror[2, :, :] / rays_mirror[0, :, :] * coords_mirror[0, :, :]
        # by definition the x coordinate is zero for intersection
        Ni, Mi = np.shape(z_intersect)
        x_intersect = np.zeros_like(z_intersect)
        y_intersect = (rays_mirror[1, :, :] / rays_mirror[2, :, :]
                       * (z_intersect - coords_mirror[2, :, :]) + coords_mirror[1, :, :])

        # put mirror intersection coordinates in a 3xNxM array
        intersect_coords = np.zeros((3,Ni,Mi))
        intersect_coords[0,:,:] = x_intersect
        intersect_coords[1,:,:] = y_intersect
        intersect_coords[2,:,:] = z_intersect

        # vectors pointing from beam location to mirror intersection
        i_vector = intersect_coords - coords_mirror

        # length of each vector
        distance_1 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        # define normals along mirror surface
        mirror_normal = np.zeros_like(rays)
        mirror_normal[0,:,:] = np.ones_like(z_intersect)

        # define crystal plane normal
        # crystal plane normal vector (this is in the coordinates of the crystal)
        c_x = np.cos(self.alphaAsym)
        c_z = np.sin(self.alphaAsym)
        c_normal = np.reshape(np.array([c_x, 0, c_z], dtype=float), (3,1))

        g_parallel = np.sum(c_normal * mirror_z_local, axis=0) * beam.lambda0 / self.d
        rays_out = np.zeros_like(rays_mirror)
        rays_out[1,:,:] = rays_mirror[1,:,:]
        rays_out[2,:,:] = rays_mirror[2,:,:] + g_parallel
        rays_out[0,:,:] = np.sqrt(np.ones_like(rays_out[0,:,:]) - rays_out[1,:,:]**2 - rays_out[2,:,:]**2)

        # calculate ray direction after interaction with ellipse using law of reflection
        # rays_out = rays_mirror - 2 * np.sum(rays_mirror*mirror_normal,axis=0) * mirror_normal

        # now find intersection with exit plane
        # we can define this simply as having a normal vector in the direction of the central ray
        # and we will define the plane to be a distance length/2*1.1 from the intersection point of the central ray
        plane_normal = np.reshape(rays_out[:,int(beam.N/2), int(beam.M/2)],(3,1,1))
        central_point = np.reshape(intersect_coords[:,int(beam.N/2),int(beam.M/2)],(3,1,1)) + plane_normal*self.length/2*1.1

        # find z intersection with this plane
        d2 = np.sum((central_point - intersect_coords)*plane_normal,axis=0)/np.sum(rays_out*plane_normal,axis=0)
        plane_intersect = intersect_coords + rays_out*d2
        i_vector = plane_intersect - intersect_coords
        distance_2 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2, :, 0], coords_mirror[0, :, 0])
            plt.plot(z_intersect, x_intersect)
            plt.plot(plane_intersect[2,:, 0],plane_intersect[0,:, 0])
            # plt.ylim(-.5, .5)
            plt.grid()
            plt.title('entrance/exit planes, mirror intersection')

        # total distance for each beam ray
        total_distance = (distance_1+distance_2)
        #
        if figon:
            plt.figure()
            plt.plot(intersect_coords[2,:,0],distance_1)
            plt.plot(intersect_coords[2,:,0],distance_2)
            plt.plot(intersect_coords[2,:,0],distance_1+distance_2)
            plt.title('distances')

        # find location of central ray in exit plane
        origin = np.reshape(plane_intersect[:,int(beam.N/2),int(beam.M/2)],(3,1,1))

        # put beam center at origin
        shifted_plane = plane_intersect-origin

        # get final k-vector for central ray
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        # convert to global coordinates
        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))
        k_f_global = k_f_global[:,0,0]

        # first rotate by the "nominal" amount
        if self.orientation==0:
            beam.rotate_nominal(delta_azimuth=self.alpha+self.beta0)
        elif self.orientation==1:
            beam.rotate_nominal(delta_elevation=self.alpha+self.beta0)
        elif self.orientation==2:
            beam.rotate_nominal(delta_azimuth=-self.alpha-self.beta0)
        elif self.orientation==3:
            beam.rotate_nominal(delta_elevation=-self.alpha-self.beta0)

        # get initial k-vector for central ray in global coordinates
        k_i = np.copy(beam.zhat)

        # find the change in the k-vector in global coordinates
        delta_k = k_f_global - k_i

        print('xhat: {}'.format(beam.xhat))
        print('yhat: {}'.format(beam.yhat))
        print('zhat: {}'.format(beam.zhat))
        print('dk: {}'.format(delta_k))

        # now make minor adjustment to k-vector based on central ray at exit plane
        # might want to do one axis at a time or change the order. Or could change the rotation
        # to rotate about the "unrotated" axes.
        delta_ax = np.arcsin(delta_k[0])
        x_sign = np.sign(np.dot(np.cross(k_i,k_f_global),beam.yhat))
        delta_ay = -np.arcsin(delta_k[1])
        y_sign = np.sign(-np.dot(np.cross(k_i,k_f_global),beam.xhat))
        beam.rotate_beam(delta_ax=x_sign*np.abs(delta_ax), delta_ay=y_sign*np.abs(delta_ay))

        # check for consistency
        print('is beam in the correct direction?')
        # print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global)))

        # mask defining mirror acceptance
        # if self.q>=0:
        #     mask = np.logical_and(coords_ellipse[0,:,:]>intersect_coords[0,:,:],
        #                           plane_intersect[0,:,:]>intersect_coords[0,:,:])
        # else:
        #     mask = np.logical_and(coords_ellipse[0, :,:] < intersect_coords[0, :,:],
        #                           plane_intersect[0, :,:] > intersect_coords[0, :,:])
        #
        # # second order fit to ray distance
        # if self.q>=0:
        #     mask = np.logical_and(mask, coords_ellipse[0,:,:]<0)
        # else:
        #     mask = np.logical_and(mask, coords_ellipse[0, :,:] > 0)

        # can define mirror acceptance by taking dot product with mirror unit vectors (in ellipse coordinate system).
        # distance along length axis
        d_length = np.sum((intersect_coords)*np.reshape(mirror_z_local,(3,1,1)),axis=0)
        # distance along width axis
        d_width = np.sum((intersect_coords)*np.reshape(mirror_y_local,(3,1,1)),axis=0)

        mask = np.ones_like(d_length)

        # d_length and d_width are equivalent to the mirror coordinates, these can be used for interpolating
        # the shape error
        # mirror shape error interpolation onto beam coordinates (if applicable)

        shapeInterp = np.zeros((beam.N,beam.M))

        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                max_ys = self.width / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                ys = np.linspace(-1,1,100)*max_ys

                print(np.shape(np.tile(self.shapeError,(100,1))))
                f = RectBivariateSpline(ys,zs,np.tile(self.shapeError,(100,1)))
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N, beam.M))*1e-9

                plt.figure()
                plt.imshow(shapeInterp)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = RectBivariateSpline(ys,zs,self.shapeError)
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N,beam.M))*1e-9
                # f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                # shapeError2 = f(zi_1d - self.dx / np.tan(self.total_alpha), yi_1d - self.dy)

        gratingPhase = -2*np.pi*np.sin(self.alphaAsym)*d_length/self.d

        # mask based on mirror length
        mask = np.logical_and(mask, np.abs(d_length)<self.length/2)
        # mask based on mirror width
        mask = np.logical_and(mask, np.abs(d_width)<self.width/2)

        # now write new beam coordinates in local beam coordinate system
        # (transforming from ellipse coordinates to local beam coordinates)
        transform_matrix2 = np.tensordot(np.reshape([beam.xhat,beam.yhat,beam.zhat], (3, 3)),
                                        np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)), axes=(1, 1))
        shifted_plane2 = np.tensordot(transform_matrix2, shifted_plane, axes=(1, 0))

        # beam's x and y coordinates in the exit beam local coordinate system
        # note these are not in general on a regularly spaced grid but are based on tracking
        # rays through the course of the reflection.
        x_eff = shifted_plane2[0,:,:]
        y_eff = shifted_plane2[1,:,:]

        # calculate desired pixel size due to expected change in beam size (based on b factor), and
        # define new coordinate grids to interpolate onto at beam exit plane.
        if self.orientation == 0 or self.orientation == 2:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx / self.b
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)
        else:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy / self.b
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)

        # 1D masks based on mirror acceptance, going through beam center.
        max_intensity = np.max(np.abs(beam.wave)**2)
        # mask2 = np.logical_and(mask,np.abs(beam.wave)**2>0.1*max_intensity)
        mask_x = mask[int(beam.N/2),:]>0
        mask_y = mask[:,int(beam.M/2)]>0
        # introducing weights for the polynomial fits based on beam intensity

        weight_x = (np.abs(beam.wave[int(beam.N/2),:])**2>0.1*max_intensity).astype(float)
        weight_y = (np.abs(beam.wave[:,int(beam.M/2)])**2>0.1*max_intensity).astype(float)

        # plt.figure()
        # plt.imshow((total_distance-np.mean(total_distance[mask]))*mask)
        #
        # plt.figure()
        # plt.imshow(np.abs(beam.wave)**2>0.1*max_intensity)

        incidence_angle = np.abs(np.arccos(np.sum(rays_out * mirror_normal, axis=0)))
        glancing = np.pi / 2 - incidence_angle

        # figure out crystal reflectivity
        beamInDotNormal = rays_mirror[0, :, :]
        beamOutDotNormal = rays_out[0, :, :]
        beamInDotHNormal = np.sum(rays_mirror * np.reshape(c_normal,(3,1,1)), axis=0)

        C1, C2 = np.array(self.crystal.get_amplitude(beam.photonEnergy,
                                                     beamInDotNormal, beamOutDotNormal, beamInDotHNormal))

        # choose complex reflectivity based on polarization
        if self.pol == 's':
            C = C1
        else:
            C = C2

        reflectivity = np.abs(C)

        plt.figure()
        plt.imshow(reflectivity)

        # reflectivity = xraydb.mirror_reflectivity(self.material, glancing, beam.photonEnergy, self.density)

        # plt.figure()
        # plt.imshow((glancing-np.mean(glancing[mask]))*mask)

        # subtract best fit parabola in x-direction and best fit line in y direction
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff)# + np.polyval(p_coeff_y[:-2],y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff)# + np.polyval(p_coeff_y[:-2], y_eff)
        #
        # TRY THIS OUT LATER
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff) + np.polyval(p_coeff_y,y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff) + np.polyval(p_coeff_x, x_eff)

        points = np.zeros((np.size(x_eff[mask]),2))
        points[:,0] = x_eff[mask]
        points[:,1] = y_eff[mask]
        # xi = np.zeros((np.size(x_out),2))
        # xi[:,0] = x_out
        # xi[:,1] = y_out
        xi_0, xi_1 = np.meshgrid(x_out, y_out)
        print('attempting interpolation')
        mask2 = interpolation.griddata(points, mask[mask], (xi_0, xi_1), method='nearest',fill_value=0)
        # mask2 = fmask(x_out,y_out)
        mask2[mask2<.9] = 0
        # mask2 = mask2.astype(int)
        mask2 = mask2 > 0.5

        # interpolate intensity onto new exit plane grid
        abs_out = interpolation.griddata(points, np.abs(beam.wave[mask]), (xi_0, xi_1), fill_value=0)

        # unwrap phase of beam at input
        angle_in = unwrap_phase(np.angle(beam.wave*C))

        # add quadratic phase if beam is not focused since this also needs to be interpolated
        quadratic = np.zeros_like(beam.x)

        if not beam.focused_x:
            quadratic += np.pi / beam.lambda0 / beam.zx * (beam.x) ** 2
        if not beam.focused_y:
            quadratic += np.pi / beam.lambda0 / beam.zy * (beam.y) ** 2

        # add the quadratic phase to the input phase
        angle_in += quadratic

        # add phase contribution from deviations in the distance traveled by each ray
        total_phase = (angle_in + 2 * np.pi / beam.lambda0 * total_distance
                       - shapeInterp * 4*np.pi*np.sin(self.alpha) / beam.lambda0 - gratingPhase)

        diff = gratingPhase-2 * np.pi / beam.lambda0 *total_distance
        diff -= np.mean(diff)

        p1 = np.polyfit(d_length[int(beam.N/2),:],total_distance[int(beam.N/2),:],1)
        plt.figure()
        plt.imshow(total_distance-np.polyval(p1,d_length))

        p1 = np.polyfit(d_length[int(beam.N/2),:],gratingPhase[int(beam.N/2),:],1)

        plt.figure()
        plt.imshow(gratingPhase-np.polyval(p1,d_length))

        # get polynomial fits based on new coordinates
        p_coeff_x = np.polyfit(x_eff[int(beam.N/2),:][mask_x], total_phase[int(beam.N/2),:][mask_x], 2,
                               w=weight_x[mask_x])

        p_coeff_y = np.polyfit(y_eff[:,int(beam.M/2)][mask_y], total_phase[:,int(beam.M/2)][mask_y], 2,
                               w=weight_y[mask_y])

        # calculate effective distance to focus based on total phase
        z_2 = np.pi / beam.lambda0 / p_coeff_x[-3]
        z_2_y = np.pi / beam.lambda0 / p_coeff_y[-3]

        # add this to what was already subtracted (might be able to do this in one step)
        # z_total_x = 1 / (1 / z_out_x + 1 / z_2)
        # z_total_y = 1 / (1 / z_out_y + 1 / z_2_y)
        z_total_x = z_2
        z_total_y = z_2_y
        print('new z: %.6f' % z_total_x)
        print('new z y: %.6f' % z_total_y)

        # calculate residual linear phase terms
        # linear += p_coeff_x[-2] * beam.lambda0/2/np.pi
        # linear_y += p_coeff_y[-2] * beam.lambda0/2/np.pi
        linear = p_coeff_x[-2] * beam.lambda0 / 2 / np.pi
        linear_y = p_coeff_y[-2] * beam.lambda0/2/np.pi

        print(linear)
        print(linear_y)

        # subtract linear terms since this should be taken care of based on central ray direction
        # Now this is being done in both tangential and sagittal directions, and we compensate with
        # a change in beam k-vector direction.
        total_phase -= np.polyval(p_coeff_x[-2:], x_eff) + np.polyval(p_coeff_y[-2:], y_eff)

        plt.figure()
        plt.imshow(total_phase * mask)

        # sutbtract off quadratic phase term if the beam is not focused
        if not beam.focused_x:
            total_phase -= np.polyval([p_coeff_x[-3], 0, 0], x_eff)
        if not beam.focused_y:
            total_phase -= np.polyval([p_coeff_y[-3], 0, 0], y_eff)

        # interpolate the phase onto the exit plane grid
        points = np.zeros((np.size(x_eff), 2))
        points[:, 0] = x_eff.flatten()
        points[:, 1] = y_eff.flatten()
        phase_interp = interpolation.griddata(points, total_phase.flatten(), (xi_0, xi_1), fill_value=0)

        # interpolate the reflectivity onto the exit plane grid
        reflectivity_interp = interpolation.griddata(points, reflectivity.flatten(), (xi_0, xi_1), fill_value=0)

        # update beam complex amplitude using what has been interpolated onto the new grid
        beam.wave = abs_out * np.exp(1j * phase_interp)
        # multiply by reflectivity if this is being used
        if self.use_reflectivity:
            beam.wave *= np.sqrt(reflectivity_interp)
        # multiply by mirror aperture that has been interpolated onto the exit plane grid
        beam.wave *= mask2

        if figon:
            plt.figure()
            plt.imshow(np.abs(beam.wave))

        ax0 = np.copy(beam.ax)
        ay0 = np.copy(beam.ay)

        # figure out where the beam is in global coordinates
        # change in angle
        k_i = rays_mirror[:, int(beam.N / 2), int(beam.M / 2)]
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))

        # compensate for removing linear phase by adjusting beam k-vector
        beam.rotate_beam(delta_ax=linear)
        beam.rotate_beam(delta_ay=linear_y)

        print(beam.ax)
        delta_cx = ax0 * self.length / 2 * 1.1
        delta_cx += beam.ax * self.length / 2 * 1.1
        delta_cx += 2 * np.dot(self.normal, beam.xhat) * self.dx
        print('change in beam center')
        print(delta_cx)
        # beam.cx = -beam.cx + delta_cx
        # print(beam.cx)

        x_out, y_out = np.meshgrid(x_out, y_out)
        beam.x = x_out
        beam.y = y_out

        beam.new_fx()

        print('is beam in the correct direction?')
        print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global[:, 0, 0])))
        print(k_f)
        print(k_f_global)

        # now figure out global coordinates
        # get back into global coordinates using inverse of transformation matrix, just looking at central ray
        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        origin_global = np.tensordot(inv_transform, origin, axes=(1,0))

        # now add the mirror center in global coordinates, so that this should be the beam location
        # in global coordinates
        origin_global += np.reshape(mirror_center, (3, 1, 1))
        # origin_global -= np.reshape(self.normal*dx,(3,1))
        print(origin_global)
        # now shift origin to ellipse origin

        beam.global_x = origin_global[0,0,0]
        beam.global_y = origin_global[1,0,0]
        beam.global_z = origin_global[2,0,0]

        # account for coordinate scaling and/or changes to Rayleigh range,
        # whether beam is classified as focused or not.
        beam.change_z(new_zx=z_total_x,new_zy=z_total_y)

        beam.new_fx()
        print('global_x: %.2f' % beam.global_x)
        print('global_y: %.2f' % beam.global_y)
        print('global_z: %.2f' % beam.global_z)

        print(np.shape(beam.x))
        print(np.shape(beam.y))

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

    def define_ki(self, beam, alpha_in):

        # unit vectors
        u_x = np.array([1, 0, 0], dtype=float)
        u_y = np.array([0, 1, 0], dtype=float)
        u_z = np.array([0, 0, 1], dtype=float)

        k_ix = np.sin(beam.ax)
        k_iy = np.sin(beam.ay)
        k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

        k_ix = k_ix * u_x
        k_iy = k_iy * u_y
        k_iz = k_iz * u_z

        k_i = k_ix + k_iy + k_iz

        # rotate based on orientation
        r1 = transform.Rotation.from_rotvec(-u_z * np.pi / 2 * self.orientation)
        Rz = r1.as_matrix()

        k_i = np.matmul(Rz, k_i)

        # now rotate into crystal coordinates based on angle of incidence
        r2 = transform.Rotation.from_rotvec(-u_y * alpha_in)
        Ry = r2.as_matrix()

        k_i = np.matmul(Ry, k_i)

        return k_i

    def define_ki_2d(self, beam, alpha_in, zi):

        # unit vectors
        u_x = np.array([1, 0, 0], dtype=float)
        u_y = np.array([0, 1, 0], dtype=float)
        u_z = np.array([0, 0, 1], dtype=float)

        k_ix = np.sin(beam.ax)
        k_iy = np.sin(beam.ay)
        k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

        # calculate gradient of wavefront
        wavefront = np.copy(beam.wave)

        if beam.focused_x:
            print('subtracting horizontal second order')
            wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zx * (beam.x - beam.cx) ** 2)
        if beam.focused_y:
            print('subtracting vertical second order')
            wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zy * (beam.y - beam.cy) ** 2)

        beam_slope_error_y, beam_slope_error_x = np.gradient(unwrap_phase(np.angle(wavefront)),
                                                             beam.y[:,0], beam.x[0,:])
        beam_slope_error_y *= beam.lambda0 / 2 / np.pi
        beam_slope_error_x *= beam.lambda0 / 2 / np.pi

        zx_eff = beam.zx + zi * np.cos(alpha_in)
        zy_eff = beam.zy + zi * np.cos(alpha_in)

        k_ix_2d = beam.x/beam.zx + k_ix + beam_slope_error_x
        k_iy_2d = beam.y/beam.zy + k_iy + beam_slope_error_y
        k_iz_2d = np.sqrt(np.ones_like(beam.x) - k_ix_2d ** 2 - k_iy_2d ** 2)

        k_ix_2d = np.multiply.outer(u_x, k_ix_2d)
        k_iy_2d = np.multiply.outer(u_y, k_iy_2d)
        k_iz_2d = np.multiply.outer(u_z, k_iz_2d)

        k_i = k_ix_2d + k_iy_2d + k_iz_2d

        # rotate based on orientation
        r1 = transform.Rotation.from_rotvec(u_z * np.pi / 2 * self.orientation)
        Rz = r1.as_matrix()

        k_i = np.tensordot(Rz, k_i, axes=(0, 0))

        # now rotate into crystal coordinates based on angle of incidence
        r2 = transform.Rotation.from_rotvec(u_y * alpha_in)
        Ry = r2.as_matrix()

        k_i = np.tensordot(Ry, k_i, axes=(0, 0))

        return k_i

    def calc_kf_2d(self, k_i_2d, slope_error_z, slope_error_y, lambda0):

        # unit vectors
        u_x = np.array([1, 0, 0], dtype=float)
        u_y = np.array([0, 1, 0], dtype=float)
        u_z = np.array([0, 0, 1], dtype=float)

        # define crystal plane normal at every coordinate, including slope error
        # A positive asymmetry angle means that the normal has a positive z component
        c_y = np.multiply.outer(u_y, slope_error_y)
        c_z = np.multiply.outer(u_z, np.sin(self.alphaAsym) + slope_error_z)
        c_x = np.multiply.outer(u_x, np.sqrt(np.ones_like(slope_error_z)
                                        - np.sum(c_y * c_y, axis=0)
                                        - np.sum(c_z * c_z, axis=0)))

        c_normal = c_x + c_y + c_z

        k_fy = k_i_2d[1,:,:] + c_y[1,:,:] * lambda0 / (self.crystal.d * 1e-10)
        k_fz = k_i_2d[2,:,:] + c_z[2,:,:] * lambda0 / (self.crystal.d * 1e-10)
        k_fx = np.multiply.outer(u_x, np.sqrt(np.ones_like(slope_error_z)
                                         - k_fy**2 - k_fz**2))

        k_fy = np.multiply.outer(u_y, k_fy)
        k_fz = np.multiply.outer(u_z, k_fz)

        k_f = k_fx + k_fy + k_fz

        return k_f, c_normal, u_x

    # def calc_kf(self, z_s, k_i_avg, wavefront, zx, zy, alpha, slope_error_z, slope_error_y, lambda0):
    #     # calculate diffraction angle at every point on the grating
    #     # beta = np.arccos(np.cos(alpha_total) - beam.lambda0 * (self.n0 + self.n1 * z_g + self.n2 * z_g ** 2))
    #     m_x = np.array([1, 0, 0], dtype=float)
    #     m_y = np.array([0, 1, 0], dtype=float)
    #     m_z = np.array([0, 0, 1], dtype=float)
    #
    #
    #
    #     # define k_i at each point along beam
    #     k_ix = np.multiply.outer(-np.sin(alpha_in), m_x)
    #     k_iy = np.multiply.outer(k_iy, m_y)
    #     # k_iz = np.outer(np.cos(alpha_total), m_z)
    #     k_iz = np.multiply.outer(
    #         np.sqrt(np.ones_like(z_s) - np.sum(k_ix * k_ix, axis=2) - np.sum(k_iy * k_iy, axis=2)) * np.sign(
    #             np.cos(alpha_in)), m_z)
    #     k_i = k_ix + k_iy + k_iz
    #
    #     # define crystal plane at every coordinate including slope error
    #     # c_x = np.outer(np.cos(self.alphaAsym - slope_error), m_x)
    #     c_z = np.multiply.outer(np.sin(self.alphaAsym - slope_error_z), m_z)
    #     c_y = np.multiply.outer(np.sin(-slope_error_y), m_y)
    #     c_x = np.multiply.outer(np.sqrt(np.ones_like(z_s) - np.sum(c_y * c_y, axis=2) - np.sum(c_z * c_z, axis=2)), m_x)
    #     c_normal = c_x + c_y + c_z
    #
    #     c_parallel_z = np.multiply.outer(np.sum(c_normal * m_z, axis=2), m_z) * lambda0 / (self.crystal.d * 1e-10)
    #     c_parallel_y = np.multiply.outer(np.sum(c_normal * m_y, axis=2), m_y) * lambda0 / (self.crystal.d * 1e-10)
    #
    #     k_fy = k_iy + c_parallel_y
    #     k_fz = k_iz + c_parallel_z
    #     k_fx = np.multiply.outer(np.sqrt(np.ones_like(z_s) - np.sum(k_fy * k_fy, axis=2) - np.sum(k_fz * k_fz, axis=2)), m_x)
    #
    #     k_f = k_fy + k_fz + k_fx
    #
    #     return k_i, k_f, m_x, c_normal

    def diffract(self, beam):
        """
        Method to calculate diffraction from a grating, including VLS parameters.
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """

        self.trace_surface(beam)
        beam.beam_prop(-self.length / 2 * 1.1)

        # total_alpha = self.alpha + self.delta
        #
        # # initialize some arrays
        # shapeError2 = np.zeros_like(beam.x)
        # k_ix = 0
        # k_iy = 0
        # k_iz = 0
        # alphaBeam = np.zeros_like(beam.x)
        # k_iy_total = np.zeros_like(beam.x)
        # zi = np.zeros_like(beam.x)
        # yi = np.zeros_like(beam.x)
        # zi_1d = np.zeros(0)
        # yi_1d = np.zeros(0)
        # cz = 0
        # cy = 0
        #
        # beamz = 0
        # beamz_y = 0
        #
        # wavefront = np.copy(beam.wave)
        #
        # if beam.focused_x:
        #     print('subtracting horizontal second order')
        #     wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zx * (beam.x - beam.cx) ** 2)
        # if beam.focused_y:
        #     print('subtracting vertical second order')
        #     wavefront *= np.exp(-1j * np.pi / beam.lambda0 / beam.zy * (beam.y - beam.cy) ** 2)
        #
        #
        #
        # if self.orientation == 0:
        #     # account for change to angle of incidence
        #     total_alpha -= beam.ax
        #
        #     # coordinate mapping for interpolation
        #     zi = beam.x / np.sin(total_alpha)
        #     zi_1d = zi[0, :]
        #     yi = beam.y
        #     yi_1d = yi[:, 0]
        #
        #
        #
        #     # k_ix = -np.sin(total_alpha)
        #     k_ix = -np.sin(self.alpha - beam.ax)
        #     k_iy = np.sin(beam.ay)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha - beam.ax))
        #     # k_iz = np.cos(total_alpha)
        #
        #
        #
        #     cz = beam.cx / np.sin(total_alpha)
        #     cy = beam.cy
        #
        #     # beam radius across grating (grating can be long enough that the additional correction is needed
        #     zEff = beam.zx + (zi - cz) * np.cos(total_alpha)
        #     alphaBeam = -beam.ax - np.arctan((zi - cz) * np.sin(total_alpha) / zEff)
        #     zEff_y = beam.zy + (zi - cz) * np.cos(total_alpha)
        #     k_iy_total = np.sin(beam.ay + yi/zEff_y)
        #
        #     self.f = -beam.zx * (np.abs(np.sin(self.beta0)/np.sin(self.alpha))**2)
        #     # self.f = -beam.zx
        #     beamz = beam.zx
        #     beamz_y = beam.zy
        #
        # elif self.orientation == 1:
        #     # account for change to angle of incidence
        #     total_alpha -= beam.ay
        #
        #     # k_ix = -np.sin(total_alpha)
        #     k_ix = -np.sin(self.alpha - beam.ay)
        #     k_iy = -np.sin(beam.ax)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha - beam.ay))
        #     # k_iz = np.cos(total_alpha)
        #
        #     # coordinate mapping for interpolation
        #     zi = beam.y / np.sin(total_alpha)
        #     zi_1d = zi[:, 0]
        #     yi = -beam.x
        #     yi_1d = yi[0, :]
        #
        #     cz = beam.cy / np.sin(total_alpha)
        #     cy = -beam.cx
        #
        #     # beam radius across grating (grating can be long enough that the additional correction is needed
        #     zEff = beam.zy + (zi - cz) * np.cos(total_alpha)
        #     alphaBeam = -beam.ay - np.arctan((zi - cz) * np.sin(total_alpha) / zEff)
        #     zEff_y = beam.zx + (zi - cz) * np.cos(total_alpha)
        #     k_iy_total = np.sin(-beam.ax + yi / zEff_y)
        #
        #     self.f = -beam.zy * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
        #     # self.f = -beam.zy
        #     beamz = beam.zy
        #     beamz_y = beam.zx
        #
        #     wavefront = np.swapaxes(wavefront, 0, 1)
        #
        # elif self.orientation == 2:
        #     # account for change to angle of incidence
        #     total_alpha += beam.ax
        #
        #     # k_ix = -np.sin(total_alpha)
        #     k_ix = -np.sin(self.alpha + beam.ax)
        #     k_iy = -np.sin(beam.ay)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha + beam.ax))
        #     # k_iz = np.cos(total_alpha)
        #
        #     # coordinate mapping for interpolation
        #     zi = -beam.x / np.sin(total_alpha)
        #     zi_1d = zi[0, :]
        #     yi = -beam.y
        #     yi_1d = yi[:, 0]
        #
        #     cz = -beam.cx / np.sin(total_alpha)
        #     cy = -beam.cy
        #
        #     # beam radius across grating (grating can be long enough that the additional correction is needed
        #     zEff = beam.zx + (zi - cz) * np.cos(total_alpha)
        #     alphaBeam = beam.ax - np.arctan((zi - cz) * np.sin(total_alpha) / zEff)
        #     zEff_y = beam.zy + (zi - cz) * np.cos(total_alpha)
        #     k_iy_total = np.sin(-beam.ay + yi / zEff_y)
        #
        #     self.f = -beam.zx * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
        #     # self.f = -beam.zx
        #     beamz = beam.zx
        #     beamz_y = beam.zy
        #
        # elif self.orientation == 3:
        #     # account fo change to angle of incidence
        #     total_alpha += beam.ay
        #
        #     # k_ix = -np.sin(total_alpha)
        #     k_ix = -np.sin(self.alpha + beam.ay)
        #     k_iy = np.sin(beam.ax)
        #     k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2) * np.sign(np.cos(self.alpha + beam.ay))
        #     # k_iz = np.cos(total_alpha)
        #
        #     # coordinate mapping for interpolation
        #     zi = -beam.y / np.sin(total_alpha)
        #     zi_1d = zi[:, 0]
        #     yi = beam.x
        #     yi_1d = yi[0, :]
        #
        #     cz = -beam.cy / np.sin(total_alpha)
        #     cy = beam.cx
        #
        #     # beam radius across grating (grating can be long enough that the additional correction is needed
        #     zEff = beam.zy + (zi - cz) * np.cos(total_alpha)
        #
        #     alphaBeam = beam.ay - np.arctan((zi - cz) * np.sin(total_alpha) / zEff)
        #     zEff_y = beam.zx + (zi - cz) * np.cos(total_alpha)
        #     k_iy_total = np.sin(beam.ax + yi / zEff_y)
        #
        #     self.f = -beam.zy * (np.abs(np.sin(self.beta0) / np.sin(self.alpha)) ** 2)
        #     # self.f = -beam.zy
        #     beamz = beam.zy
        #     beamz_y = beam.zx
        #
        #     wavefront = np.swapaxes(wavefront, 0, 1)
        #
        # # mirror shape error interpolation onto beam coordinates (if applicable)
        # if self.shapeError is not None:
        #     # get shape of shape error input
        #     mirror_shape = np.shape(self.shapeError)
        #
        #     # assume this is the central line shaper error along the long axis if only 1D
        #     if np.size(mirror_shape) == 1:
        #         # assume this is the central line and it's the same across the mirror width
        #         Ms = mirror_shape[0]
        #         # mirror coordinates (beam coordinates)
        #         max_zs = self.length / 2
        #         # mirror coordinates
        #         zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
        #         # 1D interpolation onto beam coordinates
        #         central_line = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError)
        #         # tile onto mirror short axis direction
        #         shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))
        #     # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
        #     else:
        #         # shape error array shape
        #         Ns = mirror_shape[0]
        #         Ms = mirror_shape[1]
        #         # mirror coordinates
        #         max_xs = self.length / 2
        #         # mirror coordinates
        #         zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
        #         max_ys = self.width / 2
        #         ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)
        #
        #         # 2D interpolation onto beam coordinates
        #         f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
        #
        #         shapeError2 = f(zi_1d - self.dx / np.tan(total_alpha), yi_1d - self.dy)
        #
        # # get slope error
        # # for now just do this in 1D, update for 2D later
        # Ns, Ms = np.shape(shapeError2)
        # # shapeError2 = shapeError2[int(Ns/2),:]
        #
        # # get lineouts at crystal center
        # shape_lineout_z = shapeError2[int(Ns/2),:]
        # shape_lineout_y = shapeError2[:,int(Ms/2)]
        # # shapePoly = np.polyfit(zi_1d, shapeError2[int(Ns/2),:], 16)
        # # slopePoly = np.polyder(shapePoly)
        # # slope_error = np.polyval(slopePoly, zi_1d) * 1e-9
        #
        # # to make the mask, we need coordinates that are centered on the crystal,
        # # with no offset this is zi_1d, with offset this is z_c = zi_1d - self.dx / np.tan(total_alpha)
        # z_c = zi_1d - self.dx / np.tan(total_alpha)
        # z_c_2d = zi - self.dx / np.tan(total_alpha)
        #
        # # similar coordinates for sagittal direction
        # y_c = yi_1d - self.dy
        #
        # y_c_2d = yi - self.dy
        #
        # # beam-centered coordinates are zi_1d - cz (meaning zero at beam center) - we will call this z_b
        # # This implies that z_b = z_c + self.dx / np.tan(total_alpha) - cz, which is consistent with the
        # # offset applied below
        # z_b = zi_1d - cz
        #
        # # similar coordinates for sagittal direction
        # y_b = yi_1d - cy
        #
        # # in the end we need a polynomial that's centered on the beam (meaning on z_b)
        #
        # # limit fit to size of crystal
        # mask_z = np.abs(z_c) <= self.length / 2
        # print(np.sum(mask_z) / np.size(z_c))
        # mask_y = np.abs(y_c) <= self.width / 2
        #
        # shapePoly_z = LegendreUtil(z_c[mask_z], shape_lineout_z[mask_z], 16)
        # shapePoly_y = LegendreUtil(y_c[mask_y], shape_lineout_y[mask_y], 16)
        #
        # # get second order term of Legendre fit for curved crystal calculation
        # second_order_z = shapePoly_z.quad_coeff()
        # second_order_y = shapePoly_y.quad_coeff()
        #
        # # take gradient to get slope error
        # slope_error_y, slope_error_z = np.gradient(shapeError2,y_c,z_c)
        #
        # # to get outward-facing surface normals need to multiply by -1
        # slope_error_y *= -1e-9
        # slope_error_z *= -1e-9
        #
        # # calculate nominal reflected k vector
        # k_i = np.array([k_ix, k_iy, k_iz])
        # delta_k, k_f = self.rotation_crystal(k_i, beam.lambda0)
        # print(delta_k)
        #
        # # beta at beam center
        # beta1 = np.arccos(k_f[2])
        #
        # # calculate incident k-vector in crystal coordinates
        # k_i_2d = self.define_ki_2d(beam, self.alpha + self.delta, zi-cz)
        # # calculate final k-vector after interaction with crystal
        # k_f_2d, c_normal, u_x = self.calc_kf_2d(k_i_2d, slope_error_z, slope_error_y, beam.lambda0)
        #
        # # plt.figure()
        # # plt.imshow(k_f_2d[0, :, :])
        # # plt.figure()
        # # plt.imshow(k_f_2d[1, :, :])
        # # plt.figure()
        # # plt.imshow(k_f_2d[2, :, :])
        #
        # # here we have what is needed to calculate the complex crystal reflectivity, let's go ahead and do it here
        # beamInDotNormal = k_i_2d[0,:,:]
        # beamOutDotNormal = k_f_2d[0,:,:]
        # beamInDotHNormal = np.sum(k_i_2d * c_normal, axis=0)
        #
        # C1, C2 = np.array(self.crystal.get_amplitude(beam.photonEnergy,
        #                                              beamInDotNormal, beamOutDotNormal, beamInDotHNormal))
        #
        # # choose complex reflectivity based on polarization
        # if self.pol == 's':
        #     C = C1
        # else:
        #     C = C2
        #
        # ##### figure out crystal curvature, and find effective focal lengths
        #
        # # transverse (z-direction)
        # R_t = 1 / (2 * second_order_z * 1e-9)
        # print(R_t)
        #
        # # use equation for curved grating imaging condition. Works great!
        # f_t = np.sin(self.beta0) ** 2 / (
        #         (np.sin(self.alpha) + np.sin(self.beta0)) / R_t - np.sin(self.alpha) ** 2 / beamz)
        # print('Calculated distance to transverse focus: %.6f' % f_t)
        #
        # #### Sagittal focusing (y-direction)
        #
        # R_s = 1 / (2 * second_order_y * 1e-9)
        # print(R_s)
        # # use equation for curved grating sagittal imaging condition.
        # f_s = 1 / ((np.sin(self.alpha) + np.sin(self.beta0)) / R_s - 1 / beamz_y)
        # #
        # print('Calculated distance to sagittal focus: %.6f' % f_s)
        #
        # # limit fit to size of crystal
        # mask_z = np.abs(z_c) <= self.length/2
        # mask_y = np.abs(y_c) <= self.width/2
        #
        # # subtract second order from shape error
        # shape_subtract = np.copy(shapeError2)
        # shape_subtract[:, mask_z] -= np.tile(shapePoly_z.legval(2),(Ns,1))
        # shape_subtract[mask_y, :] -= np.tile(np.reshape(shapePoly_y.legval(2),(shapePoly_y.N,1)),(1,Ms))
        #
        # shape_subtract *= 1e-9
        #
        # p_int_z = np.zeros(3)
        # p_int_y = np.zeros(3)
        # # p_int_z[-3] = legendre_z.quad_coeff()
        # # p_int_y[-3] = legendre_y.quad_coeff()
        # # p_int_z[-2] = legendre_z.linear_coeff()
        # # p_int_y[-2] = legendre_y.linear_coeff()
        # # p_int_z[-1] = legendre_z.c[0] - legendre_z.c[2]/2
        # # p_int_y[-1] = legendre_y.c[0] - legendre_y.c[2]/2
        #
        # # high_order = (2 * np.pi / beam.lambda0 * high_order_temp *
        # #               np.sin(beta1 - self.delta))
        #
        # gamma = np.pi/2 - self.beta0 - self.alpha
        # # high_order = 2*np.pi/beam.lambda0 * shape_subtract/np.sin(self.alpha) * (np.sin(gamma) - 1)
        # # high_order = (-4 * np.pi / beam.lambda0 / np.sin(total_alpha) *
        # #  np.sin((total_alpha + self.beta0 - self.delta) / 2) ** 2 * shape_subtract)
        # high_order = -2*np.pi / beam.lambda0 * (np.sin(total_alpha) + np.sin(self.beta0 - self.delta)) * shape_subtract
        #
        # # scaling between grating z-axis and new beam coordinates
        # scale = np.sin(beta1 - self.delta)
        #
        # # change coordinate systems to get proper low-order coefficients. Multiplied by sin(beta) because integration
        # # should actually happen in beam coordinates.
        # p_scaled = Util.poly_change_coords(p_int_z, scale) * np.sin(beta1 - self.delta)
        #
        # #### might need to add back in
        # # Add 2nd order phase to p_scaled
        # p_scaled[-3] += -1 / (2 * f_t)
        # #####
        #
        # # offset from center of crystal (along crystal z-axis)
        # offset = cz - self.dx / np.tan(total_alpha)
        #
        # # scale the offset
        # offset_scaled = offset * scale
        #
        # # account for any decentering
        # p_centered = Util.recenter_coeff(p_scaled, offset_scaled)
        #
        # # 2nd order phase (factoring out pi/lambda)
        # p2nd_z = 2 * p_centered[-3]
        #
        # # 1st order phase (factoring out 2 pi/lambda)
        # # (only add any 1st order phase due to de-centering since the rest is already accounted for in delta_k).
        # p1st_z = p_centered[-2] - p_scaled[-2]
        #
        # offset_y = cy - self.dy
        # # account for any decentering
        #
        # #### might need to add back in
        # # Add 2nd order phase to p_scaled
        # p_int_y[-3] += -1 / (2 * f_s)
        # #####
        #
        # p_centered_y = Util.recenter_coeff(p_int_y, offset_y)
        #
        # # 2nd order phase (factoring out pi/lambda)
        # p2nd_y = 2 * p_centered_y[-3]
        #
        # # 1st order phase (factoring out 2 pi/lambda)
        # # (only add any 1st order phase due to de-centering since the rest is already accounted for in delta_k).
        # p1st_y = p_centered_y[-2] - p_int_y[-2]
        #
        # # figure out aperturing due to mirror's finite size
        # # z_mask = (np.abs(zi_1d - self.dx / np.tan(total_alpha)) < self.length / 2).astype(float)
        # # y_mask = (np.abs(yi_1d - self.dy) < self.width / 2).astype(float)
        #
        # z_mask = np.abs(z_c_2d) < self.length / 2
        # y_mask = np.abs(y_c_2d) < self.width / 2
        #
        # # crystal aperture
        # full_mask = z_mask * y_mask
        #
        # beam_multiplier = full_mask * np.exp(1j * high_order) * C
        # # beam_multiplier = full_mask * np.exp(1j * high_order)
        # # beam_multiplier = full_mask * C
        # # beam_multiplier = np.tile(beam_multiplier, (np.size(yi_1d), 1))
        # # if self.orientation == 1 or self.orientation == 3:
        # #     beam_multiplier = np.swapaxes(beam_multiplier, 0, 1)
        # # beam_multiplier = np.rot90(beam_multiplier, self.orientation)
        #
        # # multiply beam by aperture and phase
        # beam.wave *= beam_multiplier
        #
        # if self.orientation == 0:
        #
        #     # modify beam's wave attribute by mirror aperture and phase error
        #     # beam.wavex *= z_mask * np.exp(1j * high_order) * C
        #
        #     # take into account coordinate rescaling
        #     beam.x -= beam.cx
        #     beam.asymmetry_x(np.sin(beta1) / np.sin(total_alpha))
        #     beam.cx *= np.sin(beta1) / np.sin(total_alpha)
        #     beam.x += beam.cx
        #
        #     # add quadratic phase
        #     # beam.zx = 1 / (1 / beam.zx + p2nd)
        #     # beam.zx = 1 / p2nd
        #     new_zx = 1 / p2nd_z
        #     new_zy = 1 / p2nd_y
        #     beam.change_z(new_zx=new_zx, new_zy=new_zy)
        #
        #     # take into account mirror reflection causing beam to invert
        #     beam.x *= -1
        #     # beam.wavex = np.flipud(beam.wavex)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_azimuth=self.alpha+self.beta0)
        #     delta_ax = -beam.ax + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st_z
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ay = np.arcsin(delta_k[1]) + p1st_y
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax = np.arcsin(delta_k[0] / np.cos(total_alpha)) + p1st
        #     # beam.ay += np.arcsin(delta_k[1])
        #
        #     # adjust beam position due to mirror de-centering
        #     delta_cx = 2 * self.dx * np.cos(self.alpha)
        #     beam.cx = -beam.cx + delta_cx
        #     beam.x = beam.x + delta_cx
        #
        # elif self.orientation == 1:
        #
        #     # modify beam's wave attribute by mirror aperture and phase error
        #     # beam.wavey *= z_mask * np.exp(1j * high_order) * C
        #
        #     # take into account coordinate rescaling
        #     beam.y -= beam.cy
        #     beam.asymmetry_y(np.sin(beta1) / np.sin(total_alpha))
        #     beam.cy *= np.sin(beta1) / np.sin(total_alpha)
        #     beam.y += beam.cy
        #
        #     # add quadratic phase
        #     # beam.zy = 1 / (1 / beam.zy + p2nd)
        #     # beam.zy = 1 / p2nd
        #     new_zy = 1 / p2nd_z
        #     new_zx = 1 / p2nd_y
        #     beam.change_z(new_zx=new_zx, new_zy=new_zy)
        #
        #     # take into account mirror reflection causing beam to invert
        #     beam.y *= -1
        #     beam.wavey = np.flipud(beam.wavey)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_elevation=self.alpha + self.beta0)
        #     delta_ay = -beam.ay + np.arcsin(delta_k[0] / np.cos(self.beta0)) + p1st_z
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ax = -np.arcsin(delta_k[1]) - p1st_y
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax += -np.arcsin(delta_k[1])
        #     # beam.ay = np.arcsin(delta_k[0] / np.cos(total_alpha)) + p1st
        #
        #     # adjust beam position due to mirror de-centering
        #     delta_cy = 2 * self.dx * np.cos(self.alpha)
        #     beam.cy = -beam.cy + delta_cy
        #     beam.y = beam.y + delta_cy
        #
        # elif self.orientation == 2:
        #
        #     # modify beam's wave attribute by mirror aperture and phase error
        #     # beam.wavex *= z_mask * np.exp(1j * high_order) * C
        #
        #     # take into account coordinate rescaling
        #     beam.x -= beam.cx
        #     beam.asymmetry_x(np.sin(beta1) / np.sin(total_alpha))
        #     beam.cx *= np.sin(beta1) / np.sin(total_alpha)
        #     beam.x += beam.cx
        #
        #     # add quadratic phase
        #     # beam.zx = 1 / (1 / beam.zx + p2nd)
        #     # beam.zx = 1 / p2nd
        #     new_zx = 1 / p2nd_z
        #     new_zy = 1 / p2nd_y
        #     beam.change_z(new_zx=new_zx, new_zy=new_zy)
        #
        #     # take into account mirror reflection causing beam to invert
        #     beam.x *= -1
        #     # beam.wavex = np.flipud(beam.wavex)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_azimuth=-self.alpha - self.beta0)
        #     delta_ax = -beam.ax - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st_z
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ay = np.arcsin(delta_k[1]) + p1st_y
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st
        #     # beam.ay += -np.arcsin(delta_k[1])
        #
        #     # adjust beam position due to mirror de-centering
        #     delta_cx = -2 * self.dx * np.cos(self.alpha)
        #     beam.cx = -beam.cx + delta_cx
        #     beam.x = beam.x + delta_cx
        #
        # elif self.orientation == 3:
        #
        #     # modify beam's wave attribute by mirror aperture and phase error
        #     # beam.wavey *= z_mask * np.exp(1j * high_order) * C
        #
        #     # take into account coordinate rescaling
        #     beam.y -= beam.cy
        #     beam.asymmetry_y(np.sin(beta1) / np.sin(total_alpha))
        #     beam.cy *= np.sin(beta1) / np.sin(total_alpha)
        #     beam.y += beam.cy
        #
        #     # add quadratic phase
        #     # beam.zy = 1 / (1 / beam.zy + p2nd)
        #     # beam.zy = 1 / p2nd
        #     new_zy = 1 / p2nd_z
        #     new_zx = 1 / p2nd_y
        #     beam.change_z(new_zx=new_zx, new_zy=new_zy)
        #
        #     # take into account mirror reflection causing beam to invert
        #     beam.y *= -1
        #     # beam.wavey = np.flipud(beam.wavey)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     beam.rotate_nominal(delta_elevation=-self.alpha - self.beta0)
        #     delta_ay = -beam.ay - np.arcsin(delta_k[0] / np.cos(self.beta0)) - p1st_z
        #     # delta_ax = -2*beam.ax + np.arcsin(delta_k[0])
        #     delta_ax = np.arcsin(delta_k[1]) + p1st_y
        #     beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)
        #
        #     # adjust beam direction relative to properly aligned axis
        #     # beam.ax += np.arcsin(delta_k[1])
        #     # beam.ay = - np.arcsin(delta_k[0] / np.cos(self.alpha)) - p1st
        #
        #     # adjust beam position due to mirror de-centering
        #     delta_cy = -2 * self.dx * np.cos(self.alpha)
        #     beam.cy = -beam.cy + delta_cy
        #     beam.y = beam.y + delta_cy

        return


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
        self.df_u = dF1
        self.df_d = dF2
        self.total_alpha = self.alpha + self.delta

        # check if mirror is too long for distance to focus or source
        if self.length / 2 > np.abs(self.p):
            print('Mirror is longer than distance to source. Adjusting length to be compatible.')
            self.length = 2 * self.p * .9
        if self.length / 2 > np.abs(self.q):
            print('Mirror is longer than distance to focus. Adjusting length to be compatible.')
            self.length = 2 * self.q * .9

    def bend(self, cz):
        """
        Method to calculate polynomial coefficients due to bender influence
        :return pBend: List of floats
            Polynomial coefficients following np.polyfit order convention.
        """
        # calculate 3rd order due to benders
        p3 = (self.df_d - self.df_u) / 6 / self.length
        # calculate 2nd order due to benders
        p2 = (self.df_u + self.df_d) / 4

        pBend = [p3,p2,0,0]

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

    def calc_reflectivity(self, E0):
        """
        Method to calculate reflectivity across mirror, accounting for varying angle of incidence
        :param E0: float
            photon energy in eV
        :return z1: (N,) ndarray
            ellipse z-axis coordinates
        :return reflectivity: (N,) ndarray
            reflectivity at each z
        :return inc_angle: (N,) ndarray
            reflectivity at each z
        """
        z1, x1, z0, x0, delta = self.calc_ellipse(self.p, self.q, self.alpha)

        x1m = -np.sin(delta) * (z1 - z0) + np.cos(delta) * (x1 - x0) + x0
        #     x1m = np.cos(-delta) * (x1 - x0) + np.sin(-delta) * (y1 - y0) + x0

        x1m -= np.min(x1m)
        # calculate local incidence angle
        inc_angle = np.gradient(x1m, z1) + self.alpha

        z1 -= z0

        # plt.figure()
        # plt.plot(z1-z0,x1m)
        #
        # plt.figure()
        # plt.plot(z1-z0, inc_angle)

        reflectivity = xraydb.mirror_reflectivity(self.material, inc_angle, E0, self.density)

        # plt.figure()
        # plt.plot(z1-z0,reflectivity)

        return z1, reflectivity, inc_angle

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

    def ellipse_params(self, p, q, alpha):
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

            params = {
                'L': L,
                'a': np.sqrt(a2),
                'b': np.sqrt(b2),
                'beta': beta,
                'delta': delta,
                'x0': x0,
                'z0': z0
            }

            return params

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

            params = {
                'L': L,
                'a': np.sqrt(a2),
                'b': np.sqrt(b2),
                'beta': beta,
                'delta': delta,
                'x0': x0,
                'z0': z0
            }

            return params

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

            params = {
                'L': L,
                'a': np.sqrt(a2),
                'b': np.sqrt(b2),
                'beta': beta,
                'delta': delta,
                'x0': x0,
                'z0': z0
            }

            return params

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

    def find_intersection(self, beam):

        ux = np.reshape(np.array([1, 0, 0]), (3, 1))
        uy = np.reshape(np.array([0, 1, 0]), (3, 1))
        uz = np.reshape(np.array([0, 0, 1]), (3, 1))

        params = self.ellipse_params(self.p, self.q, self.alpha)

        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx

        # define ellipse coordinate unit vectors
        # rotation angle to rotate mirror vectors into ellipse coordinates
        # mirror is already rotated by delta when drifts are added to beamline
        ellipse_rotate = params['delta']

        re = transform.Rotation.from_rotvec(-self.sagittal * ellipse_rotate)
        Re = re.as_matrix()

        ellipse_x = np.matmul(Re, self.normal)
        ellipse_y = self.sagittal
        ellipse_z = np.matmul(Re, self.transverse)

        central_ray = np.reshape(beam.zhat, (3,1))

        coords = np.reshape(beam_center, (3,1))

        coords -= np.reshape(mirror_center, (3, 1))

        coords += np.reshape(ellipse_x * params['x0'] + ellipse_z * params['z0'], (3, 1))

        # now write beam coordinates in ellipse coordinates
        transform_matrix = np.tensordot(np.reshape([ellipse_x, ellipse_y, ellipse_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_ellipse = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # now write rays in ellipse coordinates
        rays_ellipse = np.tensordot(transform_matrix, central_ray, axes=(1, 0))

        a = params['a']
        b = params['b']

        if self.q>=0:
            aq = b ** 2 / a ** 2 + (rays_ellipse[0, :] / rays_ellipse[2, :]) ** 2
            bq = (-2 * coords_ellipse[2, :] * (rays_ellipse[0, :] / rays_ellipse[2, :]) ** 2 +
                  2 * coords_ellipse[0, :] * rays_ellipse[0, :] / rays_ellipse[2, :])
            cq = (coords_ellipse[2, :] ** 2 * (rays_ellipse[0, :] / rays_ellipse[2, :]) ** 2 -
                  2 * coords_ellipse[0, :] * coords_ellipse[2, :] * rays_ellipse[0, :] / rays_ellipse[2, :] +
                  coords_ellipse[0, :] ** 2 - b ** 2)
        else:
            aq = -b ** 2 / a **2 + (rays_ellipse[0,:] / rays_ellipse[2,:]) ** 2
            bq = (-2 * coords_ellipse[2,:] * (rays_ellipse[0,:]/rays_ellipse[2,:]) ** 2 +
                  2 * coords_ellipse[0,:] * rays_ellipse[0,:]/rays_ellipse[2,:])
            cq = ((coords_ellipse[2,:] * rays_ellipse[0,:]/rays_ellipse[2,:])**2 -
                  2 * coords_ellipse[0,:] * coords_ellipse[2,:] * rays_ellipse[0,:]/rays_ellipse[2,:] +
                  coords_ellipse[0,:]**2 + b ** 2)

        z_intersect = (-bq + np.sqrt(bq ** 2 - 4 * aq * cq)) / 2 / aq
        if self.q>=0:
            x_intersect = -b * np.sqrt(np.ones_like(z_intersect) - z_intersect ** 2 / a ** 2)
        else:
            x_intersect = b * np.sqrt(z_intersect**2/a**2 - np.ones_like(z_intersect))
        y_intersect = rays_ellipse[1, :] / rays_ellipse[2, :] * (z_intersect - coords_ellipse[2, :]) + coords_ellipse[1,:]

        intersect_point = np.reshape(np.array([x_intersect,y_intersect,z_intersect]), (3,1))

        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        intersect_global = np.tensordot(inv_transform, intersect_point, axes=(1, 0))

        # subtract ellipse center, so that now this is relative to the mirror center
        intersect_global -= np.reshape(ellipse_x * params['x0'] + ellipse_z * params['z0'], (3, 1))

        intersect_global += np.reshape(mirror_center, (3, 1))

        return intersect_global

    def trace_surface(self, beam):

        figon = self.show_figures
        # global unit vectors
        ux = np.reshape(np.array([1,0,0]),(3,1))
        uy = np.reshape(np.array([0,1,0]),(3,1))
        uz = np.reshape(np.array([0,0,1]),(3,1))

        delta_z = self.length / 2 * 1.1

        print('ax: %.6e' % beam.ax)
        print('ay: %.6e' % beam.ay)

        # propagate beam to just upstream of mirror
        beam.beam_prop(-delta_z)
        # define mirror surface (this is in the normal ellipse coordinates)
        params = self.ellipse_params(self.p, self.q, self.alpha)

        # vector defining displacement from beam location to mirror center. This is in global coordinates
        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx
        beam_to_mirror = beam_center - mirror_center

        # define ellipse coordinate unit vectors
        # rotation angle to rotate mirror vectors into ellipse coordinates
        # mirror is already rotated by delta when drifts are added to beamline. Also will
        # rotate mirror by roll when drifts are added to beamline.
        ellipse_rotate = params['delta']

        re = transform.Rotation.from_rotvec(-self.sagittal*ellipse_rotate)
        Re = re.as_matrix()

        # unit vectors for ellipse coordinates, written in global coordinates
        ellipse_x = np.matmul(Re, self.normal)
        ellipse_y = self.sagittal
        ellipse_z = np.matmul(Re, self.transverse)

        print('ellipse unit vectors')
        print(self.normal)
        print(ellipse_x)
        print(ellipse_z)

        # define beam rays up to second order (assume that linear term is not needed and is
        # already captured in the k-vector
        rays_x = beam.x/beam.zx
        rays_y = beam.y/beam.zy

        # beam coordinates in global coordinates
        coords = np.multiply.outer(beam.xhat, beam.x) + np.multiply.outer(beam.yhat, beam.y)

        # reference to global origin by adding beam global center
        coords += np.reshape(beam_center, (3, 1, 1))

        # now subtract mirror center so that beam coordinates are in global coordinates,
        # but with origin at mirror center
        coords -= np.reshape(mirror_center, (3, 1, 1))
        # now shift origin to ellipse origin. This should be general since the unit vectors
        # are defined based on mirror unit vectors
        coords += np.reshape(ellipse_x * params['x0'] + ellipse_z * params['z0'], (3, 1, 1))

        print('x0 and z0')
        print(params['x0'])
        print(params['z0'])

        # now write beam coordinates in ellipse coordinates
        # (transforming from global coordinates to ellipse coordinates)
        transform_matrix = np.tensordot(np.reshape([ellipse_x, ellipse_y, ellipse_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_ellipse = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # mirror vectors written in ellipse coordinates
        mirror_x_ellipse = np.tensordot(transform_matrix, np.reshape(self.normal,(3,1,1)),axes=(1,0))
        mirror_y_ellipse = np.tensordot(transform_matrix, np.reshape(self.sagittal,(3,1,1)),axes=(1,0))
        mirror_z_ellipse = np.tensordot(transform_matrix, np.reshape(self.transverse,(3,1,1)),axes=(1,0))

        # calculate z component of rays (enforcing unit vector)
        rays_z = np.sqrt(np.ones_like(rays_x) - rays_x ** 2 - rays_y ** 2)
        # ray vectors at each point in the beam
        rays = (np.multiply.outer(beam.xhat, rays_x) + np.multiply.outer(beam.zhat, rays_z)
                + np.multiply.outer(beam.yhat, rays_y))

        # normalize rays (should be redundant)
        rays = rays / np.sqrt(np.sum(rays*rays, axis=0))

        # now write rays in ellipse coordinates
        rays_ellipse = np.tensordot(transform_matrix, rays, axes=(1,0))

        # calculate ellipse for plotting purposes
        z1, x1, z0, x0, delta = self.calc_ellipse(self.p, self.q, self.alpha)

        if figon:
            plt.figure()
            plt.plot(coords_ellipse[2,:,0],coords_ellipse[0,:,0])
            plt.plot(z1, x1)
            plt.quiver(coords_ellipse[2,:,0],coords_ellipse[0,:,0],rays_ellipse[2,:,0],rays_ellipse[0,:,0])
            plt.ylim(-.5,.5)
            plt.grid()
            plt.title('incoming rays and mirror')

        # solve quadratic eqn for ellipse/line intersection
        a = params['a']
        b = params['b']
        aq = b**2/a**2 + (rays_ellipse[0,:,:]/rays_ellipse[2,:,:])**2
        bq = (-2*coords_ellipse[2,:,:]*(rays_ellipse[0,:,:]/rays_ellipse[2,:,:])**2 +
              2*coords_ellipse[0,:,:]*rays_ellipse[0,:,:]/rays_ellipse[2,:,:])
        cq = (coords_ellipse[2,:,:]**2*(rays_ellipse[0,:,:]/rays_ellipse[2,:,:])**2-
              2*coords_ellipse[0,:,:]*coords_ellipse[2,:,:]*rays_ellipse[0,:,:]/rays_ellipse[2,:,:]+
              coords_ellipse[0,:,:]**2-b**2)

        # quadratic equation
        z_intersect = (-bq+np.sqrt(bq**2-4*aq*cq))/2/aq

        # find x and y based on z
        x_intersect = -b*np.sqrt(np.ones_like(z_intersect)-z_intersect**2/a**2)
        y_intersect = (rays_ellipse[1,:,:]/rays_ellipse[2,:,:]*(z_intersect-coords_ellipse[2,:,:])
                       + coords_ellipse[1,:,:])

        Ni, Mi = np.shape(z_intersect)

        # put mirror intersection coordinates in a 3xNxM array
        intersect_coords = np.zeros((3,Ni,Mi))
        intersect_coords[0,:,:] = x_intersect
        intersect_coords[1,:,:] = y_intersect
        intersect_coords[2,:,:] = z_intersect

        # vectors pointing from beam location to mirror intersection
        i_vector = intersect_coords - coords_ellipse

        # length of each vector
        distance_1 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        # define ellipse normals along mirror surface
        ellipse_normal = np.zeros_like(rays)
        ellipse_normal[2,:,:] = -b/a**2*z_intersect*(1-z_intersect**2/a**2)**(-.5)
        ellipse_normal[0,:,:] = np.ones_like(z_intersect)

        # normalize ellipse normals
        ellipse_normal = ellipse_normal/np.sqrt(np.sum(ellipse_normal*ellipse_normal,axis=0))

        # calculate ray direction after interaction with ellipse using law of reflection
        rays_out = rays_ellipse - 2 * np.sum(rays_ellipse*ellipse_normal,axis=0) * ellipse_normal

        # now find intersection with exit plane
        # we can define this simply as having a normal vector in the direction of the central ray
        # and we will define the plane to be a distance length/2*1.1 from the intersection point of the central ray
        plane_normal = np.reshape(rays_out[:,int(beam.N/2), int(beam.M/2)],(3,1,1))
        central_point = np.reshape(intersect_coords[:,int(beam.N/2),int(beam.M/2)],(3,1,1)) + plane_normal*self.length/2*1.1

        # find z intersection with this plane
        d2 = np.sum((central_point - intersect_coords)*plane_normal,axis=0)/np.sum(rays_out*plane_normal,axis=0)
        plane_intersect = intersect_coords + rays_out*d2
        i_vector = plane_intersect - intersect_coords
        distance_2 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        if figon:
            plt.figure()
            plt.plot(coords_ellipse[2, :, 0], coords_ellipse[0, :, 0])
            plt.plot(z1, x1)
            plt.plot(z_intersect, x_intersect)
            plt.plot(plane_intersect[2,:, 0],plane_intersect[0,:, 0])
            # plt.ylim(-.5, .5)
            plt.grid()
            plt.title('entrance/exit planes, mirror intersection')

        # total distance for each beam ray
        total_distance = (distance_1+distance_2)
        #
        if figon:
            plt.figure()
            plt.plot(intersect_coords[2,:,0],distance_1)
            plt.plot(intersect_coords[2,:,0],distance_2)
            plt.plot(intersect_coords[2,:,0],distance_1+distance_2)
            plt.title('distances')

        # find location of central ray in exit plane
        origin = np.reshape(plane_intersect[:,int(beam.N/2),int(beam.M/2)],(3,1,1))

        # put beam center at origin
        shifted_plane = plane_intersect-origin

        # get final k-vector for central ray
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        # convert to global coordinates
        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))
        k_f_global = k_f_global[:,0,0]

        # first rotate by the "nominal" amount
        if self.orientation==0:
            beam.rotate_nominal(delta_azimuth=2*self.alpha)
        elif self.orientation==1:
            beam.rotate_nominal(delta_elevation=2*self.alpha)
        elif self.orientation==2:
            beam.rotate_nominal(delta_azimuth=-2*self.alpha)
        elif self.orientation==3:
            beam.rotate_nominal(delta_elevation=-2*self.alpha)

        # get initial k-vector for central ray in global coordinates
        k_i = np.copy(beam.zhat)

        # find the change in the k-vector in global coordinates
        delta_k = k_f_global - k_i

        print('xhat: {}'.format(beam.xhat))
        print('yhat: {}'.format(beam.yhat))
        print('zhat: {}'.format(beam.zhat))
        print('dk: {}'.format(delta_k))

        # now make minor adjustment to k-vector based on central ray at exit plane
        # might want to do one axis at a time or change the order. Or could change the rotation
        # to rotate about the "unrotated" axes.
        delta_ax = np.arcsin(delta_k[0])
        x_sign = np.sign(np.dot(np.cross(k_i,k_f_global),beam.yhat))
        delta_ay = -np.arcsin(delta_k[1])
        y_sign = np.sign(-np.dot(np.cross(k_i,k_f_global),beam.xhat))
        beam.rotate_beam(delta_ax=x_sign*np.abs(delta_ax), delta_ay=y_sign*np.abs(delta_ay))

        # check for consistency
        print('is beam in the correct direction?')
        # print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global)))

        # mask defining mirror acceptance
        if self.q>=0:
            mask = np.logical_and(coords_ellipse[0,:,:]>intersect_coords[0,:,:],
                                  plane_intersect[0,:,:]>intersect_coords[0,:,:])
        else:
            mask = np.logical_and(coords_ellipse[0, :,:] < intersect_coords[0, :,:],
                                  plane_intersect[0, :,:] > intersect_coords[0, :,:])

        # second order fit to ray distance
        if self.q>=0:
            mask = np.logical_and(mask, coords_ellipse[0,:,:]<0)
        else:
            mask = np.logical_and(mask, coords_ellipse[0, :,:] > 0)

        # can define mirror acceptance by taking dot product with mirror unit vectors (in ellipse coordinate system).
        # distance along length axis
        mirror_center_ellipse = np.reshape(np.array([x0,0,z0]),(3,1,1))
        d_length = np.sum((intersect_coords - mirror_center_ellipse)*np.reshape(mirror_z_ellipse,(3,1,1)),axis=0)
        # distance along width axis
        d_width = np.sum((intersect_coords - mirror_center_ellipse)*np.reshape(mirror_y_ellipse,(3,1,1)),axis=0)

        # d_length and d_width are equivalent to the mirror coordinates, these can be used for interpolating
        # the shape error
        # mirror shape error interpolation onto beam coordinates (if applicable)

        shapeInterp = np.zeros((beam.N,beam.M))

        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                max_ys = self.width / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                ys = np.linspace(-1,1,100)*max_ys

                print(np.shape(np.tile(self.shapeError,(100,1))))
                f = RectBivariateSpline(ys,zs,np.tile(self.shapeError,(100,1)))
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N, beam.M))*1e-9

                plt.figure()
                plt.imshow(shapeInterp)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = RectBivariateSpline(ys,zs,self.shapeError)
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N,beam.M))*1e-9
                # f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                # shapeError2 = f(zi_1d - self.dx / np.tan(self.total_alpha), yi_1d - self.dy)

        # mask based on mirror length
        mask = np.logical_and(mask, np.abs(d_length)<self.length/2)
        # mask based on mirror width
        mask = np.logical_and(mask, np.abs(d_width)<self.width/2)

        # now write new beam coordinates in local beam coordinate system
        # (transforming from ellipse coordinates to local beam coordinates)
        transform_matrix2 = np.tensordot(np.reshape([beam.xhat,beam.yhat,beam.zhat], (3, 3)),
                                        np.reshape([ellipse_x, ellipse_y, ellipse_z], (3, 3)), axes=(1, 1))
        shifted_plane2 = np.tensordot(transform_matrix2, shifted_plane, axes=(1, 0))

        # beam's x and y coordinates in the exit beam local coordinate system
        # note these are not in general on a regularly spaced grid but are based on tracking
        # rays through the course of the reflection.
        x_eff = shifted_plane2[0,:,:]
        y_eff = shifted_plane2[1,:,:]

        # calculate desired pixel size due to expected change in beam size, and
        # define new coordinate grids to interpolate onto at beam exit plane.
        if self.orientation == 0 or self.orientation == 2:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx * (self.q - self.length / 2 * 1.1) / self.q
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)
        else:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy * (self.q - self.length / 2 * 1.1) / self.q
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)

        # 1D masks based on mirror acceptance, going through beam center.
        max_intensity = np.max(np.abs(beam.wave)**2)
        # mask2 = np.logical_and(mask,np.abs(beam.wave)**2>0.1*max_intensity)
        mask_x = mask[int(beam.N/2),:]>0
        mask_y = mask[:,int(beam.M/2)]>0
        # introducing weights for the polynomial fits based on beam intensity

        weight_x = (np.abs(beam.wave[int(beam.N/2),:])**2>0.1*max_intensity).astype(float)
        weight_y = (np.abs(beam.wave[:,int(beam.M/2)])**2>0.1*max_intensity).astype(float)

        # plt.figure()
        # plt.imshow((total_distance-np.mean(total_distance[mask]))*mask)
        #
        # plt.figure()
        # plt.imshow(np.abs(beam.wave)**2>0.1*max_intensity)

        incidence_angle = np.abs(np.arccos(np.sum(rays_out * ellipse_normal, axis=0)))
        glancing = np.pi / 2 - incidence_angle

        reflectivity = xraydb.mirror_reflectivity(self.material, glancing, beam.photonEnergy, self.density)

        # plt.figure()
        # plt.imshow((glancing-np.mean(glancing[mask]))*mask)

        # subtract best fit parabola in x-direction and best fit line in y direction
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff)# + np.polyval(p_coeff_y[:-2],y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff)# + np.polyval(p_coeff_y[:-2], y_eff)
        #
        # TRY THIS OUT LATER
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff) + np.polyval(p_coeff_y,y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff) + np.polyval(p_coeff_x, x_eff)

        points = np.zeros((np.size(x_eff[mask]),2))
        points[:,0] = x_eff[mask]
        points[:,1] = y_eff[mask]
        # xi = np.zeros((np.size(x_out),2))
        # xi[:,0] = x_out
        # xi[:,1] = y_out
        xi_0, xi_1 = np.meshgrid(x_out, y_out)
        print('attempting interpolation')
        mask2 = interpolation.griddata(points, mask[mask], (xi_0, xi_1), method='nearest',fill_value=0)
        # mask2 = fmask(x_out,y_out)
        mask2[mask2<.9] = 0
        # mask2 = mask2.astype(int)
        mask2 = mask2 > 0.5

        # interpolate intensity onto new exit plane grid
        abs_out = interpolation.griddata(points, np.abs(beam.wave[mask]), (xi_0, xi_1), fill_value=0)

        # unwrap phase of beam at input
        angle_in = unwrap_phase(np.angle(beam.wave))

        # add quadratic phase if beam is not focused since this also needs to be interpolated
        quadratic = np.zeros_like(beam.x)

        if not beam.focused_x:
            quadratic += np.pi / beam.lambda0 / beam.zx * (beam.x) ** 2
        if not beam.focused_y:
            quadratic += np.pi / beam.lambda0 / beam.zy * (beam.y) ** 2

        # add the quadratic phase to the input phase
        angle_in += quadratic

        # add phase contribution from deviations in the distance traveled by each ray
        total_phase = (angle_in + 2 * np.pi / beam.lambda0 * total_distance
                       - shapeInterp * 4*np.pi*np.sin(self.alpha) / beam.lambda0)

        # get polynomial fits based on new coordinates
        p_coeff_x = np.polyfit(x_eff[int(beam.N/2),:][mask_x], total_phase[int(beam.N/2),:][mask_x], 2,
                               w=weight_x[mask_x])

        p_coeff_y = np.polyfit(y_eff[:,int(beam.M/2)][mask_y], total_phase[:,int(beam.M/2)][mask_y], 2,
                               w=weight_y[mask_y])

        # calculate effective distance to focus based on total phase
        z_2 = np.pi / beam.lambda0 / p_coeff_x[-3]
        z_2_y = np.pi / beam.lambda0 / p_coeff_y[-3]

        # add this to what was already subtracted (might be able to do this in one step)
        # z_total_x = 1 / (1 / z_out_x + 1 / z_2)
        # z_total_y = 1 / (1 / z_out_y + 1 / z_2_y)
        z_total_x = z_2
        z_total_y = z_2_y
        print('new z: %.6f' % z_total_x)
        print('new z y: %.6f' % z_total_y)

        # calculate residual linear phase terms
        # linear += p_coeff_x[-2] * beam.lambda0/2/np.pi
        # linear_y += p_coeff_y[-2] * beam.lambda0/2/np.pi
        linear = p_coeff_x[-2] * beam.lambda0 / 2 / np.pi
        linear_y = p_coeff_y[-2] * beam.lambda0/2/np.pi

        print(linear)
        print(linear_y)

        # subtract linear terms since this should be taken care of based on central ray direction
        # Now this is being done in both tangential and sagittal directions, and we compensate with
        # a change in beam k-vector direction.
        total_phase -= np.polyval(p_coeff_x[-2:], x_eff) + np.polyval(p_coeff_y[-2:], y_eff)

        # sutbtract off quadratic phase term if the beam is not focused
        if not beam.focused_x:
            total_phase -= np.polyval([p_coeff_x[-3], 0, 0], x_eff)
        if not beam.focused_y:
            total_phase -= np.polyval([p_coeff_y[-3], 0, 0], y_eff)

        # interpolate the phase onto the exit plane grid
        points = np.zeros((np.size(x_eff), 2))
        points[:, 0] = x_eff.flatten()
        points[:, 1] = y_eff.flatten()
        phase_interp = interpolation.griddata(points, total_phase.flatten(), (xi_0, xi_1), fill_value=0)

        # interpolate the reflectivity onto the exit plane grid
        reflectivity_interp = interpolation.griddata(points, reflectivity.flatten(), (xi_0, xi_1), fill_value=0)

        # update beam complex amplitude using what has been interpolated onto the new grid
        beam.wave = abs_out * np.exp(1j * phase_interp)
        # multiply by reflectivity if this is being used
        if self.use_reflectivity:
            beam.wave *= np.sqrt(reflectivity_interp)
        # multiply by mirror aperture that has been interpolated onto the exit plane grid
        beam.wave *= mask2

        if figon:
            plt.figure()
            plt.imshow(np.abs(beam.wave))

        ax0 = np.copy(beam.ax)
        ay0 = np.copy(beam.ay)

        # figure out where the beam is in global coordinates
        # change in angle
        k_i = rays_ellipse[:, int(beam.N / 2), int(beam.M / 2)]
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))

        # compensate for removing linear phase by adjusting beam k-vector
        beam.rotate_beam(delta_ax=linear)
        beam.rotate_beam(delta_ay=linear_y)

        print(beam.ax)
        delta_cx = ax0 * self.length / 2 * 1.1
        delta_cx += beam.ax * self.length / 2 * 1.1
        delta_cx += 2 * np.dot(self.normal, beam.xhat) * self.dx
        print('change in beam center')
        print(delta_cx)
        # beam.cx = -beam.cx + delta_cx
        # print(beam.cx)

        x_out, y_out = np.meshgrid(x_out, y_out)
        beam.x = x_out
        beam.y = y_out

        beam.new_fx()

        print('is beam in the correct direction?')
        print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global[:, 0, 0])))
        print(params['beta'])
        print(k_f)
        print(k_f_global)

        # now figure out global coordinates
        # get back into global coordinates using inverse of transformation matrix, just looking at central ray
        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        origin_global = np.tensordot(inv_transform, origin, axes=(1,0))

        # subtract ellipse center, so that now this is relative to the mirror center
        origin_global -= np.reshape(ellipse_x * params['x0'] + ellipse_z * params['z0'], (3, 1, 1))

        # now add the mirror center in global coordinates, so that this should be the beam location
        # in global coordinates
        origin_global += np.reshape(mirror_center, (3, 1, 1))
        # origin_global -= np.reshape(self.normal*dx,(3,1))
        print(origin_global)
        # now shift origin to ellipse origin

        beam.global_x = origin_global[0,0,0]
        beam.global_y = origin_global[1,0,0]
        beam.global_z = origin_global[2,0,0]

        # account for coordinate scaling and/or changes to Rayleigh range,
        # whether beam is classified as focused or not.
        beam.change_z(new_zx=z_total_x,new_zy=z_total_y)

        beam.new_fx()
        print('global_x: %.2f' % beam.global_x)
        print('global_y: %.2f' % beam.global_y)
        print('global_z: %.2f' % beam.global_z)

        print(np.shape(beam.x))
        print(np.shape(beam.y))

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
            zi_1d = zi[0, :]
            cz = beam.cx / np.sin(self.total_alpha)
            yi = beam.y
            yi_1d = yi[:, 0]
            cy = beam.cy
            beamz = beam.zx

        elif self.orientation == 1:

            # small change to total angle of incidence
            self.total_alpha += -beam.ay

            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(self.total_alpha)
            zi_1d = zi[:, 0]
            cz = beam.cy / np.sin(self.total_alpha)
            yi = -beam.x
            yi_1d = yi[0, :]
            cy = -beam.cx
            beamz = beam.zy

        elif self.orientation == 2:

            # small change to total angle of incidence
            self.total_alpha += beam.ax

            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(self.total_alpha)
            zi_1d = zi[0, :]
            cz = -beam.cx / np.sin(self.total_alpha)
            yi = -beam.y
            yi_1d = yi[:, 0]
            cy = -beam.cy
            beamz = beam.zx

        elif self.orientation == 3:

            # small change to total angle of incidence
            self.total_alpha += beam.ay

            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(self.total_alpha)
            zi_1d = zi[:, 0]
            cz = -beam.cy / np.sin(self.total_alpha)
            yi = beam.x
            yi_1d = yi[0, :]
            cy = beam.cx
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
                central_line = np.interp(zi_1d - self.dx / np.tan(self.total_alpha), zs, self.shapeError)
                # tile onto mirror short axis direction
                shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                shapeError2 = f(zi_1d - self.dx / np.tan(self.total_alpha), yi_1d - self.dy)

        # figure out aperturing due to mirror's finite size
        z_mask = (np.abs(zi - self.dx / np.tan(self.total_alpha)) < self.length / 2).astype(float)
        y_mask = (np.abs(yi - self.dy) < self.width / 2).astype(float)

        mirror = z_mask * y_mask
        # self.misalign = self.delta * self.beamx / self.alpha

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
        # p_scaled[-3] += -1 / (2 * self.p) - 1 / (2 * self.q)

        p_scaled[-3] += (-1 / (2 * (beamz))
                         - 1 / (2 * (self.q - cz * np.cos(self.total_alpha))))

        # account for decentering
        p_scaled = Util.recenter_coeff(p_scaled, offset_scaled)

        # now add in normal focusing contribution to phase
        # factor out pi/lambda for quadratic term (so equal to 1/z)
        quadratic += 2 * p_scaled[-3]

        # factor out 2pi/lambda for linear term (so equal to change in propagation angle)
        linear += p_scaled[-2]

        self.trace_surface(beam)
        beam.beam_prop(-self.length / 2 * 1.1)

        print('x')
        print(beam.global_x)
        print('y')
        print(beam.global_y)
        print('z')
        print(beam.global_z)

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
        self.m2.dx = self.m2.dx - .68 * (self.delta_mirror + delta_mirror) - .006 * (
                np.cos(self.delta_mirror + delta_mirror) - 1)
        # pre-mirror angle adjustment
        self.m2.delta = self.delta_mirror + delta_mirror
        # grating angle of incidence
        self.grating.alpha = alpha0 + self.delta - self.delta_mirror * 2
        # grating diffraction angle
        self.grating.beta0 = beta0 - self.delta

        # set monochromator z-position to pre-mirror z-position
        self.z = self.m2.z

        self.elevation = 0
        self.azimuth = 0

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
        self.m2.propagate(beam)
        # self.YAG.propagate(beam)
        # propagate from pre-mirror to grating
        beam.beam_prop(self.grating.z - self.m2.z)

        # calculate profile on monochromator YAG
        self.yag.propagate(beam)
        # adjust beam angle to prepare for grating orientation
        beam.ay -= 2 * self.m2.delta
        # propagate beam through grating
        self.grating.propagate(beam)


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
        :param lambda0: float
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

        return delta_k, k_f

    def propagate(self, beam):
        """
        Method that overrides Mirror class propagation
        :param beam: Beam
            Beam object to diffract from grating. Modified by this function.
        :return: None
        """
        # if we're operating in zero order, just acts like a mirror
        # if self.order == 0:
        #     self.reflect(beam)
        # # if we're in first order, calculate diffraction
        # elif self.order == 1:
        #     self.diffract(beam)

        self.trace_surface(beam)
        beam.beam_prop(-self.length / 2 * 1.1)

    def trace_surface(self, beam):

        figon = self.show_figures
        # global unit vectors
        ux = np.reshape(np.array([1,0,0]),(3,1))
        uy = np.reshape(np.array([0,1,0]),(3,1))
        uz = np.reshape(np.array([0,0,1]),(3,1))

        delta_z = self.length / 2 * 1.1

        print('ax: %.6e' % beam.ax)
        print('ay: %.6e' % beam.ay)

        # propagate beam to just upstream of mirror
        beam.beam_prop(-delta_z)

        # vector defining displacement from beam location to mirror center. This is in global coordinates
        beam_center = np.array([beam.global_x, beam.global_y, beam.global_z])
        mirror_center = np.array([self.global_x, self.global_y, self.z]) + self.normal * self.dx
        beam_to_mirror = beam_center - mirror_center

        # unit vectors for ellipse coordinates, written in global coordinates
        mirror_x = self.normal
        mirror_y = self.sagittal
        mirror_z = self.transverse

        # define beam rays up to second order (assume that linear term is not needed and is
        # already captured in the k-vector
        rays_x = beam.x/beam.zx
        rays_y = beam.y/beam.zy

        # beam coordinates in global coordinates
        coords = np.multiply.outer(beam.xhat, beam.x) + np.multiply.outer(beam.yhat, beam.y)

        # reference to global origin by adding beam global center
        coords += np.reshape(beam_center, (3, 1, 1))

        # now subtract mirror center so that beam coordinates are in global coordinates,
        # but with origin at mirror center
        coords -= np.reshape(mirror_center, (3, 1, 1))

        # now write beam coordinates in mirror plane coordinates
        # (transforming from global coordinates to mirror coordinates)
        transform_matrix = np.tensordot(np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)),
                                        np.reshape([ux, uy, uz], (3, 3)), axes=(1, 1))
        coords_mirror = np.tensordot(transform_matrix, coords, axes=(1, 0))

        # mirror vectors written in mirror coordinates
        mirror_x_local = ux
        mirror_y_local = uy
        mirror_z_local = uz

        # calculate z component of rays (enforcing unit vector)
        rays_z = np.sqrt(np.ones_like(rays_x) - rays_x ** 2 - rays_y ** 2)
        # ray vectors at each point in the beam
        rays = (np.multiply.outer(beam.xhat, rays_x) + np.multiply.outer(beam.zhat, rays_z)
                + np.multiply.outer(beam.yhat, rays_y))

        # normalize rays (should be redundant)
        rays = rays / np.sqrt(np.sum(rays*rays, axis=0))

        # now write rays in mirror coordinates
        rays_mirror = np.tensordot(transform_matrix, rays, axes=(1,0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2,:,0],coords_mirror[0,:,0])
            plt.quiver(coords_mirror[2,:,0],coords_mirror[0,:,0],rays_mirror[2,:,0],rays_mirror[0,:,0])
            plt.ylim(-.5,.5)
            plt.grid()
            plt.title('incoming rays and mirror')

        z_intersect = coords_mirror[2, :, :] - rays_mirror[2, :, :] / rays_mirror[0, :, :] * coords_mirror[0, :, :]
        # by definition the x coordinate is zero for intersection
        Ni, Mi = np.shape(z_intersect)
        x_intersect = np.zeros_like(z_intersect)
        y_intersect = (rays_mirror[1, :, :] / rays_mirror[2, :, :]
                       * (z_intersect - coords_mirror[2, :, :]) + coords_mirror[1, :, :])

        # put mirror intersection coordinates in a 3xNxM array
        intersect_coords = np.zeros((3,Ni,Mi))
        intersect_coords[0,:,:] = x_intersect
        intersect_coords[1,:,:] = y_intersect
        intersect_coords[2,:,:] = z_intersect

        # vectors pointing from beam location to mirror intersection
        i_vector = intersect_coords - coords_mirror

        # length of each vector
        distance_1 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        # define normals along mirror surface
        mirror_normal = np.zeros_like(rays)
        mirror_normal[0,:,:] = np.ones_like(z_intersect)

        d_length = np.sum((intersect_coords) * np.reshape(mirror_z_local, (3, 1, 1)), axis=0)
        # distance along width axis
        d_width = np.sum((intersect_coords) * np.reshape(mirror_y_local, (3, 1, 1)), axis=0)

        # calculate grating effect
        # g_parallel = np.sum(c_normal * mirror_z_local, axis=0) * beam.lambda0 / self.d
        g_parallel = self.order*beam.lambda0 * (self.n0 + self.n1 * d_length + self.n2 * d_length ** 2)
        rays_out = np.zeros_like(rays_mirror)
        rays_out[1,:,:] = rays_mirror[1,:,:]
        rays_out[2,:,:] = rays_mirror[2,:,:] + g_parallel
        rays_out[0,:,:] = np.sqrt(np.ones_like(rays_out[0,:,:]) - rays_out[1,:,:]**2 - rays_out[2,:,:]**2)

        # calculate ray direction after interaction with ellipse using law of reflection
        # rays_out = rays_mirror - 2 * np.sum(rays_mirror*mirror_normal,axis=0) * mirror_normal

        # now find intersection with exit plane
        # we can define this simply as having a normal vector in the direction of the central ray
        # and we will define the plane to be a distance length/2*1.1 from the intersection point of the central ray
        plane_normal = np.reshape(rays_out[:,int(beam.N/2), int(beam.M/2)],(3,1,1))
        central_point = np.reshape(intersect_coords[:,int(beam.N/2),int(beam.M/2)],(3,1,1)) + plane_normal*self.length/2*1.1

        # find z intersection with this plane
        d2 = np.sum((central_point - intersect_coords)*plane_normal,axis=0)/np.sum(rays_out*plane_normal,axis=0)
        plane_intersect = intersect_coords + rays_out*d2
        i_vector = plane_intersect - intersect_coords
        distance_2 = np.sqrt(np.sum(i_vector*i_vector,axis=0))

        if figon:
            plt.figure()
            plt.plot(coords_mirror[2, :, 0], coords_mirror[0, :, 0])
            plt.plot(z_intersect, x_intersect)
            plt.plot(plane_intersect[2,:, 0],plane_intersect[0,:, 0])
            # plt.ylim(-.5, .5)
            plt.grid()
            plt.title('entrance/exit planes, mirror intersection')

        # total distance for each beam ray
        total_distance = (distance_1+distance_2)
        #
        if figon:
            plt.figure()
            plt.plot(intersect_coords[2,:,0],distance_1)
            plt.plot(intersect_coords[2,:,0],distance_2)
            plt.plot(intersect_coords[2,:,0],distance_1+distance_2)
            plt.title('distances')

        # find location of central ray in exit plane
        origin = np.reshape(plane_intersect[:,int(beam.N/2),int(beam.M/2)],(3,1,1))

        # put beam center at origin
        shifted_plane = plane_intersect-origin

        # get final k-vector for central ray
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        # convert to global coordinates
        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))
        k_f_global = k_f_global[:,0,0]

        # first rotate by the "nominal" amount
        if self.orientation==0:
            beam.rotate_nominal(delta_azimuth=self.alpha+self.beta0)
        elif self.orientation==1:
            beam.rotate_nominal(delta_elevation=self.alpha+self.beta0)
        elif self.orientation==2:
            beam.rotate_nominal(delta_azimuth=-self.alpha-self.beta0)
        elif self.orientation==3:
            beam.rotate_nominal(delta_elevation=-self.alpha-self.beta0)

        # get initial k-vector for central ray in global coordinates
        k_i = np.copy(beam.zhat)

        # find the change in the k-vector in global coordinates
        delta_k = k_f_global - k_i

        print('xhat: {}'.format(beam.xhat))
        print('yhat: {}'.format(beam.yhat))
        print('zhat: {}'.format(beam.zhat))
        print('dk: {}'.format(delta_k))

        # now make minor adjustment to k-vector based on central ray at exit plane
        # might want to do one axis at a time or change the order. Or could change the rotation
        # to rotate about the "unrotated" axes.
        delta_ax = np.arcsin(delta_k[0])
        x_sign = np.sign(np.dot(np.cross(k_i,k_f_global),beam.yhat))
        delta_ay = -np.arcsin(delta_k[1])
        y_sign = np.sign(-np.dot(np.cross(k_i,k_f_global),beam.xhat))
        beam.rotate_beam(delta_ax=x_sign*np.abs(delta_ax), delta_ay=y_sign*np.abs(delta_ay))

        # check for consistency
        print('is beam in the correct direction?')
        # print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global)))

        # mask defining mirror acceptance
        # if self.q>=0:
        #     mask = np.logical_and(coords_ellipse[0,:,:]>intersect_coords[0,:,:],
        #                           plane_intersect[0,:,:]>intersect_coords[0,:,:])
        # else:
        #     mask = np.logical_and(coords_ellipse[0, :,:] < intersect_coords[0, :,:],
        #                           plane_intersect[0, :,:] > intersect_coords[0, :,:])
        #
        # # second order fit to ray distance
        # if self.q>=0:
        #     mask = np.logical_and(mask, coords_ellipse[0,:,:]<0)
        # else:
        #     mask = np.logical_and(mask, coords_ellipse[0, :,:] > 0)

        # can define mirror acceptance by taking dot product with mirror unit vectors (in ellipse coordinate system).
        # distance along length axis


        mask = np.ones_like(d_length)

        # d_length and d_width are equivalent to the mirror coordinates, these can be used for interpolating
        # the shape error
        # mirror shape error interpolation onto beam coordinates (if applicable)

        shapeInterp = np.zeros((beam.N,beam.M))

        if self.shapeError is not None:
            # get shape of shape error input
            mirror_shape = np.shape(self.shapeError)

            # assume this is the central line shaper error along the long axis if only 1D
            if np.size(mirror_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = mirror_shape[0]
                # mirror coordinates (beam coordinates)
                max_zs = self.length / 2
                max_ys = self.width / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_zs / (Ms / 2 - 1)
                ys = np.linspace(-1,1,100)*max_ys

                print(np.shape(np.tile(self.shapeError,(100,1))))
                f = RectBivariateSpline(ys,zs,np.tile(self.shapeError,(100,1)))
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N, beam.M))*1e-9

                plt.figure()
                plt.imshow(shapeInterp)
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = RectBivariateSpline(ys,zs,self.shapeError)
                shapeInterp = np.reshape(f.ev(d_width.flatten(),d_length.flatten()),(beam.N,beam.M))*1e-9
                # f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                # shapeError2 = f(zi_1d - self.dx / np.tan(self.total_alpha), yi_1d - self.dy)

        gratingPhase = -self.order*2*np.pi*d_length * (self.n0 + self.n1 * d_length + self.n2 * d_length ** 2)

        # mask based on mirror length
        mask = np.logical_and(mask, np.abs(d_length)<self.length/2)
        # mask based on mirror width
        mask = np.logical_and(mask, np.abs(d_width)<self.width/2)

        # now write new beam coordinates in local beam coordinate system
        # (transforming from ellipse coordinates to local beam coordinates)
        transform_matrix2 = np.tensordot(np.reshape([beam.xhat,beam.yhat,beam.zhat], (3, 3)),
                                        np.reshape([mirror_x, mirror_y, mirror_z], (3, 3)), axes=(1, 1))
        shifted_plane2 = np.tensordot(transform_matrix2, shifted_plane, axes=(1, 0))

        # beam's x and y coordinates in the exit beam local coordinate system
        # note these are not in general on a regularly spaced grid but are based on tracking
        # rays through the course of the reflection.
        x_eff = shifted_plane2[0,:,:]
        y_eff = shifted_plane2[1,:,:]

        # calculate desired pixel size due to expected change in beam size (based on b factor), and
        # define new coordinate grids to interpolate onto at beam exit plane.
        self.b = (np.sin(self.alpha) / np.sin(self.beta0))

        if self.orientation == 0 or self.orientation == 2:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx / self.b
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)
        else:
            dx = beam.dx * (beam.zx + self.length / 2 * 1.1) / beam.zx
            x_out = np.linspace(-beam.M / 2 * dx, (beam.M / 2 - 1) * dx, beam.M)
            dy = beam.dy * (beam.zy + self.length * 1.1) / beam.zy / self.b
            y_out = np.linspace(-beam.N / 2 * dy, (beam.N / 2 - 1) * dy, beam.N)

        # 1D masks based on mirror acceptance, going through beam center.
        max_intensity = np.max(np.abs(beam.wave)**2)
        # mask2 = np.logical_and(mask,np.abs(beam.wave)**2>0.1*max_intensity)
        mask_x = mask[int(beam.N/2),:]>0
        mask_y = mask[:,int(beam.M/2)]>0
        # introducing weights for the polynomial fits based on beam intensity

        weight_x = (np.abs(beam.wave[int(beam.N/2),:])**2>0.1*max_intensity).astype(float)
        weight_y = (np.abs(beam.wave[:,int(beam.M/2)])**2>0.1*max_intensity).astype(float)

        # plt.figure()
        # plt.imshow((total_distance-np.mean(total_distance[mask]))*mask)
        #
        # plt.figure()
        # plt.imshow(np.abs(beam.wave)**2>0.1*max_intensity)

        incidence_angle = np.abs(np.arccos(np.sum(rays_out * mirror_normal, axis=0)))
        glancing = np.pi / 2 - incidence_angle

        # reflectivity = xraydb.mirror_reflectivity(self.material, glancing, beam.photonEnergy, self.density)

        # plt.figure()
        # plt.imshow((glancing-np.mean(glancing[mask]))*mask)

        # subtract best fit parabola in x-direction and best fit line in y direction
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff)# + np.polyval(p_coeff_y[:-2],y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff)# + np.polyval(p_coeff_y[:-2], y_eff)
        #
        # TRY THIS OUT LATER
        # if self.orientation==0 or self.orientation==2:
        #     total_distance -= np.polyval(p_coeff_x,x_eff) + np.polyval(p_coeff_y,y_eff)
        # else:
        #     total_distance -= np.polyval(p_coeff_y, y_eff) + np.polyval(p_coeff_x, x_eff)

        points = np.zeros((np.size(x_eff[mask]),2))
        points[:,0] = x_eff[mask]
        points[:,1] = y_eff[mask]
        # xi = np.zeros((np.size(x_out),2))
        # xi[:,0] = x_out
        # xi[:,1] = y_out
        xi_0, xi_1 = np.meshgrid(x_out, y_out)
        print('attempting interpolation')
        mask2 = interpolation.griddata(points, mask[mask], (xi_0, xi_1), method='nearest',fill_value=0)
        # mask2 = fmask(x_out,y_out)
        mask2[mask2<.9] = 0
        # mask2 = mask2.astype(int)
        mask2 = mask2 > 0.5

        # interpolate intensity onto new exit plane grid
        abs_out = interpolation.griddata(points, np.abs(beam.wave[mask]), (xi_0, xi_1), fill_value=0)

        # unwrap phase of beam at input
        angle_in = unwrap_phase(np.angle(beam.wave))

        # add quadratic phase if beam is not focused since this also needs to be interpolated
        quadratic = np.zeros_like(beam.x)

        if not beam.focused_x:
            quadratic += np.pi / beam.lambda0 / beam.zx * (beam.x) ** 2
        if not beam.focused_y:
            quadratic += np.pi / beam.lambda0 / beam.zy * (beam.y) ** 2

        # add the quadratic phase to the input phase
        angle_in += quadratic

        # add phase contribution from deviations in the distance traveled by each ray
        total_phase = (angle_in + 2 * np.pi / beam.lambda0 * total_distance
                       - shapeInterp * 4*np.pi*np.sin(self.alpha) / beam.lambda0 - gratingPhase)

        diff = gratingPhase-2 * np.pi / beam.lambda0 *total_distance
        diff -= np.mean(diff)

        p1 = np.polyfit(d_length[int(beam.N/2),:],total_distance[int(beam.N/2),:],1)
        plt.figure()
        plt.imshow(total_distance-np.polyval(p1,d_length))

        p1 = np.polyfit(d_length[int(beam.N/2),:],gratingPhase[int(beam.N/2),:],1)

        plt.figure()
        plt.imshow(gratingPhase-np.polyval(p1,d_length))

        # get polynomial fits based on new coordinates
        p_coeff_x = np.polyfit(x_eff[int(beam.N/2),:][mask_x], total_phase[int(beam.N/2),:][mask_x], 2,
                               w=weight_x[mask_x])

        p_coeff_y = np.polyfit(y_eff[:,int(beam.M/2)][mask_y], total_phase[:,int(beam.M/2)][mask_y], 2,
                               w=weight_y[mask_y])

        # calculate effective distance to focus based on total phase
        z_2 = np.pi / beam.lambda0 / p_coeff_x[-3]
        z_2_y = np.pi / beam.lambda0 / p_coeff_y[-3]

        # add this to what was already subtracted (might be able to do this in one step)
        # z_total_x = 1 / (1 / z_out_x + 1 / z_2)
        # z_total_y = 1 / (1 / z_out_y + 1 / z_2_y)
        z_total_x = z_2
        z_total_y = z_2_y
        print('new z: %.6f' % z_total_x)
        print('new z y: %.6f' % z_total_y)

        # calculate residual linear phase terms
        # linear += p_coeff_x[-2] * beam.lambda0/2/np.pi
        # linear_y += p_coeff_y[-2] * beam.lambda0/2/np.pi
        linear = p_coeff_x[-2] * beam.lambda0 / 2 / np.pi
        linear_y = p_coeff_y[-2] * beam.lambda0/2/np.pi

        print(linear)
        print(linear_y)

        # subtract linear terms since this should be taken care of based on central ray direction
        # Now this is being done in both tangential and sagittal directions, and we compensate with
        # a change in beam k-vector direction.
        total_phase -= np.polyval(p_coeff_x[-2:], x_eff) + np.polyval(p_coeff_y[-2:], y_eff)

        plt.figure()
        plt.imshow(total_phase * mask)

        # sutbtract off quadratic phase term if the beam is not focused
        if not beam.focused_x:
            total_phase -= np.polyval([p_coeff_x[-3], 0, 0], x_eff)
        if not beam.focused_y:
            total_phase -= np.polyval([p_coeff_y[-3], 0, 0], y_eff)

        # interpolate the phase onto the exit plane grid
        points = np.zeros((np.size(x_eff), 2))
        points[:, 0] = x_eff.flatten()
        points[:, 1] = y_eff.flatten()
        phase_interp = interpolation.griddata(points, total_phase.flatten(), (xi_0, xi_1), fill_value=0)

        # update beam complex amplitude using what has been interpolated onto the new grid
        beam.wave = abs_out * np.exp(1j * phase_interp)
        # multiply by mirror aperture that has been interpolated onto the exit plane grid
        beam.wave *= mask2

        if figon:
            plt.figure()
            plt.imshow(np.abs(beam.wave))

        ax0 = np.copy(beam.ax)
        ay0 = np.copy(beam.ay)

        # figure out where the beam is in global coordinates
        # change in angle
        k_i = rays_mirror[:, int(beam.N / 2), int(beam.M / 2)]
        k_f = rays_out[:, int(beam.N / 2), int(beam.M / 2)]

        k_f_global = np.tensordot(np.linalg.inv(transform_matrix), np.reshape(k_f, (3, 1, 1)), axes=(1, 0))
        k_f_global = k_f_global / np.sqrt(np.sum(np.abs(k_f_global ** 2)))

        # compensate for removing linear phase by adjusting beam k-vector
        beam.rotate_beam(delta_ax=linear)
        beam.rotate_beam(delta_ay=linear_y)

        print(beam.ax)
        delta_cx = ax0 * self.length / 2 * 1.1
        delta_cx += beam.ax * self.length / 2 * 1.1
        delta_cx += 2 * np.dot(self.normal, beam.xhat) * self.dx
        print('change in beam center')
        print(delta_cx)
        # beam.cx = -beam.cx + delta_cx
        # print(beam.cx)

        x_out, y_out = np.meshgrid(x_out, y_out)
        beam.x = x_out
        beam.y = y_out

        beam.new_fx()

        print('is beam in the correct direction?')
        print(np.arccos(np.dot(beam.zhat, k_f)))
        print(np.arccos(np.dot(beam.zhat, k_f_global[:, 0, 0])))
        print(k_f)
        print(k_f_global)

        # now figure out global coordinates
        # get back into global coordinates using inverse of transformation matrix, just looking at central ray
        inv_transform = np.linalg.inv(transform_matrix)

        # rotate into global coordinate system, but origin is still at ellipse center
        origin_global = np.tensordot(inv_transform, origin, axes=(1,0))

        # now add the mirror center in global coordinates, so that this should be the beam location
        # in global coordinates
        origin_global += np.reshape(mirror_center, (3, 1, 1))
        # origin_global -= np.reshape(self.normal*dx,(3,1))
        print(origin_global)
        # now shift origin to ellipse origin

        beam.global_x = origin_global[0,0,0]
        beam.global_y = origin_global[1,0,0]
        beam.global_z = origin_global[2,0,0]

        # account for coordinate scaling and/or changes to Rayleigh range,
        # whether beam is classified as focused or not.
        beam.change_z(new_zx=z_total_x,new_zy=z_total_y)

        beam.new_fx()
        print('global_x: %.2f' % beam.global_x)
        print('global_y: %.2f' % beam.global_y)
        print('global_z: %.2f' % beam.global_z)

        print(np.shape(beam.x))
        print(np.shape(beam.y))

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

        beamz = 0

        if self.orientation == 0:
            k_ix = -np.sin(self.alpha - beam.ax)
            k_iy = np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.x / np.sin(total_alpha)
            zi_1d = zi[0, :]
            yi = beam.y
            yi_1d = yi[:, 0]

            cz = beam.cx / np.sin(total_alpha)
            cy = beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = -beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            beamz = beam.zx

        elif self.orientation == 1:
            k_ix = -np.sin(self.alpha - beam.ay)
            k_iy = -np.sin(beam.ax)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = beam.y / np.sin(self.alpha + self.delta)
            zi_1d = zi[:, 0]
            yi = -beam.x
            yi_1d = yi[0, :]

            cz = beam.cy / np.sin(total_alpha)
            cy = -beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d-cz)*np.cos(total_alpha)
            alphaBeam = -beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            beamz = beam.zy

        elif self.orientation == 2:
            k_ix = -np.sin(self.alpha + beam.ax)
            k_iy = -np.sin(beam.ay)
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.x / np.sin(self.alpha + self.delta)
            zi_1d = zi[0, :]
            yi = -beam.y
            yi_1d = yi[:, 0]

            cz = -beam.cx / np.sin(total_alpha)
            cy = -beam.cy

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zx + (zi_1d - cz) * np.cos(total_alpha)
            alphaBeam = beam.ax - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            beamz = beam.zx

        elif self.orientation == 3:
            k_ix = -np.sin(self.alpha + beam.ay)
            k_iy = beam.ax
            k_iz = np.sqrt(1 - k_ix ** 2 - k_iy ** 2)

            # coordinate mapping for interpolation
            zi = -beam.y / np.sin(self.alpha + self.delta)
            zi_1d = zi[:, 0]
            yi = beam.x
            yi_1d = yi[0, :]

            cz = -beam.cy / np.sin(total_alpha)
            cy = beam.cx

            # beam radius across grating (grating can be long enough that the additional correction is needed
            zEff = beam.zy + (zi_1d - cz) * np.cos(total_alpha)

            alphaBeam = beam.ay - np.arctan((zi_1d - cz) * np.sin(total_alpha) / zEff)

            beamz = beam.zy

        k_i = np.array([k_ix, k_iy, k_iz])
        delta_k, k_f = self.rotation_grating(k_i, beam.lambda0)

        # beta at beam center
        beta1 = np.arccos(k_f[2])

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
                central_line = np.interp(zi_1d - self.dx / np.tan(total_alpha), zs, self.shapeError)
                # tile onto mirror short axis direction
                shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))*1e-9
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = mirror_shape[0]
                Ms = mirror_shape[1]
                # mirror coordinates
                max_xs = self.length / 2
                # mirror coordinates
                zs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.width / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = interpolation.interp2d(zs, ys, self.shapeError, fill_value=0)
                shapeError2 = f(zi_1d - self.dx / np.tan(total_alpha), yi_1d - self.dy)*1e-9

            if self.orientation == 1:
                shapeError2 = np.swapaxes(shapeError2, 0, 1)

            elif self.orientation == 3:
                shapeError2 = np.swapaxes(shapeError2, 0, 1)


        # project beam angle onto grating axis
        # Also take into account grating shift in dx (+dx corresponds to dz = -dx/alpha)

        # grating coordinates (along z-axis)
        z_g = np.linspace(-self.length / 2, self.length / 2, 1024)

        # deviation from average angle of incidence at each point along the grating
        # alphaBeamG = Util.interp_flip(z_g, zi_1d - self.dx / np.tan(total_alpha), alphaBeam)

        # account for all contributions to alpha
        alpha_total = self.alpha + self.delta + alphaBeam

        z_g = zi_1d - self.dx / np.tan(total_alpha)

        # calculate diffraction angle at every point on the grating
        beta = np.arccos(np.cos(alpha_total) - beam.lambda0 * (self.n0 + self.n1 * z_g + self.n2 * z_g ** 2))

        # calculate new source position
        # self.f = beamz*(self.beta0/self.alpha)**2

        # figure out distance to focus
        D1 = self.n1 / self.n0**2
        D0 = 1 / self.n0
        grating_focal_length = 1 / (self.lambda0 * D1 / D0 ** 2 / np.sin(self.beta0) ** 2)
        object_distance = beamz*(np.sin(self.beta0)/np.sin(self.alpha))**2
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
        # slope_error = -np.tan(beta - self.beta0)

        # plt.figure()
        # plt.plot(z_g, slope_error)

        # calculate phase contribution by integrating slope error. This is kind of equivalent to a height error but
        # we don't need to double-count it.
        # (do this with a polynomial fit up to 3rd order for now)
        # put center of z_g at zero
        # limit this to size of grating
        mask = np.abs(z_g) <= self.length/2
        p = np.polyfit(z_g[mask], slope_error[mask], 3)

        # integrate slope error (eventually move integration to after change of coordinates)
        p_int = np.polyint(p)

        # --- high order coefficients
        # change coordinate systems to new beam coordinates
        # scaling between grating z-axis and new beam coordinates
        scale = np.sin(self.beta0 - self.delta)
        p_scaled = Util.poly_change_coords(p_int, scale) * np.sin(self.beta0 - self.delta)

        # offset from center (along mirror z-axis)
        offset = cz - self.dx / np.tan(total_alpha)

        # scale the offset
        offset_scaled = offset * scale

        p_centered = Util.recenter_coeff(p_scaled, offset_scaled)

        # high order phase
        high_order = (2 * np.pi / beam.lambda0 * Util.polyval_high_order(p_centered, (zi-cz)*np.sin(self.beta0)))

        # offset from center (along mirror z-axis)
        offset = cz - self.dx / np.tan(total_alpha)

        # change coordinate systems to get back into beam coordinates
        # p_scaled = Util.poly_change_coords(p_int, scale) * np.sin(self.beta0 - self.delta)
        #
        # # scale the offset
        # offset_scaled = offset * scale

        # account for decentering
        p_recentered = Util.recenter_coeff(p_int, offset)

        # high order phase. Multiplied by sin(beta) because integration should actually happen in beam coordinates.
        # high_order = (2 * np.pi / beam.lambda0 * Util.polyval_high_order(p_recentered, zi - cz))

        # --- low order coefficients
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
        mirror = z_mask * y_mask

        # add shape error contribution to phase error
        high_order += (-4*np.pi/beam.lambda0/np.sin(total_alpha) *
                       np.sin((total_alpha+self.beta0-self.delta)/2)**2 * shapeError2)

        # multiply beam by aperture and phase
        beam.wave *= mirror * np.exp(1j * high_order)

        # handle beam re-pointing depending on the orientation
        if self.orientation == 0:
            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.rescale_x(np.sin(self.beta0) / np.sin(self.alpha))
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
            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.rescale_y(np.sin(self.beta0) / np.sin(self.alpha))
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
            # take into account coordinate rescaling
            beam.x -= beam.cx
            beam.rescale_x(np.sin(self.beta0) / np.sin(self.alpha))
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
            # take into account coordinate rescaling
            beam.y -= beam.cy
            beam.rescale_y(np.sin(self.beta0) / np.sin(self.alpha))
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

    def multiply(self, beam):
        """
        Method to multiply the beam by the collimator aperture.
        :param beam: Beam
            Beam object to propagate through the collimator. Beam is modified by this method.
        :return: None
        """
        # define aperture in beam coordinates
        aperture = (np.abs((beam.x - self.dx) ** 2 + (beam.y - self.dy) ** 2) < (self.diameter / 2) ** 2).astype(float)
        # multiply beam by aperture
        beam.wave *= aperture

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

    def multiply(self, beam):
        """
        Method to propagate beam through the slit.
        :param beam: Beam
            Beam object to propagate through slits. Beam is modified by this method.
        :return: None
        """
        # define slit aperture in beam coordinates
        aperture = ((np.abs(beam.x - self.dx) < self.x_width / 2).astype(float) *
                    (np.abs(beam.y - self.dy) < self.y_width / 2).astype(float))

        # multiply beam by aperture
        beam.wave *= aperture

    def propagate(self, beam):
        """
        Method to propagate beam through aperture. Calls multiply.
        :param beam: Beam
            Beam object to propagate through slits. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


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
        self.dz = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        self.z = (downstream_component.z + upstream_component.z) / 2.
        self.global_x = 0
        self.global_y = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None

    # def propagate(self, beam):
    #     """
    #     Method to propagate through a Drift section
    #     :param beam: Beam
    #         Beam object to propagate through the drift section. Beam is modified.
    #     :return: None
    #     """
    #     # propagate the beam along the full length of the Drift.
    #
    #     # can put re-calculation of distance here
    #     # get beam k
    #     k = beam.get_k()
    #     print('global_x %.2f' % beam.global_x)
    #     print('global_y %.2f' % beam.global_y)
    #
    #     if issubclass(type(self.downstream_component), Mirror):
    #         # beam global coordinates are currently on surface of upstream component
    #         # get global alpha for mirror
    #         alpha = self.downstream_component.global_alpha
    #         z_m = self.downstream_component.z
    #         x_m = self.downstream_component.global_x
    #         y_m = self.downstream_component.global_y
    #         # find z location where two lines intersect
    #         if self.downstream_component.orientation == 0:
    #             z_intersect = ((-k[0]/k[2]*beam.global_z + beam.global_x + np.tan(alpha)*z_m - x_m)/
    #                            (np.tan(alpha) - k[0]/k[2]))
    #
    #         elif self.downstream_component.orientation == 1:
    #             z_intersect = ((-k[1]/k[2]*beam.global_z + beam.global_y + np.tan(alpha)*z_m - y_m)/
    #                            (np.tan(alpha) - k[1]/k[2]))
    #
    #         elif self.downstream_component.orientation == 2:
    #             z_intersect = ((-k[0] / k[2] * beam.global_z + beam.global_x + np.tan(alpha) * z_m - x_m) /
    #                            (np.tan(alpha) - k[0] / k[2]))
    #
    #         else:
    #             z_intersect = ((-k[1] / k[2] * beam.global_z + beam.global_y + np.tan(alpha) * z_m - y_m) /
    #                            (np.tan(alpha) - k[1] / k[2]))
    #
    #     else:
    #         z_m = self.downstream_component.z
    #         x_m = self.downstream_component.global_x
    #         y_m = self.downstream_component.global_y
    #         # elev = self.downstream_component.elevation
    #         azim = self.downstream_component.azimuth
    #         z_intersect = (((x_m - beam.global_x) * np.tan(azim) * k[2] + z_m * k[2] + k[0] * beam.global_z * np.tan(
    #             azim)) / \
    #                        (np.tan(azim) * k[0] + k[2]))
    #
    #     x_intersect = k[0] / k[2] * (z_intersect - beam.global_z) + beam.global_x
    #     print('x intersect: %.4e' % x_intersect)
    #     print('component x: %.4e' % self.downstream_component.global_x)
    #     y_intersect = k[1] / k[2] * (z_intersect - beam.global_z) + beam.global_y
    #     print('y intersect: %.4e' % y_intersect)
    #     print('component y: %.4e' % self.downstream_component.global_y)
    #     dx = x_intersect - beam.global_x
    #     dy = y_intersect - beam.global_y
    #     dz = z_intersect - beam.global_z
    #     # re-calculate propagation distance
    #     old_z = np.copy(self.dz)
    #
    #     self.dz = np.sqrt(dx**2 + dy**2 + dz**2)
    #     print('delta z: %.2f' % ((self.dz - old_z)*1e6))
    #
    #     beam.beam_prop(self.dz)

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

            # if self.downstream_component.orientation==0 or self.downstream_component.orientation==1:
            #     mirror_center += normal*self.downstream_component.dx
            # else:
            #     mirror_center -= normal*self.downstream_component.dx
            # making this consistent with trace_surface
            mirror_center += normal * self.downstream_component.dx

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

        if issubclass(type(self.downstream_component), Mirror):

            print('found curved mirror')
            intersection = self.downstream_component.find_intersection(beam).flatten()
            print(intersection)
            x_intersect = intersection[0]
            y_intersect = intersection[1]
            z_intersect = intersection[2]
            dx = x_intersect - beam.global_x
            dy = y_intersect - beam.global_y
            dz = z_intersect - beam.global_z
        # re-calculate propagation distance
        old_z = np.copy(self.dz)

        self.downstream_component.x_intersect = x_intersect
        self.downstream_component.y_intersect = y_intersect
        self.downstream_component.z_intersect = z_intersect

        self.dz = np.sqrt(dx**2 + dy**2 + dz**2)
        self.downstream_component.correction = self.dz - old_z
        print('delta z: %.2f' % ((self.dz - old_z)*1e6))

        # beam.global_x = x_intersect
        # beam.global_y = y_intersect
        # beam.global_z =

        beam.beam_prop(self.dz)


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
        aperture = ((np.abs(beam.x - self.dx) < self.x_width / 2).astype(float) *
                    (np.abs(beam.y - self.dy) < self.y_width / 2).astype(float))

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
        transmission = np.exp(-beam.k0 * beta * thickness) * aperture

        # multiply by transmission
        beam.wave *= transmission

        # adjust beam direction
        beam.ax += p1_x
        beam.ay += p1_y

    def propagate(self, beam):
        """
        Method to propagate beam through prism. Calls multiply.
        :param beam: Beam
            Beam object to propagate through prism. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class CRL:
    """
    Class to represent parabolic compound refractive lenses (CRLs).

    Attributes
    ----------
    name: str
        Name of the device (e.g. CRL1)
    diameter: float
        Diameter beyond which the lenses absorb all photons. (meters)
    roc: float
        Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
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

    def __init__(self, name, **kwargs):
        """
        Method to create a CRL object.
        :param name: str
            Name of the device (e.g. CRL1)
        :param diameter: float
            Diameter beyond which the lenses absorb all photons. (meters)
        :param roc: float
            Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
        :param material: str
            Lens material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
        :param z: float
            z location of lenses along beamline.
        :param dx: float
            Lens de-centering along beam's x-axis.
        :param dy: float
            Lens de-centering along beam's y-axis.
        """

        # set some attributes
        self.name = name
        self.diameter = 300e-6
        self.roc = 50e-6
        self.E0 = None
        self.f = None
        self.material = 'Be'
        self.dx = 0
        self.dy = 0
        self.z = 0
        self.global_x = 0
        self.global_y = 0
        self.shapeError = None
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None
        self.x_intersect = 0
        self.y_intersect = 0
        self.z_intersect = 0

        # set allowed kwargs
        allowed_arguments = ['diameter', 'roc', 'E0', 'f', 'material', 'dx', 'dy', 'z', 'shapeError']

        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

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
            self.f = self.roc / 2 / delta

    def multiply(self, beam):
        """
        Method to propagate beam through CRL
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """

        beam_shift = np.array([self.x_intersect - self.global_x,
                               self.y_intersect - self.global_y,
                               self.z_intersect - self.z])
        x_shift = np.dot(beam_shift, self.xhat)
        y_shift = np.dot(beam_shift, self.yhat)

        xi = beam.x + x_shift
        yi = beam.y + y_shift
        xi_1d = xi[0,:]
        yi_1d = yi[:,0]

        shapeError2 = np.zeros_like(xi)

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # check for phase error
        if self.shapeError is not None:
            # get shape of shape error input
            lens_shape = np.shape(self.shapeError)

            # assume this is the central line shape error along the long axis if only 1D
            if np.size(lens_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = lens_shape[0]
                # mirror coordinates (beam coordinates)
                max_xs = self.diameter / 2
                # lens coordinates
                xs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                # 1D interpolation onto beam coordinates
                central_line = np.interp(xi_1d - self.dx, xs, self.shapeError)
                # tile onto mirror short axis direction
                shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = lens_shape[0]
                Ms = lens_shape[1]
                # mirror coordinates
                max_xs = self.diameter / 2
                # mirror coordinates
                xs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.diameter / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = interpolation.interp2d(xs, ys, self.shapeError, fill_value=0)
                shapeError2 = f(xi_1d - self.dx, yi_1d - self.dy)

        # CRL thickness (for now assuming perfect lenses but might add aberrations later)
        thickness = 2 * self.roc * (1 / 2 * ((xi - self.dx) ** 2 + (yi - self.dy) ** 2) / self.roc ** 2)

        thickness += shapeError2

        # lens aperture
        mask = (((xi - self.dx) ** 2 + (yi - self.dy) ** 2) < (self.diameter / 2) ** 2).astype(float)

        # subtract 2nd order and linear terms
        phase = -beam.k0 * delta * (thickness - 2 / 2 / self.roc * ((xi - self.dx) ** 2 + (yi - self.dy) ** 2))

        # 2nd order
        p2 = -beam.k0 * delta * 2 / 2 / self.roc
        # 1st order
        p1_x = p2 * 2 * (x_shift - self.dx)
        p1_y = p2 * 2 * (y_shift - self.dy)

        # lens transmission based on beta and thickness profile
        transmission = np.exp(-beam.k0 * beta * thickness) * np.exp(1j * phase) * mask

        # adjust beam properties
        new_zx = 1 / (1 / beam.zx + p2 * beam.lambda0 / np.pi)
        new_zy = 1 / (1 / beam.zy + p2 * beam.lambda0 / np.pi)
        beam.change_z(new_zx=new_zx, new_zy=new_zy)

        delta_ax = p1_x * beam.lambda0 / 2 / np.pi
        delta_ay = p1_y * beam.lambda0 / 2 / np.pi

        beam.rotate_beam(delta_ax=delta_ax, delta_ay=delta_ay)

        print('focal length: %.2f' % (-1/(p2*beam.lambda0/np.pi)))

        # multiply beam by CRL transmission function and any high order phase
        beam.wave *= transmission

    def propagate(self, beam):
        """
        Method to propagate beam through CRL. Calls multiply.
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """
        self.multiply(beam)


class CRL1D(CRL):
    """
    Class to represent parabolic compound refractive lenses (CRLs).

    Attributes
    ----------
    name: str
        Name of the device (e.g. CRL1)
    diameter: float
        Diameter beyond which the lenses absorb all photons. (meters)
    roc: float
        Lens radius of curvature. Lenses are actually parabolic but are labeled this way. (meters)
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
    orientation: int
        Whether or not this is a horizontal or vertical lens (0 for horizontal, 1 for vertical).
    """
    def __init__(self, name, orientation=0, **kwargs):
        super().__init__(name, **kwargs)

        self.orientation = orientation

    def multiply(self, beam):
        """
        Method to propagate beam through 1D CRL
        :param beam: Beam
            Beam object to propagate through CRL. Beam is modified by this method.
        :return: None
        """

        beam_shift = np.array([self.x_intersect - self.global_x,
                               self.y_intersect - self.global_y,
                               self.z_intersect - self.z])
        x_shift = np.dot(beam_shift, self.xhat)
        y_shift = np.dot(beam_shift, self.yhat)

        xi = beam.x + x_shift
        yi = beam.y + y_shift
        xi_1d = xi[0, :]
        yi_1d = yi[:, 0]

        shapeError2 = np.zeros_like(xi)

        # interpolate to find index of refraction at beam's energy
        delta = np.interp(beam.photonEnergy, self.energy, self.delta)
        beta = np.interp(beam.photonEnergy, self.energy, self.beta)

        # check for phase error
        if self.shapeError is not None:
            # get shape of shape error input
            lens_shape = np.shape(self.shapeError)

            # assume this is the central line shape error along the long axis if only 1D
            if np.size(lens_shape) == 1:
                # assume this is the central line and it's the same across the mirror width
                Ms = lens_shape[0]
                # mirror coordinates (beam coordinates)
                max_xs = self.diameter / 2
                # lens coordinates
                xs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                # 1D interpolation onto beam coordinates
                central_line = np.interp(xi_1d - self.dx, xs, self.shapeError)
                # tile onto mirror short axis direction
                shapeError2 = np.tile(central_line, (np.size(yi_1d), 1))
            # if 2D, assume index 0 corresponds to short axis, index 1 to long axis
            else:
                # shape error array shape
                Ns = lens_shape[0]
                Ms = lens_shape[1]
                # mirror coordinates
                max_xs = self.diameter / 2
                # mirror coordinates
                xs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * max_xs / (Ms / 2 - 1)
                max_ys = self.diameter / 2
                ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * max_ys / (Ns / 2 - 1)

                # 2D interpolation onto beam coordinates
                f = interpolation.interp2d(xs, ys, self.shapeError, fill_value=0)
                shapeError2 = f(xi_1d - self.dx, yi_1d - self.dy)

        if self.orientation == 0:
            beamx = xi
            beamz = beam.zx
            beamc = beam.cx
            dx_lens = self.dx
        else:
            beamx = yi
            beamz = beam.zy
            beamc = beam.cy
            dx_lens = self.dy

        # CRL thickness (for now assuming perfect lenses but might add aberrations later)
        thickness = 2 * self.roc * (1 / 2 * ((beamx - dx_lens) ** 2) / self.roc ** 2)

        thickness += shapeError2

        # lens aperture
        mask = (((xi - self.dx) ** 2 + (yi - self.dy) ** 2) < (self.diameter / 2) ** 2).astype(float)

        # subtract 2nd order and linear terms
        phase = -beam.k0 * delta * (thickness - 2 / 2 / self.roc * ((beamx - dx_lens) ** 2))

        # 2nd order
        p2 = -beam.k0 * delta * 2 / 2 / self.roc
        # 1st order
        p1_x = p2 * 2 * (beamc - self.dx)

        # lens transmission based on beta and thickness profile
        transmission = np.exp(-beam.k0 * beta * thickness) * np.exp(1j * phase) * mask
        # adjust beam properties
        new_zx = 1 / (1 / beamz + p2 * beam.lambda0 / np.pi)

        if self.orientation == 0:
            beam.change_z(new_zx=new_zx)
            delta_ax = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ax=delta_ax)

        else:
            beam.change_z(new_zy=new_zx)
            delta_ay = p1_x * beam.lambda0 / 2 / np.pi
            beam.rotate_beam(delta_ay=delta_ay)


        # multiply beam by CRL transmission function and any high order phase
        beam.wave *= transmission

class PPM:
    """
    Class to represent profile monitor output from PPMs.

    Attributes
    ----------
    name: str
        device name (e.g. IM1K4)
    FOV: float
        width of the (restricted to be square) field of view
    n: int
        number of pixels across the image. Image is NxN.
    dx: float
        PPM pixel size
    z: float
        z location along beamline
    blur: bool
        Blur beam intensity prior to interpolation if True, simulating blurring due to finite resolution of
        microscope. Mainly important for wavefront sensor profile monitors.
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
    """

    def __init__(self, name, **kwargs):
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
        :param calc_phase: bool
            whether to calculate/interpolate the phase profile at the PPM. Used with Pulse class.
        """

        # set defaults
        self.FOV = 10e-3
        self.z = None
        self.global_x = 0
        self.global_y = 0
        self.N = 2048
        self.blur = False
        self.view_angle_x = 90
        self.view_angle_y = 90
        self.resolution = 5e-6
        self.calc_phase = False
        self.threshold = 0.0
        self.azimuth = 0
        self.elevation = 0
        self.xhat = None
        self.yhat = None
        self.zhat = None
        self.x_intersect = 0
        self.y_intersect = 0
        self.z_intersect = 0

        # set allowed kwargs
        allowed_arguments = ['N', 'dx', 'FOV', 'z', 'blur', 'view_angle_x',
                             'view_angle_y', 'resolution', 'calc_phase', 'threshold']
        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        # set some attributes
        # self.N = N
        self.M = np.copy(self.N)
        self.dx = self.FOV / self.N
        # self.FOV = FOV
        # self.z = z
        self.name = name
        # self.blur = blur
        # self.view_angle_x = view_angle_x
        # self.view_angle_y = view_angle_y
        # self.resolution = resolution
        # self.calc_phase = calc_phase

        # calculate PPM coordinates
        self.x = np.linspace(-self.N / 2, self.N / 2 - 1, self.N) * self.dx
        self.y = np.copy(self.x)

        f_x = np.linspace(-self.N / 2., self.N / 2. - 1., self.N) / self.N / self.dx
        f_y = np.linspace(-self.N / 2., self.N / 2. - 1., self.N) / self.N / self.dx

        # get 2D coordinate arrays
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.f_x, self.f_y = np.meshgrid(f_x, f_y)

        # initialize some attributes
        self.profile = np.zeros((self.N, self.N))
        self.phase = np.zeros((self.N, self.N), dtype=complex)
        self.zx = 0
        self.zy = 0
        self.cx_beam = 0
        self.cy_beam = 0
        self.x_lineout = np.zeros(self.M)
        self.y_lineout = np.zeros(self.N)
        self.fit_x = np.zeros(self.M)
        self.fit_y = np.zeros(self.N)
        self.amp_x = 0
        self.amp_y = 0
        self.cx = 0
        self.cy = 0
        self.wx = 0
        self.wy = 0
        self.xbin = 1
        self.lambda0 = 0.0
        self.centroid_is_valid = 0
        self.wavefront_is_valid = 0
        self.group_delay = 0

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

        self.amp_x = np.max(line_x)-np.min(line_x)
        self.amp_y = np.max(line_y)-np.min(line_y)

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

        fit_validity = 1

        # Gaussian fitting. Using try/except to deal with any fitting errors
        try:
            # only fit in the region where we have signal
            mask = line_x > .1
            # Gaussian fit using Scipy curve_fit. Using only data that has >10% of the max
            px, pcovx = optimize.curve_fit(Util.fit_gaussian, self.x[mask] * 1e6, line_x[mask], p0=guessx)
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
            py, pcovy = optimize.curve_fit(Util.fit_gaussian, self.y[mask] * 1e6, line_y[mask], p0=guessy)
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
        validity = ((self.amp_x > 0) and (self.amp_y > 0) and fit_validity and
                    (fwhm_x < np.max(2*self.x)) and (fwhm_y < np.max(2*self.y)))

        self.centroid_is_valid = validity

        return cx, cy, fwhm_x, fwhm_y, fwx_guess, fwy_guess

    def calc_profile(self, beam):
        """
        Method to calculate the beam profile at the PPM screen.
        :param beam: Beam
            Beam object for viewing at PPM location. The Beam object is not modified by this method.
        :return: None
        """

        beam_shift = np.array([self.x_intersect - self.global_x,
                               self.y_intersect - self.global_y,
                               self.z_intersect - self.z])
        x_shift = np.dot(beam_shift, self.xhat)
        y_shift = np.dot(beam_shift, self.yhat)

        # Calculate intensity from complex beam
        profile = np.abs(beam.wave) ** 2

        # coordinate scaling due to off-axis viewing angle
        scaling_x = 1 / np.sin(self.view_angle_x * np.pi / 180)
        scaling_y = 1 / np.sin(self.view_angle_y * np.pi / 180)

        # if blurring is used, apply a gaussian filter
        if self.blur:
            # calculate blur widths in pixels, based on beam's pixel size
            x_width = self.resolution / beam.dx
            y_width = self.resolution / beam.dy
            # apply blurring using ndimage gaussian_filter
            profile = ndimage.filters.gaussian_filter(profile, sigma=(y_width, x_width))

        # get beam coordinates for interpolation
        x = beam.x[0,:] + x_shift
        y = beam.y[:,0] + y_shift
        # interpolating function from Scipy's interp2d. Extrapolation value is set to zero.
        f = interpolation.interp2d(x * scaling_x, y * scaling_y, profile, fill_value=0)
        # do the interpolation to get the profile we'll see on the PPM
        self.profile = f(self.x, self.y)

        if self.calc_phase:
            phase = unwrap_phase(np.angle(beam.wave))
            f_phase = interpolation.interp2d(x * scaling_x, y * scaling_y, phase, fill_value=0)
            self.phase = f_phase(self.x, self.y)

            if not beam.focused_x:
                # self.phase += np.pi / beam.lambda0 / beam.zx * (self.xx - beam.cx)**2
                self.zx = beam.zx
                self.cx_beam = beam.cx
            if not beam.focused_y:
                # self.phase += np.pi / beam.lambda0 / beam.zy * (self.yy - beam.cy)**2
                self.zy = beam.zy
                self.cy_beam = beam.cy
            self.phase += 2 * np.pi / beam.lambda0 * beam.ax * (self.xx - beam.cx)
            self.phase += 2 * np.pi / beam.lambda0 * beam.ay * (self.yy - beam.cy)

        self.group_delay = beam.group_delay

        # calculate horizontal lineout
        self.x_lineout = np.sum(self.profile, axis=0)
        # calculate vertical lineout
        self.y_lineout = np.sum(self.profile, axis=1)

        # get beam wavelength
        self.lambda0 = beam.lambda0

        # calculate centroids and beam widths
        self.cx, self.cy, self.wx, self.wy, wx2, xy2 = self.beam_analysis(self.x_lineout, self.y_lineout)

    def propagate(self, beam):
        """
        Method to propagate beam through PPM. Calls calc_profile.
        :param beam: Beam
            Beam object for viewing at PPM location. The Beam object is not modified by this method.
        :return: None
        """
        self.calc_profile(beam)

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

        # print('retrieving wavefront')

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
        x_center = Util.coordinate_to_pixel(self.cx, self.dx, self.M)
        y_center = Util.coordinate_to_pixel(self.cy, self.dx, self.N)

        # get lineouts from 2d profile
        lineout_x = Util.get_horizontal_lineout(self.profile, x_center=x_center, y_center=y_center,
                                                half_length=x_lim, half_width=lineout_width / 2)
        lineout_y = Util.get_vertical_lineout(self.profile, x_center=x_center, y_center=y_center,
                                              half_length=y_lim, half_width=lineout_width / 2)

        # expected spatial frequency of Talbot pattern (1/m)
        peak = 1. / mag / wfs.pitch * fraction

        # spatial frequency now in units of (1/pixels)
        fc = peak * self.dx

        # calculate pitch from lineouts. See pitch module.
        # print('getting lineouts')
        xline = TalbotLineout(lineout_x, fc, fraction, pad=True)
        yline = TalbotLineout(lineout_y, fc, fraction, pad=True)

        # parameters for calculating Legendre coefficients
        wfs_param = {
                "dg": wfs.pitch,  # wavefront sensor pitch (m)
                "fraction": fraction,  # wavefront sensor fraction
                "dx": self.dx*self.xbin,  # PPM pixel size
                "zT": zT,  # distance between WFS and PPM
                "lambda0": self.lambda0,  # beam wavelength
                "fc": fc  # spatial frequency of expected peak (1/pixels)
                }

        # calculate Legendre coefficients
        # print('getting Legendre coefficients')
        wfs_param['dg'] = wfs.x_pitch_sim
        z_x, coeff_x, x_prime, x_res, fit_object = xline.get_legendre(wfs_param)
        wfs_param['dg'] = wfs.y_pitch_sim
        z_y, coeff_y, y_prime, y_res, fit_object = yline.get_legendre(wfs_param)
        # print('found Legendre coefficients')

        # pixel size for retrieved wavefront
        dx_prime = x_prime[1] - x_prime[0]
        dy_prime = y_prime[1] - y_prime[0]

        # re-center residual phase coordinates on beam center
        x_prime += (x_center-self.M/2) * dx_prime
        y_prime += (y_center-self.N/2) * dy_prime

        # convert coordinates to microns
        x_prime = x_prime * 1e6
        y_prime = y_prime * 1e6

        rms_x = np.std(x_res)
        rms_y = np.std(y_res)

        # print calculated distance to focus
        # print('Distance to source: '+str(z_x))
        # print('Distance to source: '+str(z_y))

        # output. See method docstring for descriptions.
        wfs_data = {
                'x_res': x_res,
                'x_prime': x_prime,
                'y_res': y_res,
                'y_prime': y_prime,
                'coeff_x': coeff_x,
                'coeff_y': coeff_y,
                'z_x': z_x,
                'z_y': z_y
                }

        return wfs_data, wfs_param

    def retrieve_wavefront2D(self, basis_file, wfs, threshold=0.01):
        """
        Method to calculate wavefront in the case where there is a wavefront sensor upstream of the PPM.
        :param basis_file: string
            Path to file containing pickled Legendre basis object.
        :param wfs: WFS object
            Grating structure that generates Talbot interferometry patterns. Passed to this method to gain access
            to its attributes.
        :param threshold: float
            Optional, controls how much to threshold the intensity when recovering the wavefront
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

        # go ahead and retrieve 1D wavefront first
        wfs_data, wfs_param = self.retrieve_wavefront(wfs)

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Talbot image processing
        # load basis
        with open(basis_file, 'rb') as f:
            fit_object = pickle.load(f)

        # initialize Talbot image processing
        image_calc = TalbotImage(self.profile, wfs_param['fc'], fraction)

        # add parameters for calculating Legendre coefficients
        wfs_param['downsample'] = 3
        wfs_param['zf'] = wfs.f0
        wfs_param['dg'] = wfs.x_pitch_sim

        # calculate 2D legendre coefficients
        print('getting 2D Legendre coefficients')
        recovered_beam, fit_params = image_calc.get_legendre(fit_object, wfs_param, threshold=threshold)

        # get complete wavefront with defocus
        x = fit_params['x']
        y = fit_params['y']
        px = fit_params['px']
        py = fit_params['py']
        coeff = fit_params['coeff']

        # add defocus to wavefront fit
        full_wave = fit_object.wavefront_fit(coeff) + px * x**2 + py * y**2

        # output. See method docstring for descriptions.
        wfs_data2D = {
                'recovered': recovered_beam,
                'wave': full_wave
                }

        wfs_data.update(fit_params)

        wfs_data.update(wfs_data2D)

        return wfs_data

    def add_complex_profile(self, beam, zx_ref=None, zy_ref=None):

        if zx_ref is None:
            zx_ref = np.inf
        if zy_ref is None:
            zy_ref = np.inf

        beam_amp = np.abs(beam.wave)
        beam_phase = np.angle(beam.wave)

        # get beam coordinates for interpolation
        x = beam.x[0, :]
        y = beam.y[:, 0]
        # interpolating function from Scipy's interp2d. Extrapolation value is set to zero.
        f_amp = interpolation.interp2d(x, y, beam_amp, fill_value=0)
        f_phase = interpolation.interp2d(x, y, beam_phase, fill_value=0)

        # do the interpolation to get the profile we'll see on the PPM
        amp_interp = f_amp(self.x, self.y)
        phase_interp = f_phase(self.x, self.y)

        # add linear phase (centered on beam)
        phase_interp += 2*np.pi/beam.lambda0 * (beam.ax * self.xx + beam.ay * self.yy)

        # add quadratic phase (centered on beam)
        phase_interp += np.pi/beam.lambda0 * (self.xx**2*(1/beam.zx - 1/zx_ref)
                                              + self.yy**2 * (1/beam.zy - 1/zy_ref))

        # figure out quadratic phase later
        complex_profile_add = amp_interp * np.exp(1j*phase_interp)

        # add to current complex profile
        self.complex_profile += complex_profile_add

        # update profile
        self.profile = np.abs(self.complex_profile)**2

        # calculate horizontal lineout
        self.x_lineout = np.sum(self.profile, axis=0)
        # calculate vertical lineout
        self.y_lineout = np.sum(self.profile, axis=1)

        # calculate centroids and beam widths
        self.cx, self.cy, self.wx, self.wy, wx2, xy2 = self.beam_analysis(self.x_lineout, self.y_lineout)

    def add_profile(self, profile):
        self.profile += profile
        # calculate horizontal lineout
        self.x_lineout = np.sum(self.profile, axis=0)
        # calculate vertical lineout
        self.y_lineout = np.sum(self.profile, axis=1)

        # calculate centroids and beam widths
        self.cx, self.cy, self.wx, self.wy, wx2, xy2 = self.beam_analysis(self.x_lineout, self.y_lineout)

    def view_horizontal(self, ax=None, normalized=True, log=False, show_fit=True, legend=False, label='Lineout'):
        """
        Method to view
        :param normalized: whether to normalize the lineout
        :return:
        """

        gaussian_fit = np.exp(-(self.x - self.cx) ** 2 / 2 / (self.wx / 2.355) ** 2)

        if ax is None:
            # generate the figure
            plt.figure()
            ax = plt.subplot2grid((1,1), (0, 0))
        if normalized:
            # show the vertical lineout (distance in microns)
            if log:
                ax.semilogy(self.x * 1e6, self.x_lineout / np.max(self.x_lineout), label=label)
            else:
                ax.plot(self.x * 1e6, self.x_lineout / np.max(self.x_lineout), label=label)
                ax.set_ylim(0, 1.05)
            ax.set_ylabel('Intensity (normalized)')
        else:
            # show the vertical lineout (distance in microns)
            if log:
                ax.semilogy(self.x * 1e6, self.x_lineout, label=label)
            else:
                ax.plot(self.x * 1e6, self.x_lineout, label=label)
            gaussian_fit *= np.max(self.x_lineout)
            ax.set_ylabel('Intensity (arbitrary units)')
        # also plot the Gaussian fit
        if show_fit:
            if log:
                ax.semilogy(self.x*1e6, gaussian_fit, label='fit')
            else:
                ax.plot(self.x * 1e6, gaussian_fit, label='fit')
        if legend:
            ax.legend()
        ax.set_xlabel('X Coordinates (\u03BCm)')
        # show a grid
        ax.grid(True)
        # set limits


        return ax

    def view_vertical(self, ax_y=None, normalized=True, log=False, show_fit=True, legend=False, label='Lineout', lineout=False):
        """
        Method to view
        :param normalized: whether to normalize the lineout
        :return:
        """

        gaussian_fit = np.exp(-(self.y - self.cy) ** 2 / 2 / (self.wy / 2.355) ** 2)

        # calculated beam center in pixels
        x_center = Util.coordinate_to_pixel(self.cx, self.dx, self.M)
        y_center = Util.coordinate_to_pixel(self.cy, self.dx, self.N)

        # lineout boundaries in pixels (distance from center)
        x_lim = int(self.wx / self.dx) * 4
        y_lim = int(self.wy / self.dx) * 4

        if lineout:
            # get lineouts from 2d profile
            lineout_y = Util.get_vertical_lineout(self.profile, x_center=x_center, y_center=y_center, half_width=0)
        else:
            lineout_y = self.y_lineout

        if ax_y is None:
            # generate the figure
            plt.figure()
            ax_y = plt.subplot2grid((1,1), (0, 0))
        if normalized:
            # show the vertical lineout (distance in microns)
            if log:
                ax_y.semilogy(self.y * 1e6, lineout_y / np.max(lineout_y), label=label)
            else:
                ax_y.plot(self.y * 1e6, lineout_y / np.max(lineout_y), label=label)
                ax_y.set_ylim(0, 1.05)
            ax_y.set_ylabel('Intensity (normalized)')
        else:
            # show the vertical lineout (distance in microns)
            if log:
                ax_y.semilogy(self.y * 1e6, lineout_y, label=label)
            else:
                ax_y.plot(self.y * 1e6, lineout_y, label=label)
            gaussian_fit *= np.max(lineout_y)
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

        all_extrema = np.array([minx,maxx,miny,maxy])
        min_extrema = np.min(np.abs(all_extrema))
        if min_extrema<1:
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

    def complex_beam(self):
        if self.calc_phase:
            # reshape into 2 dimensional representation
            complex_beam = np.sqrt(self.profile) * np.exp(1j*self.phase)
        else:
            complex_beam = np.sqrt(self.profile)
        return complex_beam, self.group_delay, self.zx, self.zy, self.cx_beam, self.cy_beam


class PPM_Device(PPM):
    """
    Child class of PPM that is used for a physical PPM, rather than simulated.
    """
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.imager_prefix = name
        self.threshold = 0.0001

        # set allowed kwargs
        allowed_arguments = ['average','threshold','fit_object']

        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        # get Y motor state
        self.state = EpicsSignalRO(self.imager_prefix+'MMS:STATE:GET_RBV', auto_monitor=True)
        # define possible states depending on imager type
        if 'XTES' in self.imager_prefix:
            self.states_list = ['Unknown', 'OUT', 'YAG', 'DIAMOND', 'RETICLE']
        elif 'PPM' in self.imager_prefix:
            self.states_list = ['Unknown', 'OUT', 'POWERMETER', 'YAG1', 'YAG2']
        else:
            self.states_list = []

        self.cam_name = self.imager_prefix + 'CAM:'
        if 'MONO' in self.imager_prefix:
            self.cam_name = self.imager_prefix
        self.epics_name = self.cam_name + 'IMAGE3:'
        # get acquisition info (this is in seconds)
        self.acquisition_period = PV(self.epics_name[:-7] + 'AcquirePeriod_RBV').get()

        # check if Image3 is available
        port = PV(self.epics_name + 'PortName_RBV').get()
        array_rate = PV(self.epics_name + 'ROI:EnableCallbacks').get()

        if port is None or array_rate==0:
            self.epics_name = self.imager_prefix + 'CAM:IMAGE1:'
            self.acquisition_period = PV(self.imager_prefix + 'CAM:AcquirePeriod_RBV').get()

        port = PV(self.epics_name + 'PortName_RBV').get()

        array_rate = PV(self.epics_name + 'ROI:EnableCallbacks').get()


        if port is None or array_rate==0:
            self.epics_name = self.imager_prefix + 'CAM:IMAGE2:'
            self.acquisition_period = PV(self.imager_prefix + 'CAM:AcquirePeriod_RBV').get()

        port = PV(self.epics_name + 'PortName_RBV').get()

        if port is None:
            self.epics_name = self.imager_prefix + 'DATA1:'
            self.acquisition_period = PV(self.imager_prefix + 'AcquirePeriod_RBV').get()
        

        self.orientation = 'action0'

        print(self.epics_name)


        FOV_dict = {
            'IM2K4': 8.5,
            'IM3K4': 8.5,
            'IM4K4': 5.0,
            'IM5K4': 8.5,
            'IM6K4': 8.5,
            'IM1K1': 8.5,
            'IM2K1': 8.5,
            'IM1K2': 8.5,
            'IM2K2': 18.5,
            'IM3K2': 18.5,
            'IM4K2': 8.5,
            'IM5K2': 8.5,
            'IM6K2': 5.0,
            'IM7K2': 5.0,
            'IM1L1': 8.5,
            'IM2L1': 8.5,
            'IM3L1': 8.5,
            'IM4L1': 8.5,
            'IM1K3': 8.5,
            'IM2K3': 8.5,
            'IM3K3': 8.5,
            'IM3L0': 5.0
        }

        z_dict = {
            'IM1L0': 699.5576832,
            'IM2L0': 736.50848,
            'IM3L0': 746.0000167,
            'IM4L0': 753.5587416,
            'IM1K0': 699.4677942,
            'IM2K0': 732.3403281,
            'IM1K1': 738.0279162,
            'IM2K1': 742.15,
            'IM1K2': 777.93,
            'IM2K2': 780.425,
            'IM3K2': 781.9,
            'IM4K2': 783.455,
            'IM5K2': 787.417,
            'IM6K2': 792.167,
            'IM7K2': 798.5,
            'IM1L1': 745.4046250,
            'IM2L1': 759.02,
            'IM3L1': 778.96,
            'IM4L1': 778.96,
            'IM1K3': 740.804,
            'IM2K3': 750,
            'IM3K3': 778.66,
            'IM2K4': 755.32096,
            'IM3K4': 758.889,
            'IM4K4': 761.101,
            'IM5K4': 764.313
            #'IM5K4': 764.45591 - 0.03
        }

        try:
            self.distance = FOV_dict[self.epics_name[0:5]] * 1e3
            self.z = z_dict[self.epics_name[0:5]]
        except:
            self.distance = 8500.0
            self.z = z_dict['IM1L0']


        try:
            self.gige = PCDSAreaDetector(self.cam_name, name='gige')
            self.reset_camera()
        except Exception:
            print('\nSomething wrong with camera server')
            self.gige = None

        # load in pixel size
        try:
            with open('/cds/home/s/seaberg/Commissioning_Tools/PPM_centroid/imagers.db') as json_file:
                data = json.load(json_file)
           
            key_name = self.epics_name[0:5]
            if 'MONO' in self.epics_name:
                if '3' in self.epics_name:
                    key_name = 'MONO_03'
                elif '4' in self.epics_name:
                    key_name = 'MONO_04'

            imager_data = data[key_name]
            #imager_data = data[self.epics_name[0:5]]
            self.dx = float(imager_data['pixel'])
            self.distance = float(imager_data['FOV']) * 1e3
            self.z = float(imager_data['z'])

            try:
                self.cx_target = float(imager_data['cx'])
                self.cy_target = float(imager_data['cy'])
            except KeyError:
                self.cx_target = 0
                self.cy_target = 0

        except json.decoder.JSONDecodeError:
            self.dx = 5.5/1.2
            self.cx_target = 0.0
            self.cy_target = 0.0
        except KeyError:
            print('pixel size not calibrated. units are pixels.')
            self.dx = 1
            self.cx_target = 0.0
            self.cy_target = 0.0

        

        #self.cx_target = 0
        #self.cy_target = 0

        print(self.dx)

        # if len(sys.argv)>1:
        #     self.cam_name = sys.argv[1]
        #     self.epics_name = sys.argv[1] + 'IMAGE2:'

        self.image_pv = PV(self.epics_name + 'ArrayData')

        # get ROI info
        xmin = PV(self.epics_name + 'ROI:MinX_RBV').get()
        xmax = xmin + PV(self.epics_name + 'ROI:SizeX_RBV').get() - 1
        ymin = PV(self.epics_name + 'ROI:MinY_RBV').get()
        ymax = ymin + PV(self.epics_name + 'ROI:SizeY_RBV').get() - 1
        # get binning
        self.xbin = PV(self.epics_name + 'ROI:BinX_RBV').get()
        self.ybin = PV(self.epics_name + 'ROI:BinY_RBV').get()

        # get array size
        self.xsize = PV(self.epics_name + 'ROI:ArraySizeX_RBV').get()
        self.ysize = PV(self.epics_name + 'ROI:ArraySizeY_RBV').get()

        # pixel size in meters, per pixel so need to take binning into account
        self.dxm = self.dx * 1e-6 * self.xbin

        print(self.xsize)
        if self.xsize == 0:
            self.xsize = PV(self.epics_name + 'ArraySize0_RBV').get()
            self.ysize = PV(self.epics_name + 'ArraySize1_RBV').get()
            xmin = 0
            ymin = 0
            xmax = self.xsize - 1
            ymax = self.ysize - 1

        #self.x = np.linspace(0, self.xsize - 1, self.xsize, dtype=float)
        #self.x -= self.xsize/2
        #self.y = np.linspace(0, self.ysize - 1, self.ysize, dtype=float)
        #self.y -= self.ysize/2

        self.x = np.linspace(xmin, xmax - (self.xbin - 1), self.xsize, dtype=float)
        self.x -= (xmax + 1) / 2
        self.y = np.linspace(ymin, ymax - (self.ybin - 1), self.ysize, dtype=float)
        self.y -= (ymax + 1) / 2

        self.x *= self.dx
        self.y *= self.dx
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.x0 = np.copy(self.x)
        self.y0 = np.copy(self.y)

        print(self.epics_name)
        print(self.xsize)
        print(self.ysize)

        self.FOV = np.max(self.x) - np.min(self.x)

        self.N, self.M = np.shape(self.xx)

        self.profile = np.zeros_like(self.xx)
        self.x_lineout = np.zeros(self.M)
        self.y_lineout = np.zeros(self.N)
        self.x_projection = np.zeros(self.M)
        self.y_projection = np.zeros(self.N)
        if 'K' in self.epics_name:
            self.photon_energy = PV('PMPS:KFE:PE:UND:CurrentPhotonEnergy_RBV').get()
        else:
            self.photon_energy = PV('PMPS:LFE:PE:UND:CurrentPhotonEnergy_RBV').get()

        print('photon energy: %.2f' % self.photon_energy)
        self.lambda0 = 1239.8/self.photon_energy*1e-9
        self.time_stamp = 0.0
        self.cx = 0
        self.cy = 0
        self.wx = 0
        self.wy = 0
        self.intensity = 0

        f_x = np.linspace(-self.M / 2., self.M / 2. - 1., self.M) / self.M / self.dxm
        f_y = np.linspace(-self.N / 2., self.N / 2. - 1., self.N) / self.N / self.dxm

        self.f_x, self.f_y = np.meshgrid(f_x, f_y)

        self.downsample = 3

        self.Nd = int(self.N / (2 ** self.downsample))
        self.Md = int(self.M / (2 ** self.downsample))

        self.fit_object = None

        # load in dummy image
        self.dummy_image = np.load('/cds/home/s/seaberg/Commissioning_Tools/PPM_centroid/im2l0_sim.npy')

    def set_orientation(self, orientation):
        self.orientation = orientation

    def add_fit_object(self, fit_object):
        self.fit_object = fit_object

    def retrieve_wavefront(self, wfs, focusFOV=10, focus_z=0):
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

        # print('retrieving wavefront')

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Distance from wavefront sensor to PPM, 
        # including correction based on z stage
        zT = self.z - wfs.z - wfs.zPos()
        
        # include correction to f0 (distance between focus and grating)
        # based on z stage
        f0 = wfs.f0 + wfs.zPos()
        print('f0: %.3f' % f0)
        #print('zT: %.2f' % zT)

        # magnification of Talbot pattern
        mag = (zT + f0) / f0

        # number of pixels to sum across to get lineout
        lineout_width = int(wfs.pitch / self.dxm * 5 * mag)

        im1 = self.profile

        # expected spatial frequency of Talbot pattern (1/m)
        peak = 1. / mag / wfs.pitch

        fc = peak * self.dxm

        x_mask = ((self.f_x - fc / self.dxm) ** 2 + self.f_y ** 2) < (fc / 4 / self.dxm) ** 2
        x_mask = x_mask * (((self.f_x - fc / self.dxm) ** 2 + self.f_y ** 2) >
                           (fc / 4. / self.dxm - 2. / self.M / self.dxm) ** 2)
        x_mask = x_mask.astype(float)
        y_mask = ((self.f_x) ** 2 + (self.f_y - fc / self.dxm) ** 2) < (fc / 4 / self.dxm) ** 2
        y_mask = y_mask * (((self.f_x) ** 2 + (self.f_y - fc / self.dxm) ** 2) >
                           (fc / 4. / self.dxm - 2. / self.N / self.dxm) ** 2)
        y_mask = y_mask.astype(float)

        # parameters for calculating Legendre coefficients
        wfs_param = {
                "dg": wfs.pitch,  # wavefront sensor pitch (m)
                "fraction": fraction,  # wavefront sensor fraction
                "dx": self.dxm,  # PPM pixel size
                "zT": zT,  # distance between WFS and PPM
                "lambda0": self.lambda0,  # beam wavelength
                "downsample": 3, # Fourier downsampling power of 2
                "zf": f0  # nominal distance from focus to grating
                }

        talbot_image = TalbotImage(im1, fc, fraction)
        recovered_beam, wfs_param_out = talbot_image.get_legendre(self.fit_object, wfs_param, threshold=.1)

        # check validity
        # right now this is requiring that the peak is within half of the masked radius in the Fourier plane
        validity = ((np.abs(wfs_param_out['h_peak'] - peak) < (peak/8)) and
                    (np.abs(wfs_param_out['v_peak'] - peak) < (peak/8)))

        # check target is in
        target_in = 'TARGET' in wfs.check_state()

        # for now require that centroid data is also valid
        self.wavefront_is_valid = self.centroid_is_valid and validity and target_in

        wave = self.fit_object.wavefront_fit(wfs_param_out['coeff'])
        mask = np.abs(recovered_beam.wave[256 - int(self.Nd / 2):256 + int(self.Nd / 2),
                      256 - int(self.Md / 2):256 + int(self.Md / 2)]) > 0
        wave *= mask

        mask_x = mask[int(self.Nd / 2), :]
        mask_y = mask[:, int(self.Md / 2)]

        x_prime = recovered_beam.x[256, 256 - int(self.Md / 2):256 + int(self.Md / 2)] * 1e6
        y_prime = recovered_beam.y[256 - int(self.Nd / 2):256 + int(self.Nd / 2), 256] * 1e6
        x_prime = x_prime[mask_x]
        y_prime = y_prime[mask_y]
        x_res = wave[int(self.Nd / 2), :][mask_x]
        y_res = wave[:, int(self.Md / 2)][mask_y]
        # print('x_res: %d' % np.size(x_res))

        # going to try getting the third order Legendre polynomial here and try to get it to zero using benders
        try:
            leg_x = np.polynomial.legendre.legfit(x_prime*1e-6, x_res, 3)
            leg_y = np.polynomial.legendre.legfit(y_prime*1e-6, y_res, 3)
            coma_x = leg_x[3]
            coma_y = leg_y[3]
        except:
            self.wavefront_is_valid = False
            coma_x = 0
            coma_y = 0

        # setting rms_x/rms_y to third order Legendre coefficient for now.
        rms_x = np.std(x_res)
        rms_y = np.std(y_res)

        x_width = np.std(x_res)
        y_width = np.std(y_res)

        zf_x = -(recovered_beam.zx - zT - f0) * 1e3
        zf_y = -(recovered_beam.zy - zT - f0) * 1e3

        # annotated Fourier transform
        F0 = np.abs(wfs_param_out['F0'])

        F0 = F0 / np.max(F0)
        F0 += x_mask + y_mask

        # plane to propagate to relative to IP (focus_z is given in mm)
        z_plane = focus_z*1e-3

        # propagate to focus
        recovered_beam.beam_prop(-zT-f0 + z_plane)
        focus = recovered_beam.wave
        dx_focus = recovered_beam.dx
        dy_focus = recovered_beam.dy
        print('dx: %.2e' % dx_focus)
        print('dy: %.2e' % dy_focus)
        #focus = np.abs(focus)**2/np.max(np.abs(focus)**2)

        focus_PPM = PPM('focus', FOV=focusFOV*1e-6, N=256)
        focus_PPM.propagate(recovered_beam)
        
       
        focus = focus_PPM.profile/np.max(focus_PPM.profile)
        focus_horizontal = focus_PPM.x_lineout/np.max(focus_PPM.x_lineout)
        focus_vertical = focus_PPM.y_lineout/np.max(focus_PPM.y_lineout)
        focus_fwhm_horizontal = focus_PPM.wx
        focus_fwhm_vertical = focus_PPM.wy

        xf = focus_PPM.x * 1e6

        #x_focus = recovered_beam.x[0, :]
        #y_focus = recovered_beam.y[:, 0]
        #x_interp = np.linspace(-256, 255, 512, dtype=float)*focusFOV*1e-6/512
        #f = interpolation.interp2d(x_focus, y_focus, focus, fill_value=0)
        #focus = f(x_interp, x_interp)
        #focus_horizontal = np.sum(focus, axis=0)
        #focus_vertical = np.sum(focus, axis=1)



        # rms_x = np.std(x_res)
        # rms_y = np.std(y_res)

        # output. See method docstring for descriptions.
        wfs_data = {
                'x_res': x_res,
                'x_prime': x_prime,
                'y_res': y_res,
                'y_prime': y_prime,
                'z_x': zf_x,
                'z_y': zf_y,
                'rms_x': rms_x,
                'rms_y': rms_y,
                'coma_x': coma_x,
                'coma_y': coma_y,
                'F0': F0,
                'focus': focus,
                #'xf': x_interp*1e6,
                'xf': xf,
                'focus_fwhm_horizontal': focus_fwhm_horizontal,
                'focus_fwhm_vertical': focus_fwhm_vertical,
                'focus_horizontal': focus_horizontal,
                'focus_vertical': focus_vertical,
                'wave': wave,
                'dxf': dx_focus,
                'dyf': dy_focus
                }

        return wfs_data, wfs_param_out

    def retrieve_wavefront2D(self, basis_file, wfs):
        """
        Method to calculate wavefront in the case where there is a wavefront sensor upstream of the PPM.
        :param basis_file: string
            Path to file containing pickled Legendre basis object.
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

        # go ahead and retrieve 1D wavefront first
        wfs_data, wfs_param = self.retrieve_wavefront(wfs)

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Talbot image processing
        # load basis
        with open(basis_file, 'rb') as f:
            fit_object = pickle.load(f)

        # initialize Talbot image processing
        image_calc = TalbotImage(self.profile, wfs_param['fc'], fraction)

        # add parameters for calculating Legendre coefficients
        wfs_param['downsample'] = 3
        wfs_param['zf'] = f0
        wfs_param['dg'] = wfs.x_pitch_sim

        # calculate 2D legendre coefficients
        print('getting 2D Legendre coefficients')
        recovered_beam, fit_params = image_calc.get_legendre(fit_object, wfs_param)

        # get complete wavefront with defocus
        x = fit_params['x']
        y = fit_params['y']
        px = fit_params['px']
        py = fit_params['py']
        coeff = fit_params['coeff']

        # add defocus to wavefront fit
        full_wave = fit_object.wavefront_fit(coeff) + px * x**2 + py * y**2

        # output. See method docstring for descriptions.
        wfs_data2D = {
                'recovered': recovered_beam,
                'wave': full_wave
                }

        wfs_data.update(fit_params)

        wfs_data.update(wfs_data2D)

        return wfs_data

    def stop(self):
        self.running = False
        try:
            pass
            #self.gige.cam.acquire.put(0, wait=True)
        except AttributeError:
            pass

    def check_rate(self):
        rate = PV(self.cam_name+'ArrayRate_RBV').get()

        return rate

    def reset_camera(self):
        
        try:
            if self.check_rate()>0:
                print('camera is acquiring')
            else:
                print('resetting camera')
                self.gige.cam.acquire.put(0, wait=True)
                self.gige.cam.acquire.put(1)
        except:
            print('no camera')

    def get_dummy_image(self):
        return self.dummy_image

    def get_image(self, angle=0):
        try:
            # do averaging
            if hasattr(self, 'average'):
                numImages = getattr(self, 'average').get_numImages()
            else:
                numImages = 1
            image_data = self.image_pv.get_with_metadata()
            img = np.reshape(image_data['value'], (self.ysize, self.xsize)).astype(float)
            if numImages > 1:
                for i in range(numImages-1):
                    # wait for the next image
                    sleep(self.acquisition_period)
                    image_data = self.image_pv.get_with_metadata()
                    imgTemp = np.reshape(image_data['value'], (self.ysize, self.xsize)).astype(float)
                    img += imgTemp


            img = img/numImages

            time_stamp = image_data['timestamp']
            # time_stamp = image_data.time_stamp
            # img = np.array(image_data.shaped_image,dtype='float')
            # img = np.array(self.gige.image2.image,dtype='float')
            #img = Util.threshold_array(img, self.threshold)

            if self.orientation == 'action0':
                self.profile = img
                self.x = self.x0
                self.y = self.y0
            elif self.orientation == 'action90':
                self.profile = np.rot90(img)
                self.x = self.y0
                self.y = self.x0
            elif self.orientation == 'action180':
                self.profile = np.rot90(img,2)
                self.x = self.x0
                self.y = self.y0
            elif self.orientation == 'action270':
                self.profile = np.rot90(img,3)
                self.x = self.y0
                self.y = self.x0
            elif self.orientation == 'action0_flip':
                self.profile = np.fliplr(img)
                self.x = self.x0
                self.y = self.y0
            elif self.orientation == 'action90_flip':
                self.profile = np.rot90(np.fliplr(img))
                self.x = self.y0
                self.y = self.x0
            elif self.orientation == 'action180_flip':
                self.profile = np.rot90(np.fliplr(img),2)
                self.x = self.x0
                self.y = self.y0
            elif self.orientation == 'action270_flip':
                self.profile = np.rot90(np.fliplr(img),3)
                self.x = self.y0
                self.y = self.x0

            self.N = np.size(self.y)
            self.M = np.size(self.x)

            #print(self.M)
            #print(self.N)

            #angle = -0.2
            self.profile = ndimage.rotate(self.profile, angle, reshape=False)

            temp_profile = Util.threshold_array(self.profile, self.threshold)

            self.intensity = np.mean(temp_profile)
            self.projection_x = np.mean(temp_profile, axis=0)
            self.projection_y = np.mean(temp_profile, axis=1)

            # get beam statistics
            self.cx, self.cy, self.wx, self.wy, wx2, wy2 = self.beam_analysis(self.projection_x, self.projection_y)

            # add imager state to validity
            if 'MONO' in self.imager_prefix or 'SL' in self.imager_prefix:
                imager_state = 'YAG'
            else:
                imager_state = self.states_list[self.state.value]
            imager_in = 'YAG' in imager_state or 'DIAMOND' in imager_state

            self.centroid_is_valid = self.centroid_is_valid and imager_in

            x_center = Util.coordinate_to_pixel(self.cx, self.dx*self.xbin, self.M)
            y_center = Util.coordinate_to_pixel(self.cy, self.dx*self.ybin, self.N)

            #print(self.cx)
            #print(self.cy)

            #print(x_center)
            #print(y_center)

            try:
                self.lineout_x = temp_profile[int(y_center), :]
                self.lineout_y = temp_profile[:, int(x_center)]
            except:
                self.lineout_x = self.projection_x
                self.lineout_y = self.projection_y

            #print('got lineouts')

            # gaussian fits
            try:
                fit_x = self.amp_x * np.exp(
                    -(self.x - self.cx) ** 2 / 2 / (self.wx / 2.355) ** 2)
            except RuntimeWarning:
                fit_x = np.zeros_like(self.lineout_x)
            try:
                fit_y = self.amp_y * np.exp(
                    -(self.y - self.cy) ** 2 / 2 / (self.wy / 2.355) ** 2)
            except RuntimeWarning:
                fit_y = np.zeros_like(self.lineout_y)



            self.fit_x = fit_x
            self.fit_y = fit_y

            self.time_stamp = time_stamp

            return img, time_stamp
        except:
            self.lineout_x = np.zeros_like(self.x_lineout)
            self.lineout_y = np.zeros_like(self.y_lineout)
            print('no image')
            return np.zeros((2048, 2048))


class EXS_Device(PPM):
    """
    Child class of PPM that is used for a physical PPM, rather than simulated.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.imager_prefix = name
        self.threshold = 0.0001

        # set allowed kwargs
        allowed_arguments = ['average', 'threshold', 'fit_object']

        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        # get Y motor state
        self.state = EpicsSignalRO(self.imager_prefix + 'MMS:STATE:GET_RBV', auto_monitor=True)
        # define possible states depending on imager type
        if 'XTES' in self.imager_prefix:
            self.states_list = ['Unknown', 'OUT', 'YAG', 'DIAMOND', 'RETICLE']
        elif 'PPM' in self.imager_prefix:
            self.states_list = ['Unknown', 'OUT', 'POWERMETER', 'YAG1', 'YAG2']
        else:
            self.state_list = []

        self.cam_name = self.imager_prefix
        self.epics_name = self.cam_name + 'IMAGE3:'
        # get acquisition info (this is in seconds)
        self.acquisition_period = PV(self.epics_name[:-7] + 'AcquirePeriod_RBV').get()

        # check if Image3 is available
        port = PV(self.epics_name + 'PortName_RBV').get()
        array_rate = PV(self.epics_name + 'ROI:EnableCallbacks').get()

        if port is None or array_rate == 0:
            self.epics_name = self.imager_prefix + 'IMAGE1:'
            self.acquisition_period = PV(self.imager_prefix + 'CAM:AcquirePeriod_RBV').get()

        port = PV(self.epics_name + 'PortName_RBV').get()

        array_rate = PV(self.epics_name + 'ROI:EnableCallbacks').get()

        if port is None or array_rate == 0:
            self.epics_name = self.imager_prefix + 'CAM:IMAGE2:'
            self.acquisition_period = PV(self.imager_prefix + 'CAM:AcquirePeriod_RBV').get()

        port = PV(self.epics_name + 'PortName_RBV').get()

        if port is None:
            self.epics_name = self.imager_prefix + 'DATA1:'
            self.acquisition_period = PV(self.imager_prefix + 'AcquirePeriod_RBV').get()

        self.orientation = 'action0'

        print(self.epics_name)

        FOV_dict = {
            'IM2K4': 8.5,
            'IM3K4': 8.5,
            'IM4K4': 5.0,
            'IM5K4': 8.5,
            'IM6K4': 8.5,
            'IM1K1': 8.5,
            'IM2K1': 8.5,
            'IM1K2': 8.5,
            'IM2K2': 18.5,
            'IM3K2': 18.5,
            'IM4K2': 8.5,
            'IM5K2': 8.5,
            'IM6K2': 5.0,
            'IM7K2': 5.0,
            'IM1L1': 8.5,
            'IM2L1': 8.5,
            'IM3L1': 8.5,
            'IM4L1': 8.5,
            'IM1K3': 8.5,
            'IM2K3': 8.5,
            'IM3K3': 8.5,
            'IM3L0': 5.0
        }

        z_dict = {
            'IM1L0': 699.5576832,
            'IM2L0': 736.50848,
            'IM3L0': 746.0000167,
            'IM4L0': 753.5587416,
            'IM1K0': 699.4677942,
            'IM2K0': 732.3403281,
            'IM1K1': 738.0279162,
            'IM2K1': 742.15,
            'IM1K2': 777.93,
            'IM2K2': 780.425,
            'IM3K2': 781.9,
            'IM4K2': 783.455,
            'IM5K2': 787.417,
            'IM6K2': 792.167,
            'IM7K2': 798.5,
            'IM1L1': 745.4046250,
            'IM2L1': 759.02,
            'IM3L1': 778.96,
            'IM4L1': 778.96,
            'IM1K3': 740.804,
            'IM2K3': 750,
            'IM3K3': 778.66,
            'IM2K4': 755.32096,
            'IM3K4': 758.889,
            'IM4K4': 761.101,
            'IM5K4': 764.313
            # 'IM5K4': 764.45591 - 0.03
        }

        try:
            self.distance = FOV_dict[self.epics_name[0:5]] * 1e3
            self.z = z_dict[self.epics_name[0:5]]
        except:
            self.distance = 8500.0
            self.z = z_dict['IM1L0']

        try:
            self.gige = PCDSAreaDetector(self.cam_name, name='gige')
            self.reset_camera()
        except Exception:
            print('\nSomething wrong with camera server')
            self.gige = None

        # load in pixel size
        try:
            with open('/cds/home/s/seaberg/Python/slit_alignment/PPM_centroid/imagers.db') as json_file:
                data = json.load(json_file)

            imager_data = data[self.epics_name[0:5]]
            self.dx = float(imager_data['pixel'])
            self.distance = float(imager_data['FOV']) * 1e3
            self.z = float(imager_data['z'])

            try:
                self.cx_target = float(imager_data['cx'])
                self.cy_target = float(imager_data['cy'])
            except KeyError:
                self.cx_target = 0
                self.cy_target = 0

        except json.decoder.JSONDecodeError:
            print('file error')
            self.dx = 5.5 / 1.2
        except KeyError:
            print('pixel size not calibrated. units are pixels.')
            self.dx = 1

        self.cx_target = 0
        self.cy_target = 0

        print(self.dx)

        # if len(sys.argv)>1:
        #     self.cam_name = sys.argv[1]
        #     self.epics_name = sys.argv[1] + 'IMAGE2:'

        self.image_pv = PV(self.epics_name + 'ArrayData')

        # get ROI info
        xmin = PV(self.epics_name + 'ROI:MinX_RBV').get()
        xmax = xmin + PV(self.epics_name + 'ROI:SizeX_RBV').get() - 1
        ymin = PV(self.epics_name + 'ROI:MinY_RBV').get()
        ymax = ymin + PV(self.epics_name + 'ROI:SizeY_RBV').get() - 1
        # get binning
        self.xbin = PV(self.epics_name + 'ROI:BinX_RBV').get()
        self.ybin = PV(self.epics_name + 'ROI:BinY_RBV').get()

        # get array size
        self.xsize = PV(self.epics_name + 'ROI:ArraySizeX_RBV').get()
        self.ysize = PV(self.epics_name + 'ROI:ArraySizeY_RBV').get()

        # pixel size in meters, per pixel so need to take binning into account
        self.dxm = self.dx * 1e-6 * self.xbin

        print(self.xsize)
        if self.xsize == 0:
            self.xsize = PV(self.epics_name + 'ArraySize0_RBV').get()
            self.ysize = PV(self.epics_name + 'ArraySize1_RBV').get()
            xmin = 0
            ymin = 0
            xmax = self.xsize - 1
            ymax = self.ysize - 1

        # self.x = np.linspace(0, self.xsize - 1, self.xsize, dtype=float)
        # self.x -= self.xsize/2
        # self.y = np.linspace(0, self.ysize - 1, self.ysize, dtype=float)
        # self.y -= self.ysize/2

        self.x = np.linspace(xmin, xmax - (self.xbin - 1), self.xsize, dtype=float)
        self.x -= (xmax + 1) / 2
        self.y = np.linspace(ymin, ymax - (self.ybin - 1), self.ysize, dtype=float)
        self.y -= (ymax + 1) / 2

        self.x *= self.dx
        self.y *= self.dx
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        print(self.epics_name)
        print(self.xsize)

        self.FOV = np.max(self.x) - np.min(self.x)

        self.N, self.M = np.shape(self.xx)

        self.profile = np.zeros_like(self.xx)
        self.x_lineout = np.zeros(self.M)
        self.y_lineout = np.zeros(self.N)
        self.x_projection = np.zeros(self.M)
        self.y_projection = np.zeros(self.N)
        if 'K' in self.epics_name:
            self.photon_energy = PV('PMPS:KFE:PE:UND:CurrentPhotonEnergy_RBV').get()
        else:
            self.photon_energy = PV('PMPS:LFE:PE:UND:CurrentPhotonEnergy_RBV').get()

        print('photon energy: %.2f' % self.photon_energy)
        self.lambda0 = 1239.8 / self.photon_energy * 1e-9
        self.time_stamp = 0.0
        self.cx = 0
        self.cy = 0
        self.wx = 0
        self.wy = 0
        self.intensity = 0

        f_x = np.linspace(-self.M / 2., self.M / 2. - 1., self.M) / self.M / self.dxm
        f_y = np.linspace(-self.N / 2., self.N / 2. - 1., self.N) / self.N / self.dxm

        self.f_x, self.f_y = np.meshgrid(f_x, f_y)

        self.downsample = 3

        self.Nd = int(self.N / (2 ** self.downsample))
        self.Md = int(self.M / (2 ** self.downsample))

        self.fit_object = None

        # load in dummy image
        self.dummy_image = np.load('/cds/home/s/seaberg/Commissioning_Tools/PPM_centroid/im2l0_sim.npy')

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

        self.amp_x = np.max(line_x) - np.min(line_x)
        self.amp_y = np.max(line_y) - np.min(line_y)

        # normalize lineouts
        if np.max(line_x) > 0:
            line_x -= np.min(line_x)
            line_x = line_x / np.max(line_x)

        if np.max(line_y) > 0:
            line_y -= np.min(line_y)
            line_y = line_y / np.max(line_y)

        # set 20% threshold
        thresh_x = np.max(line_x) * self.threshold
        thresh_y = np.max(line_y) * self.threshold
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
            cx = np.sum(norm_x * self.x) / np.sum(norm_x)
            # calculate second moments. Converted to microns to help with fitting
            sx = np.sqrt(np.sum(norm_x * (self.x - cx) ** 2) / np.sum(norm_x)) * 1e6

        else:
            cx = 0
            sx = 0
        if np.sum(norm_y) > 0:
            cy = np.sum(norm_y * self.y) / np.sum(norm_y)
            # calculate second moments. Converted to microns to help with fitting
            sy = np.sqrt(np.sum(norm_y * (self.y - cy) ** 2) / np.sum(norm_y)) * 1e6

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
            px, pcovx = optimize.curve_fit(Util.fit_gaussian, self.x[mask] * 1e6, line_x[mask], p0=guessx)
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
            py, pcovy = optimize.curve_fit(Util.fit_gaussian, self.y[mask] * 1e6, line_y[mask], p0=guessy)
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
        validity = ((self.amp_x > 30) and (self.amp_y > 30) and fit_validity and
                    (fwhm_x < np.max(2 * self.x)) and (fwhm_y < np.max(2 * self.y)))

        self.centroid_is_valid = validity

        return cx, cy, fwhm_x, fwhm_y, fwx_guess, fwy_guess

    def set_orientation(self, orientation):
        self.orientation = orientation

    def add_fit_object(self, fit_object):
        self.fit_object = fit_object

    def retrieve_wavefront(self, wfs, focusFOV=10, focus_z=0):
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

        # print('retrieving wavefront')

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Distance from wavefront sensor to PPM,
        # including correction based on z stage
        zT = self.z - wfs.z - wfs.zPos()

        # include correction to f0 (distance between focus and grating)
        # based on z stage
        f0 = wfs.f0 + wfs.zPos()
        print('f0: %.3f' % f0)
        # print('zT: %.2f' % zT)

        # magnification of Talbot pattern
        mag = (zT + f0) / f0

        # number of pixels to sum across to get lineout
        lineout_width = int(wfs.pitch / self.dxm * 5 * mag)

        im1 = self.profile

        # expected spatial frequency of Talbot pattern (1/m)
        peak = 1. / mag / wfs.pitch

        fc = peak * self.dxm

        x_mask = ((self.f_x - fc / self.dxm) ** 2 + self.f_y ** 2) < (fc / 4 / self.dxm) ** 2
        x_mask = x_mask * (((self.f_x - fc / self.dxm) ** 2 + self.f_y ** 2) >
                           (fc / 4. / self.dxm - 2. / self.M / self.dxm) ** 2)
        x_mask = x_mask.astype(float)
        y_mask = ((self.f_x) ** 2 + (self.f_y - fc / self.dxm) ** 2) < (fc / 4 / self.dxm) ** 2
        y_mask = y_mask * (((self.f_x) ** 2 + (self.f_y - fc / self.dxm) ** 2) >
                           (fc / 4. / self.dxm - 2. / self.N / self.dxm) ** 2)
        y_mask = y_mask.astype(float)

        # parameters for calculating Legendre coefficients
        wfs_param = {
            "dg": wfs.pitch,  # wavefront sensor pitch (m)
            "fraction": fraction,  # wavefront sensor fraction
            "dx": self.dxm,  # PPM pixel size
            "zT": zT,  # distance between WFS and PPM
            "lambda0": self.lambda0,  # beam wavelength
            "downsample": 3,  # Fourier downsampling power of 2
            "zf": f0  # nominal distance from focus to grating
        }

        talbot_image = TalbotImage(im1, fc, fraction)
        recovered_beam, wfs_param_out = talbot_image.get_legendre(self.fit_object, wfs_param, threshold=.1)

        # check validity
        # right now this is requiring that the peak is within half of the masked radius in the Fourier plane
        validity = ((np.abs(wfs_param_out['h_peak'] - peak) < (peak / 8)) and
                    (np.abs(wfs_param_out['v_peak'] - peak) < (peak / 8)))

        # check target is in
        target_in = 'TARGET' in wfs.check_state()

        # for now require that centroid data is also valid
        self.wavefront_is_valid = self.centroid_is_valid and validity and target_in

        wave = self.fit_object.wavefront_fit(wfs_param_out['coeff'])
        mask = np.abs(recovered_beam.wave[256 - int(self.Nd / 2):256 + int(self.Nd / 2),
                      256 - int(self.Md / 2):256 + int(self.Md / 2)]) > 0
        wave *= mask

        mask_x = mask[int(self.Nd / 2), :]
        mask_y = mask[:, int(self.Md / 2)]

        x_prime = recovered_beam.x[256, 256 - int(self.Md / 2):256 + int(self.Md / 2)] * 1e6
        y_prime = recovered_beam.y[256 - int(self.Nd / 2):256 + int(self.Nd / 2), 256] * 1e6
        x_prime = x_prime[mask_x]
        y_prime = y_prime[mask_y]
        x_res = wave[int(self.Nd / 2), :][mask_x]
        y_res = wave[:, int(self.Md / 2)][mask_y]
        # print('x_res: %d' % np.size(x_res))

        # going to try getting the third order Legendre polynomial here and try to get it to zero using benders
        try:
            leg_x = np.polynomial.legendre.legfit(x_prime * 1e-6, x_res, 3)
            leg_y = np.polynomial.legendre.legfit(y_prime * 1e-6, y_res, 3)
            coma_x = leg_x[3]
            coma_y = leg_y[3]
        except:
            self.wavefront_is_valid = False
            coma_x = 0
            coma_y = 0

        # setting rms_x/rms_y to third order Legendre coefficient for now.
        rms_x = np.std(x_res)
        rms_y = np.std(y_res)

        x_width = np.std(x_res)
        y_width = np.std(y_res)

        zf_x = -(recovered_beam.zx - zT - f0) * 1e3
        zf_y = -(recovered_beam.zy - zT - f0) * 1e3

        # annotated Fourier transform
        F0 = np.abs(wfs_param_out['F0'])

        F0 = F0 / np.max(F0)
        F0 += x_mask + y_mask

        # plane to propagate to relative to IP (focus_z is given in mm)
        z_plane = focus_z * 1e-3

        # propagate to focus
        recovered_beam.beam_prop(-zT - f0 + z_plane)
        focus = recovered_beam.wave
        dx_focus = recovered_beam.dx
        dy_focus = recovered_beam.dy
        print('dx: %.2e' % dx_focus)
        print('dy: %.2e' % dy_focus)
        # focus = np.abs(focus)**2/np.max(np.abs(focus)**2)

        focus_PPM = PPM('focus', FOV=focusFOV * 1e-6, N=256)
        focus_PPM.propagate(recovered_beam)

        focus = focus_PPM.profile / np.max(focus_PPM.profile)
        focus_horizontal = focus_PPM.x_lineout / np.max(focus_PPM.x_lineout)
        focus_vertical = focus_PPM.y_lineout / np.max(focus_PPM.y_lineout)
        focus_fwhm_horizontal = focus_PPM.wx
        focus_fwhm_vertical = focus_PPM.wy

        xf = focus_PPM.x * 1e6

        # x_focus = recovered_beam.x[0, :]
        # y_focus = recovered_beam.y[:, 0]
        # x_interp = np.linspace(-256, 255, 512, dtype=float)*focusFOV*1e-6/512
        # f = interpolation.interp2d(x_focus, y_focus, focus, fill_value=0)
        # focus = f(x_interp, x_interp)
        # focus_horizontal = np.sum(focus, axis=0)
        # focus_vertical = np.sum(focus, axis=1)

        # rms_x = np.std(x_res)
        # rms_y = np.std(y_res)

        # output. See method docstring for descriptions.
        wfs_data = {
            'x_res': x_res,
            'x_prime': x_prime,
            'y_res': y_res,
            'y_prime': y_prime,
            'z_x': zf_x,
            'z_y': zf_y,
            'rms_x': rms_x,
            'rms_y': rms_y,
            'coma_x': coma_x,
            'coma_y': coma_y,
            'F0': F0,
            'focus': focus,
            # 'xf': x_interp*1e6,
            'xf': xf,
            'focus_fwhm_horizontal': focus_fwhm_horizontal,
            'focus_fwhm_vertical': focus_fwhm_vertical,
            'focus_horizontal': focus_horizontal,
            'focus_vertical': focus_vertical,
            'wave': wave,
            'dxf': dx_focus,
            'dyf': dy_focus
        }

        return wfs_data, wfs_param_out

    def retrieve_wavefront2D(self, basis_file, wfs):
        """
        Method to calculate wavefront in the case where there is a wavefront sensor upstream of the PPM.
        :param basis_file: string
            Path to file containing pickled Legendre basis object.
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

        # go ahead and retrieve 1D wavefront first
        wfs_data, wfs_param = self.retrieve_wavefront(wfs)

        # get Talbot fraction that we're using (fractional Talbot effect)
        fraction = wfs.fraction

        # Talbot image processing
        # load basis
        with open(basis_file, 'rb') as f:
            fit_object = pickle.load(f)

        # initialize Talbot image processing
        image_calc = TalbotImage(self.profile, wfs_param['fc'], fraction)

        # add parameters for calculating Legendre coefficients
        wfs_param['downsample'] = 3
        wfs_param['zf'] = f0
        wfs_param['dg'] = wfs.x_pitch_sim

        # calculate 2D legendre coefficients
        print('getting 2D Legendre coefficients')
        recovered_beam, fit_params = image_calc.get_legendre(fit_object, wfs_param)

        # get complete wavefront with defocus
        x = fit_params['x']
        y = fit_params['y']
        px = fit_params['px']
        py = fit_params['py']
        coeff = fit_params['coeff']

        # add defocus to wavefront fit
        full_wave = fit_object.wavefront_fit(coeff) + px * x ** 2 + py * y ** 2

        # output. See method docstring for descriptions.
        wfs_data2D = {
            'recovered': recovered_beam,
            'wave': full_wave
        }

        wfs_data.update(fit_params)

        wfs_data.update(wfs_data2D)

        return wfs_data

    def stop(self):
        self.running = False
        try:
            pass
            # self.gige.cam.acquire.put(0, wait=True)
        except AttributeError:
            pass

    def check_rate(self):
        rate = PV(self.cam_name + 'ArrayRate_RBV').get()

        return rate

    def reset_camera(self):

        try:
            if self.check_rate() > 0:
                print('camera is acquiring')
            else:
                print('resetting camera')
                self.gige.cam.acquire.put(0, wait=True)
                self.gige.cam.acquire.put(1)
        except:
            print('no camera')

    def get_dummy_image(self):
        return self.dummy_image

    def get_image(self, angle=0):
        try:
            # do averaging
            if hasattr(self, 'average'):
                numImages = getattr(self, 'average').get_numImages()
            else:
                numImages = 1

            image_data = self.image_pv.get_with_metadata()

            img = np.reshape(image_data['value'], (self.ysize, self.xsize)).astype(float)
            if numImages > 1:
                for i in range(numImages - 1):
                    # wait for the next image
                    sleep(self.acquisition_period)
                    image_data = self.image_pv.get_with_metadata()
                    imgTemp = np.reshape(image_data['value'], (self.ysize, self.xsize)).astype(float)
                    img += imgTemp

            img = img / numImages

            time_stamp = image_data['timestamp']
            # time_stamp = image_data.time_stamp
            # img = np.array(image_data.shaped_image,dtype='float')
            # img = np.array(self.gige.image2.image,dtype='float')
            # img = Util.threshold_array(img, self.threshold)

            if self.orientation == 'action0':
                self.profile = img
            elif self.orientation == 'action90':
                self.profile = np.rot90(img)
            elif self.orientation == 'action180':
                self.profile = np.rot90(img, 2)
            elif self.orientation == 'action270':
                self.profile = np.rot90(img, 3)
            elif self.orientation == 'action0_flip':
                self.profile = np.fliplr(img)
            elif self.orientation == 'action90_flip':
                self.profile = np.rot90(np.fliplr(img))
            elif self.orientation == 'action180_flip':
                self.profile = np.rot90(np.fliplr(img), 2)
            elif self.orientation == 'action270_flip':
                self.profile = np.rot90(np.fliplr(img), 3)

            # angle = -0.2
            self.profile = ndimage.rotate(self.profile, angle, reshape=False)

            temp_profile = Util.threshold_array(self.profile, self.threshold)

            self.intensity = np.mean(temp_profile)
            self.projection_x = np.mean(temp_profile, axis=0)
            self.projection_y = np.mean(temp_profile, axis=1)

            # get beam statistics
            self.cx, self.cy, self.wx, self.wy, wx2, wy2 = self.beam_analysis(self.projection_x, self.projection_y)

            # add imager state to validity
            # imager_state = self.states_list[self.state.value]
            # imager_in = 'YAG' in imager_state or 'DIAMOND' in imager_state
            # self.centroid_is_valid = self.centroid_is_valid and imager_in

            x_center = Util.coordinate_to_pixel(self.cx, self.dx * self.xbin, self.M)
            y_center = Util.coordinate_to_pixel(self.cy, self.dx * self.ybin, self.N)

            self.lineout_x = temp_profile[int(y_center), :]
            self.lineout_y = temp_profile[:, int(x_center)]

            # gaussian fits
            try:
                #fit_x = self.amp_x * np.sinc((self.x-self.cx)/self.wx)**2
                fit_x = self.amp_x * np.exp(
                    -(self.x - self.cx) ** 2 / 2 / (self.wx / 2.355) ** 2)
            except RuntimeWarning:
                fit_x = np.zeros_like(self.lineout_x)
            try:
                #fit_y  = self.amp_y * np.sinc((self.y - self.cy)/self.wy)**2
                fit_y = self.amp_y * np.exp(
                    -(self.y - self.cy) ** 2 / 2 / (self.wy / 2.355) ** 2)
            except RuntimeWarning:
                fit_y = np.zeros_like(self.lineout_y)

            self.fit_x = fit_x
            self.fit_y = fit_y

            self.time_stamp = time_stamp
            return img, time_stamp
        except:
            self.lineout_x = np.zeros_like(self.x_lineout)
            self.lineout_y = np.zeros_like(self.y_lineout)
            print('no image')
            return np.zeros((2048, 2048))


class WFS:
    """
    Class to represent Talbot wavefront sensor gratings/pinhole arrays.

    Attributes
    ----------

    """

    def __init__(self, name, **kwargs):
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
            For now this is limited to a checkerboard grating.
        :param enabled: bool
            If True, wavefront sensor influences the beam, otherwise it is effectively "moved out" of the beam.
        :param fraction: int
            Set to 1, 2, or 3 based on which Talbot fractional plane is being used.
        """

        # set attributes
        self.name = name
        self.pitch = None
        self.duty_cycle = 0.1
        self.f0 = 100
        self.z = None
        self.global_x = 0
        self.global_y = 0
        self.phase = False
        self.enabled = True
        self.fraction = 1
        self.azimuth = 0
        self.elevation = 0

        # set allowed kwargs
        allowed_arguments = ['pitch', 'duty_cycle', 'f0', 'z', 'phase', 'enabled',
                             'fraction']
        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        # initialize some calculated attributes
        self.x_pitch = 0.
        self.y_pitch = 0.
        self.x_pitch_sim = 0
        self.y_pitch_sim = 0
        self.grating = np.zeros(0)

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
        N, M = np.shape(beam.x)

        # Number of pixels per grating period
        self.x_pitch = np.round(self.pitch/beam.dx)
        if np.mod(self.x_pitch,2)!=0:
                self.x_pitch += 1
        self.x_pitch_sim = self.x_pitch*beam.dx
        self.y_pitch = np.round(self.pitch/beam.dy)
        if np.mod(self.y_pitch,2)!=0:
                self.y_pitch += 1
        self.y_pitch_sim = self.y_pitch*beam.dy
        print('actual pitch: %.2f microns' % (self.x_pitch*beam.dx*1e6))

        # print(self.pitch/beam.dx)
        # print(self.pitch/beam.dy)

        # re-initialize 1D gratings
        self.grating = np.zeros((N, M))

        # calculate number of periods in the grating
        Mg = np.floor(M / self.x_pitch)
        Ng = np.floor(N / self.y_pitch)

        # width of feature based on duty cycle
        x_width = int(self.x_pitch/2*self.duty_cycle)
        y_width = int(self.y_pitch/2*self.duty_cycle)

        # loop through periods in the grating
        # for i in range(int(Mg)):
        #     for j in range(int(Ng)):
        #         minY = int(self.y_pitch) * (j+1) - y_width
        #         maxY = int(self.y_pitch) * (j + 1) + y_width
        #         minX = int(self.x_pitch) * (i + 1) - x_width
        #         maxX = int(self.x_pitch) * (i + 1) + x_width
        #         self.grating[minY:maxY, minX:maxX] = (1 + (-1)**(i+j) / 2)

        # convert to checkerboard pi phase grating if desired
        if self.phase:

            # loop through periods in the grating
            for i in range(int(Mg)):
                for j in range(int(Ng)):
                    minY = int(self.y_pitch) * (j + 1) - y_width
                    maxY = int(self.y_pitch) * (j + 1) + y_width
                    minX = int(self.x_pitch) * (i + 1) - x_width
                    maxX = int(self.x_pitch) * (i + 1) + x_width
                    self.grating[minY:maxY, minX:maxX] = (1 + (-1) ** (i + j) / 2)
            self.grating = np.exp(1j*np.pi*self.grating)
        # otherwise make a pinhole array
        else:
            # loop through periods in the grating
            for i in range(int(Mg)):
                for j in range(int(Ng)):
                    minY = int(self.y_pitch) * (j + 1) - y_width
                    maxY = int(self.y_pitch) * (j + 1) + y_width
                    minX = int(self.x_pitch) * (i + 1) - x_width
                    maxX = int(self.x_pitch) * (i + 1) + x_width
                    self.grating[minY:maxY, minX:maxX] = 1
        # multiply beam by grating
        beam.wave *= self.grating


class WFS_Device(WFS):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        # set allowed kwargs
        allowed_arguments = ['state']

        # update attributes based on kwargs
        for key, value in kwargs.items():
            if key in allowed_arguments:
                setattr(self, key, value)

        self.epics_name = self.name + ':WFS:'

        self.state = EpicsSignalRO(self.epics_name+'MMS:STATE:GET_RBV', auto_monitor=True)
        self.z_motor = EpicsSignalRO(self.epics_name+'MMS:Z.RBV', auto_monitor=True)
        self.states_list = ['Unknown', 'OUT', 'TARGET1', 'TARGET2', 'TARGET3', 'TARGET4', 'TARGET5']

        # define z offset (in mm)
        self.z_offset = 31.0

        pitch_dict = {
            'PF1K0': [39.6, 41],
            'PF1L0': [28.4, 29.9, 31.7, 33.9, 36.6],
            'PF1K4': [35.8, 35.8, 35.8, 35.8, 34.6],
            'PF2K4': [33.3],
            'PF1K2': [32, 36.4]
        }

        z_dict = {
            'PF1K0': 731.5855078,
            'PF1L0': 735.6817413,
            'PF1K4': 763.515,
            #'PF1K4': 763.66694 - .0093,
            'PF2K4': 768.583,
            'PF1K2': 786.918,
        }

        f0_dict = {
            'PF1K0': 100,
            'PF1L0': 100,
            #'PF1K4': 1.772,
            'PF1K4': 1.768,
            #'PF1K4': 763.66694-.0093 - 761.89013,
            'PF2K4': 0.996,
            'PF1K2': 1.668
        }

        #state_rbv = PV(self.epics_name + 'MMS:STATE:GET_RBV').get()
        #self.z_pv = EpicsSignalRO(self.epics_name+'MMS:Z', name='omitted')

        # state 0 is OUT, need to subtract 1 to align with target positions
        state = self.state.value - 2
        #state = self.state.value
        print(state)
        # for testing purposes we will set OUT to state 0
        if state < 0:
            state = 0

        try:
            # account for Talbot fraction here
            self.pitch = pitch_dict[self.name][state] * 1.e-6
            print(self.pitch)
            self.pitch *= 1/self.fraction
            self.z = z_dict[self.name]
            self.f0 = f0_dict[self.name]
            print('pitch: %.4e' % self.pitch)
        except:
            self.pitch = 30.0
            self.z = z_dict['PF1L0']
            self.f0 = 100

    def check_state(self):

        return self.states_list[self.state.value]

    def zPos(self):
        # z position in meters
        zPos = (self.z_motor.value - self.z_offset)*1e-3
        return zPos

class PhasePlate:
    """
    Attributes
    ----------
    name: str
        Name of the device (e.g. CRL1)
    plateThickness: float
        Thickness profile of the phase plate. (meters)
    x_plate: float
        Phase plate size in x. (meters)
    y_plate: float
        Phase plate size in y. (meters)
    E0: float or None
        photon energy in eV for calculating the corresponding phase difference of a given thickness
    material: str
        Phase plate material. Currently only Be is implemented but may add CVD diamond in the future.
        Looks up downloaded data from CXRO.
    dx: float
        Phase plate de-centering along beam's x-axis.
    dy: float
        Phase plate de-centering along beam's y-axis.
    z: float
        z location of phase plate along beamline.
    energy: (N,) ndarray
        List of photon energies from CXRO file (eV).
    delta: (N,) ndarray
        Real part of index of refraction. n = 1 - delta + 1j * beta
    beta: (N,) ndarray
        Imaginary part of index of refraction. n = 1 - delta + 1j * beta
    """

    def __init__(self, name, platePhase=None, x_plate=None, y_plate=None, E0=None, z=0, dx=0, dy=0, orientation=0):
        """
        Method to create a PhasePlate object.
        :param name: str
            Name of the device (e.g. Phase1)
        :param plateThickness: float
            Thickness profile of the phase plate. (meters)
        :x_plate: float
            Phase plate size in x. (meters)
        :y_plate: float
            Phase plate size in y. (meters)
        :param E0: float
            photon energy for calculating radius of curvature for a given focal length (eV)
        :param material: str
            Lens material. Currently only Be is implemented but may add CVD diamond in the future.
            Looks up downloaded data from CXRO.
        :param z: float
            z location of lenses along beamline.
        :param dx, dy: float
            PhasePlate de-centering along beam's x,y-axis.
        :param orientation: int
            Whether or not this is a horizontal or vertical lens (0 for horizontal, 1 for vertical).
        """

        # set some attributes
        self.name = name
        self.platePhase = platePhase
        self.x_plate = x_plate
        self.y_plate = y_plate
        self.E0 = E0
        self.orientation = orientation
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
        self.x_intersect = 0
        self.y_intersect = 0
        self.z_intersect = 0

    def multiply(self, beam):
        """
        Method to propagate beam through PhasePlate
        :param beam: Beam
            Beam object to propagate through PhasePlate. Beam is modified by this method.
        :return: None
        """

        beam_shift = np.array([self.x_intersect - self.global_x,
                               self.y_intersect - self.global_y,
                               self.z_intersect - self.z])
        x_shift = np.dot(beam_shift, self.xhat)
        y_shift = np.dot(beam_shift, self.yhat)

        # get shape of phase plate thickness
        plate_shape = np.shape(self.platePhase)

        Ns = 0
        Ms = 0
        beamx = beam.x + x_shift
        beamy = beam.y + y_shift

        if len(plate_shape)>1:
            Ns = plate_shape[0]
            Ms = plate_shape[1]

            central_line_x = self.platePhase[np.int(Ns / 2), :]
            central_line_y = self.platePhase[:, np.int(Ms / 2)]

        else:
            if self.orientation==0:
                central_line_x = self.platePhase
                central_line_y = None
                Ms = plate_shape[0]
                Ns = 0
            elif self.orientation==1:
                central_line_x = None
                central_line_y = self.platePhase
                Ns = plate_shape[0]
                Ms = 0

        xs = np.linspace(-Ms / 2, Ms / 2 - 1, Ms) * self.x_plate / Ms  # phase plate x coordinate
        ys = np.linspace(-Ns / 2, Ns / 2 - 1, Ns) * self.y_plate / Ns  # phase plate y coordinate

        # interpolation onto beam coordinates
        if central_line_x is not None:
            phase_x = np.interp(beamx - self.dx, xs, central_line_x, left=0, right=0)
            phase_x = np.tile(phase_x, (beam.N, 1))
        else:
            phase_x = np.zeros_like(beamx)
        if central_line_y is not None:
            phase_y = np.interp(beamy - self.dy, ys, central_line_y, left=0, right=0)
            phase_y = np.tile(phase_y, (1, beam.M))
        else:
            phase_y = np.zeros_like(beamy)

        # transmission based on beta and thickness profile
        mask_x = (((beamx - self.dx) ** 2) < (self.x_plate / 2) ** 2).astype(float)
        mask_y = (((beamy - self.dy) ** 2) < (self.y_plate / 2) ** 2).astype(float)
        mask_full = mask_x*mask_y

        transmission = np.exp(1j * phase_x) * np.exp(1j * phase_y) * mask_full
        # transmission_y = np.exp(1j * phase_y) * mask_y


        beam.wave *= transmission
        # beam.zx = 100000
        # beam.wavey *= transmission_y

    def propagate(self, beam):
        """
        Method to propagate beam through PhasePlate. Calls multiply.
        :param beam: Beam
            Beam object to propagate through PhasePlate. Beam is modified by this method.
        :return: None
        """
        if self.platePhase is not None:
            self.multiply(beam)