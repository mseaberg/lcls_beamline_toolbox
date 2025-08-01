"""
beamline2d module

Part of the xraybeamline2d package.

Interface between beam and optics modules.

Currently implements the following classes:
Beamline: stores list of optics devices, and interfaces with Beam to propagate through beamline sections.
"""

from .optics1d import Drift, Mono, Mirror, Grating, TransmissionGrating, PPM
from . import optics1d as optics
from ..utility import util
# import matplotlib.pyplot as plt
import copy
import json
import numpy as np
import pandas as pd
from lcls_beamline_toolbox.utility.util import Util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Beamline:
    """
    Class to store various optics devices and handle beam propagation across the beamline.

    Attributes
    ----------
    device_list: list of optics objects (see optics module)
        A list of all the devices contained along a given beamline (excluding Drifts).
    full_list: list of optics devices (see optics module)
        A list of all the devices contained along a given beamline (including Drifts).
    """

    def __init__(self, device_list=[], RTD_filename=None, photon_source='L0', ordered=False, suppress=True, x_offset=0.0, y_offset=0.0):
        """
        Initialize a Beamline object.
        :param device_list: list of optics objects (see optics module)
            A list of all the devices contained along a given beamline (excluding Drifts).
        """

        self.suppress = suppress
        # set attribute for whether devices are in correct order
        self.ordered = ordered
        self.x_offset = x_offset
        self.y_offset = y_offset

        if RTD_filename is not None:
            self.device_list = self.read_RTD_file(RTD_filename, photon_source)
            # initialize full array without drifts
            self.full_list = self.device_list.copy()
            self.add_drifts_RTD()
        else:
            # set device_list as an attribute
            self.device_list = device_list

            # initialize full array without drifts
            self.full_list = self.device_list.copy()

            # calculate drifts and add them to full_list
            self.add_drifts()

        # set devices to attributes of self
        for device in self.device_list:
            setattr(self, device.name, device)

    def add_drifts(self):
        """
        Method to calculate drift sections. Creates a bunch of Drift objects and adds them to self.full_list.
        :return: None
        """

        if not self.ordered:
            # sort device list based on z
            self.device_list.sort(key=lambda device: device.z)

        # initialize drift list
        drift_list = []

        dummy_device = PPM('dummy', z=self.device_list[0].z-1, N=1)

        # initialize coordinates
        x = np.copy(self.x_offset)
        y = np.copy(self.y_offset)
        # x = 0
        # y = 0

        dummy_device.global_x = np.copy(x)
        dummy_device.global_y = np.copy(y)

        self.device_list.insert(0, dummy_device)

        # initialize angles
        elevation = 0
        azimuth = 0

        # beam direction
        k = Util.get_k(elevation, azimuth)

        xhat = np.array([1, 0, 0])
        yhat = np.array([0, 1, 0])
        zhat = np.array([0, 0, 1])

        k = np.copy(zhat)

        # initialize drift number
        i = 0
        # initialize previous device
        prev_device = None
        for device in self.device_list:
            # don't need any drifts upstream of first device
            if i > 0:

                # z difference between devices
                dz = device.z - prev_device.z

                # figure out device location
                x += k[0] / k[2] * dz
                y += k[1] / k[2] * dz
                # x += k[0] * dz
                # y += k[1] * dz
                # update device
                device.global_x = np.copy(x)
                device.global_y = np.copy(y)
                # device.global_x = x
                # device.global_y = y

                # set drift name
                name = 'drift%d' % i
                if isinstance(prev_device, Mono):
                    prev_device = prev_device.grating
                drift_list.append(Drift(name, upstream_component=prev_device,
                                        downstream_component=device))
            # else:
            #     device.global_x = np.copy(x)
            #     device.global_y = np.copy(y)

            # print(device.global_x)

            # update global orientation
            if issubclass(type(device), Mirror):
                # update global alpha
                if device.orientation == 0:
                    device.normal, device.sagittal, device.tangential = Util.rotate_3d_trace(xhat,yhat,zhat,
                                                                                       delta=device.alpha+device.delta)

                    xhat,yhat,zhat = Util.rotate_3d_trace(xhat,yhat,zhat,delta=device.alpha+device.beta0)
                    # device.global_alpha = device.alpha + azimuth
                    # azimuth += device.alpha + device.beta0
                    # print('after %s: %.4f' % (device.name, azimuth))
                elif device.orientation == 1:
                    sagittal, normal, tangential = Util.rotate_3d_trace(xhat,yhat,zhat,
                                                                                       delta=-device.alpha-device.delta,
                                                                                       dir='elevation')
                    device.sagittal = -sagittal
                    device.normal = normal
                    device.tangential = tangential
                    xhat,yhat,zhat = Util.rotate_3d_trace(xhat,yhat,zhat,delta=-device.alpha-device.beta0,dir='elevation')
                    # device.global_alpha = device.alpha + elevation
                    # elevation += device.alpha + device.beta0
                    # print('after %s: %.4f' % (device.name, elevation))
                elif device.orientation == 2:
                    normal, sagittal, tangential = Util.rotate_3d_trace(xhat, yhat, zhat,
                                                                                       delta=-device.alpha-device.delta)
                    device.normal = -normal
                    device.sagittal = -sagittal
                    device.tangential = tangential

                    xhat, yhat, zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=-device.alpha - device.beta0)

                    # device.global_alpha = azimuth - device.alpha
                    # azimuth -= (device.alpha + device.beta0)
                    # print('after %s: %.4f' % (device.name, azimuth))

                elif device.orientation == 3:
                    sagittal, normal, tangential = Util.rotate_3d_trace(xhat, yhat, zhat,
                                                                                       delta=device.alpha+device.delta,
                                                                                       dir='elevation')
                    device.sagittal = sagittal
                    device.normal = -normal
                    device.tangential = tangential
                    xhat, yhat, zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=device.alpha + device.beta0,
                                                      dir='elevation')

                    # device.global_alpha = elevation - device.alpha
                    # elevation -= (device.alpha + device.beta0)
                    # print('after %s: %.4f' % (device.name, elevation))

                # update k
                k = np.copy(zhat)
            else:
                device.xhat = xhat
                device.yhat = yhat
                device.zhat = zhat

                if issubclass(type(device), TransmissionGrating):
                    if device.orientation==0:
                        xhat, yhat, zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=device.beta0)
                    elif device.orientation==1:
                        xhat, yhat, zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=device.beta0, dir='elevation')

                k = np.copy(zhat)
                # device.azimuth = azimuth
                # device.elevation = elevation
            # update previous device
            prev_device = device
            # increment drift number
            i += 1

        self.full_list.insert(0, dummy_device)

        if not self.ordered:
            # add drifts to full_list
            self.full_list.extend(drift_list)

            # sort list based on z
            self.full_list.sort(key=lambda device: device.z)
        else:
            # keep everything in the same order, interleave drifts in between devices
            for num, drift in enumerate(drift_list):
                self.full_list.insert(2*num+1, drift)

    def write_json(self, filename):
        file_dict = {}
        for device in self.device_list:
            if issubclass(type(device), Mirror):
                xhat = device.normal
                yhat = device.sagittal
                zhat = device.tangential

            else:
                xhat = device.xhat
                yhat = device.yhat
                zhat = device.zhat
            device_dict = {'Z': device.z,
                           'X': device.global_x,
                           'Y': device.global_y,
                           'xhat': xhat.tolist(),
                           'yhat': yhat.tolist(),
                           'zhat': zhat.tolist()}
            file_dict[device.name] = device_dict

        json_object = json.dumps(file_dict, indent=4)
        with open(filename,'w') as f:
            f.write(json_object)

    def write_beamline(self,filename):

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('Device Name, Z, X, Y, Transformation matrix\n')
            for device in self.device_list:
                if issubclass(type(device), Mirror):
                    xhat = device.normal
                    yhat = device.sagittal
                    zhat = device.tangential

                else:
                    xhat = device.xhat
                    yhat = device.yhat
                    zhat = device.zhat

                f.write(
                    '{}, {:.10e}, {:.10e}, {:.10e}'.format(device.name, device.z, device.global_x, device.global_y))
                f.write(',{:.10e},{:.10e},{:.10e}\n'.format(xhat[0],xhat[1],xhat[2]))
                f.write(',,,,{:.10e},{:.10e},{:.10e}\n'.format(yhat[0],yhat[1],yhat[2]))
                f.write(',,,,{:.10e},{:.10e},{:.10e}\n'.format(zhat[0],zhat[1],zhat[2]))

    def add_drifts_RTD(self):
        """
        Method to calculate drift sections. Creates a bunch of Drift objects and adds them to self.full_list.
        :return: None
        """

        if not self.ordered:
            # sort device list based on z
            self.device_list.sort(key=lambda device: device.z)

        # initialize drift list
        drift_list = []

        dummy_device = PPM('dummy', z=self.device_list[0].z-1, N=1)

        dummy_device.global_x = np.copy(self.device_list[0].global_x)
        dummy_device.global_y = np.copy(self.device_list[0].global_y)
        dummy_device.xhat = np.copy(self.device_list[0].xhat)
        dummy_device.yhat = np.copy(self.device_list[0].yhat)
        dummy_device.zhat = np.copy(self.device_list[0].zhat)

        self.device_list.insert(0, dummy_device)

        # initialize drift number
        i = 0
        # initialize previous device
        prev_device = None
        for device in self.device_list:
            # don't need any drifts upstream of first device
            if i > 0:

                # set drift name
                name = 'drift%d' % i
                if isinstance(prev_device, Mono):
                    prev_device = prev_device.grating
                drift_list.append(Drift(name, upstream_component=prev_device,
                                        downstream_component=device))

            # update previous device
            prev_device = device
            # increment drift number
            i += 1

        self.full_list.insert(0, dummy_device)

        if not self.ordered:
            # add drifts to full_list
            self.full_list.extend(drift_list)

            # sort list based on z
            self.full_list.sort(key=lambda device: device.z)
        else:
            # keep everything in the same order, interleave drifts in between devices
            for num, drift in enumerate(drift_list):
                self.full_list.insert(2 * num + 1, drift)

    def read_RTD_file(self, filename, photon_source):

        ux = np.array([1,0,0])
        uy = np.array([0,1,0])
        uz = np.array([0,0,1])

        df = pd.read_csv(filename)
        df = df[df['Beamlines'].astype(str).str.contains(photon_source)]

        device_list = []

        for index, row in df.iterrows():
            name = str(row['FC'])
            fungible = str(row['Fungible'])
            x_size = row['X_dim']
            y_size = row['Y_dim']
            z_size = row['Z_dim']
            x = row['LCLS_X_loc']
            y = row['LCLS_Y_loc']
            z = row['LCLS_Z_loc']
            roll = row['LCLS_Z_roll']
            pitch = row['LCLS_X_pitch']
            yaw = row['LCLS_Y_yaw']
            inc_angle = row['inc_angle']

            xhat = np.copy(ux)
            yhat = np.copy(uy)
            zhat = np.copy(uz)

            xhat,yhat,zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=pitch, dir='elevation')
            xhat,yhat,zhat = Util.rotate_3d_trace(xhat, yhat, zhat, delta=yaw, dir='azimuth')

            device = None

            if (name[:2] == 'IM') or ('IP' in name):
                if 'PPM' in fungible or 'profile' in fungible.lower() or 'IP' in fungible:
                    print('found PPM {}'.format(name))
                    device = PPM(name, FOV=x_size, z=z, aspect_ratio=y_size/x_size)
            elif name[:2] == 'MR':

                orientation = int(roll/(np.pi/2))+1
                if orientation > 3:
                    orientation = 0

                if 'KB' in fungible:
                    print('found mirror {}'.format(name))
                    p = row['p nominal']
                    q = row['q nominal']
                    device = optics.CurvedMirror(name, p=p, q=q, length=z_size, orientation=orientation,
                                            alpha=inc_angle, z=z)
                elif 'HOMS' in fungible:
                    print('found mirror {}'.format(name))
                    device = optics.Mirror(name, length=z_size, orientation=orientation, alpha=inc_angle,
                                      z=z)

            # elif name[:2] == 'PC':
            #     print('found collimator {}'.format(name))
            #     device = optics.Collimator(name, diameter=x_size, z=z)
            elif name[:2]=='SL' and 'slit' in fungible.lower():
                print('found slit {}'.format(name))
                device = optics.Slit(name, x_width=x_size, y_width=y_size, z=z)

            if issubclass(type(device), Mirror):
                if device.orientation==0:
                    normal = np.copy(ux)
                    sagittal = np.copy(uy)
                    tangential = np.copy(uz)
                elif device.orientation==1:
                    normal = np.copy(uy)
                    sagittal = -np.copy(ux)
                    tangential = np.copy(uz)
                elif device.orientation==2:
                    normal = -np.copy(ux)
                    sagittal = -np.copy(uy)
                    tangential = np.copy(uz)
                elif device.orientation==3:
                    normal = -np.copy(uy)
                    sagittal = np.copy(ux)
                    tangential = np.copy(uz)
                device.normal = normal
                device.sagittal = sagittal
                device.tangential = tangential
                device.global_x = x
                device.global_y = y
                Util.rotate_about_point(device, Util.get_pos(device), pitch * ux)
                Util.rotate_about_point(device, Util.get_pos(device), yaw * uy)
            elif device is not None:
                device.global_x = x
                device.global_y = y
                device.xhat = xhat
                device.yhat = yhat
                device.zhat = zhat

            if device is not None:
                device_list.append(device)

        return device_list

    def write_RTD_file(self, filename):
        pass

    def load_beamline(self,filename):
        with open(filename,'r') as f:
            device_info = json.load(f)

        for device in self.device_list:
            device_data = device_info[device.name]
            device.z = float(device_data['Z'])
            device.global_x = float(device_data['X'])
            device.global_y = float(device_data['Y'])
            if issubclass(type(device), Mirror):
                device.normal = np.array(device_data['xhat'])
                device.sagittal = np.array(device_data['yhat'])
                device.tangential = np.array(device_data['zhat'])
            else:
                device.xhat = np.array(device_data['xhat'])
                device.yhat = np.array(device_data['yhat'])
                device.zhat = np.array(device_data['zhat'])

    def draw_beamline(self,figsize=None,ax=None,scatter=True,fig=None):

        if ax is None:
            if fig is None:
                if figsize is not None:
                    fig = plt.figure(figsize=figsize)
                else:
                    fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        # collect vertices
        xs = np.zeros(len(self.device_list))
        ys = np.zeros(len(self.device_list))
        zs = np.zeros(len(self.device_list))
        for num, device in enumerate(self.device_list):
            xs[num] = device.global_x
            ys[num] = device.global_y
            zs[num] = device.z

        vertices = ax.plot(zs,xs,zs=ys)
        if scatter:
            ax.scatter(zs,xs,zs=ys)

        xb = np.zeros(len(self.device_list)-1)
        yb = np.zeros_like(xb)
        zb = np.zeros_like(xb)

        for num, device in enumerate(self.device_list[1:]):
            xb[num] = device.x_intersect
            yb[num] = device.y_intersect
            zb[num] = device.z_intersect

        devices = ax.plot(zb, xb, zs=yb)
        if scatter:
            ax.scatter(zb, xb, zs=yb)

        patches = []
        dirs = []
        mirror_z = []

        verts = []
        for num, device in enumerate(self.device_list):
            # ax.text(zs[num],xs[num],ys[num],device.name, 'y')
            if issubclass(type(device), Mirror):
                # if (device.orientation==0) or (device.orientation==2):
                #     # x = device.z-device.length/2
                #     # y = device.global_x
                #     # mirror_z.append(device.global_y)
                #     # dirs.append('x')
                #     # mirror_patch = Rectangle((x,y),device.length,device.width)
                #     # patches.append(mirror_patch)
                #
                #     x1 = device.z-(device.length/2 * np.cos(device.global_alpha))
                #     x2 = device.z + (device.length/2 * np.cos(device.global_alpha))
                #     y1 = device.global_x - device.length/2*np.sin(device.global_alpha)
                #     y2 = device.global_x + device.length/2*np.sin(device.global_alpha)
                #     z1 = device.global_y-device.width/2
                #     z2 = device.global_y+device.width/2
                #     verts.append([(x1,y1,z1),(x2,y2,z1),(x2,y2,z2),(x1,y1,z2)])
                # else:
                #     x1 = device.z - (device.length / 2 * np.cos(device.global_alpha))
                #     x2 = device.z + (device.length / 2 * np.cos(device.global_alpha))
                #     y1 = device.global_x - device.width / 2
                #     y2 = device.global_x + device.width / 2
                #     z1 = device.global_y - device.length / 2*np.sin(device.global_alpha)
                #     z2 = device.global_y + device.length / 2*np.sin(device.global_alpha)
                #     verts.append([(x1, y1, z1), (x2, y1, z2), (x2, y2, z2), (x1, y2, z1)])

                s = np.array([device.sagittal[2],device.sagittal[0],device.sagittal[1]])
                t = np.array([device.tangential[2],device.tangential[0],device.tangential[1]])

                point = np.array([device.z,device.global_x,device.global_y])
                verts.append([tuple(point+t*device.length/2+s*device.width/2),
                          tuple(point+t*device.length/2-s*device.width/2),
                          tuple(point-t*device.length/2-s*device.width/2),
                          tuple(point-t*device.length/2+s*device.width/2)])

        col = Poly3DCollection(verts,facecolors=['r'],edgecolors=((0,0,0,1)))

        # ax.set_ylim(np.min(ys),np.max(ys))
        # ax.set_zlim(np.min(zs),np.max(zs))
        # ax.set_xlim(np.min(xs), np.max(xs))
        # ax.set_ylim(np.min(ys),np.max(ys))
        # ax.set_zlim(np.min(zs),np.max(zs))
        # print(len(patches))
        # coll = PatchCollection([patches[0]])
        ax.add_collection3d(col)

        artists = [col,vertices,devices]

        return ax, zs, artists

    def update_devices(self):
        """
        Method to re-calculate drifts if device positions changed
        :return: None
        """

        # re-initialize full array without drifts
        self.full_list = self.device_list.copy()

        # re-calculate drifts
        self.add_drifts()

    def adjust_device(self, device_name, axis, adjustment):
        """
        Method to adjust position/motor of a device in the beamline
        :param device_name: str
            name of the device
        :param axis: str
            motor name
        :param adjustment: float
            Relative motor motion
        :return: None
        """
        
        # get device from device name
        device = getattr(self, device_name)
        # make adjustment
        device.adjust_motor(axis, adjustment)
        
        # if device moved in z, drifts need to be re-calculated
        if axis == 'z':
            self.update_devices()

    def first_device(self):
        """
        Method to return beamline's first device.
        :return: optics object
            First optics object along the beamline
        """
        
        return self.full_list[0]

    def propagate_beamline(self, beam_in):
        """
        Method to propagate a beam along all the devices on the beamline.
        :param beam_in: Beam
            Beam input to upstream end of beamline.
        :return beam: Beam
            Beam output from downstream end of beamline.
        """
        # make a full copy of the beam input so that we don't modify the input
        beam = copy.deepcopy(beam_in)

        # distance between source and first device
        dz = self.full_list[0].z - beam.z_source
        beam.reinitialize(dz)

        # loop through all devices including drifts
        for device in self.full_list:
            # print name
            if not self.suppress:
                print('\033[1m' +device.name+'\033[0m')

            # propagate through device. beam is modified directly.
            success = device.propagate(beam)
            # print some beam info
            if not self.suppress:
                print('zx: %.6f' % beam.zx)
                print('zy: %.6f' % beam.zy)
                print('azimuth %.2f mrad' % (beam.global_azimuth*1e3))
            # print('ay: %.2f microrad' % (beam.ay*1e6))
            # print('cy: %.2f microns' % (beam.cy*1e6))

            if not success:
                break

        # return the output of the beamline
        return beam

    def propagate_until(self, beam_in, last_device, include_last=True):
        """
        Method to propagate from upstream end of beamline up until a given device. To be clear the beam is
        propagated through the last device.
        :param beam_in: Beam
            Beam input to upstream end of beamline.
        :param last_device: str
            Name of most downstream device to propagate through.
        :param include_last: bool
            Whether to propagate through the last device. Defaults to True
        :return beam: Beam
            Beam output from beamline section.
        """

        beam_in = copy.deepcopy(beam_in)
        # distance between source and first device
        dz = self.full_list[0].z - beam_in.z_source
        beam_in.reinitialize(dz)

        # save a few lines of code by using more general propagate_between method
        beam = self.propagate_between(beam_in, self.first_device().name, last_device,
                                      include_first=True, include_last=include_last)

        # return beam at the end of the beamline if we never found the last device
        return beam

    def propagate_between(self, beam_in, first_device, last_device, include_first=False, include_last=True):
        """
        Method to propagate between two devices along the beamline.
        :param beam_in: Beam
            Beam input to beamline section.
        :param first_device: str
            Name of upstream device.
        :param last_device: str
            Name of downstream device
        :param include_first: bool
            Whether to propagate through first device. Defaults to False.
        :param include_last: bool
            Whether to propagate through last device. Defaults to True.
        :return beam: Beam
            Beam output from beamline section.
        """

        # make a full copy of the beam input so that we don't modify the input
        beam = copy.deepcopy(beam_in)

        # get the first device
        try:
            device1 = getattr(self, first_device)
        except AttributeError:
            # if device doesn't exist, start at the beginning.
            device1 = self.full_list[0]

        if include_first:
            # include the first device
            index1 = self.full_list.index(device1)
        else:
            # skip the first device
            index1 = self.full_list.index(device1) + 1

        # get the last device
        try:
            device2 = getattr(self, last_device)
        except AttributeError:
            # if the device doesn't exist, go all the way to the end
            device2 = self.full_list[-1]

        if include_last:
            # include the last device
            index2 = self.full_list.index(device2)
        else:
            # skip the last device
            index2 = self.full_list.index(device2) - 1

        # return a partial list between the two devices, including the last device but not the first
        partial_list = self.full_list[index1:index2+1]



        # loop through devices
        for device in partial_list:
            # propagate through current device and print the name
            if not self.suppress:
                print('\033[1m' +device.name+'\033[0m')
            device.propagate(beam)
            if not self.suppress:
                print('zx: %.6f' % beam.zx)
                print('zy: %.6f' % beam.zy)
                print('azimuth %.2f mrad' % (beam.global_azimuth * 1e3))

        return beam

    # STILL NEEDS WORK!
    def alignment(self, beam_in, devices=None, screen1=None, screen2=None):
        """
        Method to align beamline section
        :param beam_in: Beam
            Beam upstream of any device provided.
        :param devices: list of optics objects
            Up to two devices used for aligning the beam in this section. Any devices beyond the second entry in the
            list are ignored.
        :param screen1: PPM
            First screen for alignment. Must be downstream of any devices in order to work properly.
        :param screen2: PPM
            Second screen for alignment. Must be downstream of any devices in order to work properly.
        :return: None
        """

        # if no devices are provided, the list is empty
        if devices is None:
            devices = []

        # make sure we have at least one motor
        try:
            device1 = getattr(self, devices[0])
        except AttributeError:
            # give up if there aren't any motors
            if not self.suppress:
                print('Need at least one motor')
            return
        # check if there's a second device. If not just use one.
        try:
            device2 = getattr(self, devices[1])
        except AttributeError:
            device2 = None
        # check if we have at least one screen
        try:
            screen1 = getattr(self, screen1)
        except AttributeError:
            # give up if there aren't any screens
            if not self.suppress:
                print('Need at least one screen')
            return
        # check if there is a second screen
        try:
            screen2 = getattr(self, screen2)
        except AttributeError:
            screen2 = None

        # only use a single device if only one was provided.
        if device2 is None:

            # check which degree of freedom we have
            orientation = device1.orientation

            direction = None

            if orientation == 0 or orientation == 2:
                direction = 'horizontal'
            elif orientation == 1 or orientation == 3:
                direction = 'vertical'

            # if we only have one screen, just use mirror angle to achieve desired centroid.
            if screen2 is None:

                # just use mirror angle to achieve desired centroid
                error = 0.0
                if direction == 'horizontal':
                    error = screen1.cx
                elif direction == 'vertical':
                    error = screen1.cy

                # z separation between screen and device
                dz = screen1.z - device1.z
                if dz < 0:
                    # give up if screen is upstream of device
                    if not self.suppress:
                        print('Screen is upstream of device')
                    return

                # make adjustment
                delta = -error / dz / 2
                device1.adjust_motor('delta', delta)
                if not self.suppress:
                    print('Changed %s angle by %.2f microrad' % (device1.name, delta*1e6))
                return

            else:
                # use both mirror dx and angle to achieve desired centroid
                error1 = 0.0
                error2 = 0.0
                if direction == 'horizontal':
                    error1 = screen1.cx
                    error2 = screen2.cx
                elif direction == 'vertical':
                    error1 = screen1.cy
                    error2 = screen2.cy

                # z separation
                dz1 = screen1.z - device1.z
                dz2 = screen2.z - device1.z

                # calculate motion needed to align both centroids
                delta = -(error1 - error2) / 2. / (dz1-dz2)
                dx = -(error2 * dz1 - error1 * dz2) / 2. / (dz1-dz2)
                device1.adjust_motor('delta', delta)
                device1.adjust_motor('dx', dx)

                # print message
                if not self.suppress:
                    print('Changed %s angle by %.2f microrad' % (device1.name, delta*1e6))
                    print('Moved %s x by %.2f microns' % (device1.name, dx*1e6))

        # if we have two devices just use the angle of both for alignment. May need to adjust this to also keep
        # second mirror centered with dx also.
        else:
            # make sure there are two screens
            if screen2 is None:
                if not self.suppress:
                    print('Need a second screen')
                return
            # we have two screens and two mirrors/devices!
            else:
                # calibrate motion influence on beam position at screens
                A = self.calibrate(beam_in, device1, device2, screen1, screen2)
                if not self.suppress:
                    print(A)

                # propagate beam up until first device
                beamOut = self.propagate_until(beam_in, device1.name)
                # propagate beam between first device and second screen
                self.propagate_between(beamOut, device1.name, screen2.name)

                # still needs work, this only works for horizontal motion...
                deltaC = np.array([screen1.cx, screen2.cx])
                deltaX = np.dot(A, deltaC)
                if not self.suppress:
                    print(deltaX)

                # move devices
                device1.adjust_motor('delta', -deltaX[0])
                device2.adjust_motor('delta', -deltaX[1])

    # NEEDS WORK!
    def calibrate(self, beam_in, device1, device2, screen1, screen2):
        """
        Method to calibrate alignment of beamline section. Only works for horizontal mirrors at the moment.
        :param beam_in: Beam
            Beam input to beamline section
        :param device1: optics object
            Upstream optic used for alignment
        :param device2: optics object
            Downstream optic used for alignment
        :param screen1: PPM
            Upstream screen used for alignment
        :param screen2: PPM
            Downstream screen used for alignment
        :return A: (2,2) ndarray
            calibration matrix
        """

        # initialize inverse calibration matrix
        A_inverse = np.zeros((2, 2))

        # propagate until just before first device
        beam_cal = self.propagate_until(beam_in, device1.name, include_last=False)

        # check centroids before moving anything. Assume beam_in has been propagated
        self.propagate_between(beam_cal, device1.name, screen2.name, include_first=True)

        # get centroids
        c1 = screen1.cx
        c2 = screen2.cx

        # make small adjustment to first mirror
        device1.adjust_motor('delta', 0.1e-6)

        # propagation between device1 and screen 2 (device1 inclusive)
        self.propagate_between(beam_cal, device1.name, screen2.name, include_first=True)

        # calculate centroid errors (scaled by motor motion)
        error1 = (screen1.cx - c1) / 0.1e-6
        error2 = (screen2.cx - c2) / 0.1e-6

        # start populating inverse array
        A_inverse[:, 0] = np.array([error1, error2])

        # move first mirror back
        device1.adjust_motor('delta', -0.1e-6)

        # make small adjustment to second mirror
        device2.adjust_motor('delta', 0.1e-6)

        # propagation between device1 and screen 2 (device1 inclusive)
        self.propagate_between(beam_cal, device1.name, screen2.name, include_first=True)

        # move second motor back
        device2.adjust_motor('delta', -0.1e-6)

        # calculate centroid errors (scaled by motor motion)
        error1 = (screen1.cx - c1) / 0.1e-6
        error2 = (screen2.cx - c2) / 0.1e-6

        # populate the rest of the inverse array
        A_inverse[:, 1] = np.array([error1, error2])

        # invert to get response matrix
        A = np.linalg.inv(A_inverse)

        return A

    def align_focus1(self, beam_in, parameters, goals, Ax, Ay):
        """
        Method to align a focus section.
        :param beam_in: Beam
            Beam input to this section.
        :param parameters: dict
            Various parameters for alignment.
        :param goals: dict
            alignment goals
        :param Ax: (2,2) ndarray
            horizontal motor response matrix
        :param Ay: (2,2) ndarray
            vertical motor response matrix
        :return: None
        """

        # unpack parameters and goals
        mirror_x_name = parameters['xControls'][0]
        mirror_y_name = parameters['yControls'][0]
        wfs_name = parameters['wfs_name']
        first_device = parameters['positions'][0]
        last_device = parameters['positions'][1]
        xGoals = goals['xGoals']
        yGoals = goals['yGoals']

        # get name of wavefront sensor screen
        wfs_screen_name = parameters['screens'][1]

        # get wavefront sensor screen
        wfs_screen = getattr(self, wfs_screen_name)

        # get list of screens
        screens = [getattr(self, screen_name)
                   for screen_name in parameters['screens']]
        # remove first screen (this is the focus screen)
        screens.pop(0)

        # get mirror objects
        mirror_x = getattr(self, mirror_x_name)
        mirror_y = getattr(self, mirror_y_name)
        # get wavefront object
        wfs = getattr(self, wfs_name)

        # disable the wavefront sensor
        wfs.disable()

        # propagate beamline section
        self.propagate_between(beam_in, first_device, last_device)

        # initialize horizontal wavefront measurements
        mx = {
                'z': 0,
                'L3': 0
                }
        # initialize vertical wavefront measurements
        my = {
                'z': 0,
                'L3': 0
                }

        # put in centroid measurements
        for num, screen in enumerate(screens):
            mx['c%d' % num] = screen.cx
            my['c%d' % num] = screen.cy

        # enable wfs
        wfs.enable()

        # get initial wavefront
        self.propagate_between(beam_in, first_device, last_device)
        # retrieve wavefront
        wfs_data = wfs_screen.retrieve_wavefront(wfs)
        # get wavefront data into measurement dicts
        mx['z'] = wfs_data['z2x']
        my['z'] = wfs_data['z2y']
        if not self.suppress:
            print(mx['z'])
            print(my['z'])
        mx['L3'] = wfs_data['coeff_x'][2]
        my['L3'] = wfs_data['coeff_y'][2]

        # initialize errors
        deltaC_x = []
        deltaC_y = []
        # populate error lists with centroid errors
        for num, screen in enumerate(screens):
            centroid = 'c%d' % num
            deltaC_x.append(mx[centroid] - xGoals[centroid])
            deltaC_y.append(my[centroid] - yGoals[centroid])

        # now add in wavefront errors
        deltaC_x.append(mx['z']-xGoals['z'])
        deltaC_x.append(mx['L3']-xGoals['L3'])
        deltaC_y.append(my['z']-yGoals['z'])
        deltaC_y.append(my['L3']-yGoals['L3'])

        if not self.suppress:
            print('deltaC_x: ' + str(deltaC_x))
            print('deltaC_y: ' + str(deltaC_y))

        # calculate motion necessary for alignment
        deltaX_x = np.dot(Ax, np.array(deltaC_x))
        deltaX_y = np.dot(Ay, np.array(deltaC_y))

        # get available motors
        motors_x = mirror_x.motor_list
        motors_y = mirror_y.motor_list

        if not self.suppress:
            print(motors_x)

        # move motors accordingly
        for num, motor in enumerate(motors_x):
            self.adjust_device(mirror_x_name, motor, -deltaX_x[num])
        for num, motor in enumerate(motors_y):
            self.adjust_device(mirror_y_name, motor, -deltaX_y[num])

        # should be aligned now
        return

    def calibrate_focus1(self, beam_in, parameters, calc_object):
        """
        Method to calibrate focus alignment. Not really implemented here yet.
        :param beam_in: Beam
            Beam input to focus section.
        :param parameters: dict
            alignment parameters
        :param calc_object: object from GUI.
        :return Ax: (2,2) ndarray
            horizontal response matrix
        :return Ay: (2,2) ndarray
            vertical response matrix
        """

        # get device names
        mirror_x_name = parameters['xControls'][0]
        mirror_y_name = parameters['yControls'][0]
        wfs_name = parameters['wfs_name']
        # get device names from focus section boundaries
        first_device = parameters['positions'][0]
        last_device = parameters['positions'][1]

        # get devices
        mirror_x = getattr(self, mirror_x_name)
        mirror_y = getattr(self, mirror_y_name)
        wfs = getattr(self, wfs_name)

        # get wavefront sensor screen name
        wfs_screen_name = parameters['screens'][1]

        # get wavefront sensor screen
        wfs_screen = getattr(self, wfs_screen_name)

        # get alignment screens
        screens = [getattr(self, screen_name)
                   for screen_name in parameters['screens']]
        # remove first screen. This is the IP camera
        screens.pop(0)

        # print screen names
        if not self.suppress:
            for screen in screens:
                print(screen.name)

        # number of measurements per axis. One per screen and 2 wavefront measurements
        N = len(screens) + 2
        num_screens = len(screens)

        # initial measurements
        mx0 = np.zeros(N)
        my0 = np.zeros(N)

        # disable wfs
        wfs.disable()

        # get initial positions
        # make copy of beam for propagation
        # propagate through beamline section
        self.propagate_between(beam_in, first_device, last_device)
        # get positions on screens

        # populate measurement arrays with centroids
        for num, screen in enumerate(screens):
            mx0[num] = screen.cx
            my0[num] = screen.cy

        # enable wfs
        wfs.enable()

        # get initial wavefront
        self.propagate_between(beam_in, first_device, last_device)
        wfs_data = wfs_screen.retrieve_wavefront(wfs)

        # initialize dict for sending plot data
        plot_data = {'beamline': self}
        for key in wfs_data.keys():
            plot_data[key] = wfs_data[key]

        # send data to be plotted
        calc_object.send_data(plot_data)

        # add in wavefront data to measurement arrays
        mx0[num_screens] = wfs_data['z2x']
        mx0[num_screens+1] = wfs_data['coeff_x'][2]
        my0[num_screens] = wfs_data['z2y']
        my0[num_screens+1] = wfs_data['coeff_y'][2]

        # re-initialize dict for sending plot data
        plot_data = {'beamline': self}
        for key in wfs_data.keys():
            plot_data[key] = wfs_data[key]

        # send data to be plotted
        calc_object.send_data(plot_data)

        # get motor names
        motors_x = mirror_x.motor_list
        motors_y = mirror_y.motor_list

        # print motors
        if not self.suppress:
            print(motors_x)
            print(motors_y)

        # number of measurements per axis
        measurementsPerAxis = 2+len(screens)

        # initialize response matrices
        Ax_inverse = np.zeros((measurementsPerAxis, len(motors_x)))
        Ay_inverse = np.zeros((measurementsPerAxis, len(motors_y)))

        # adjustment amount for calibration
        adjustment = 100.e-6

        # loop through motors
        for num, motor in enumerate(motors_x):

            # initialize measurement
            m1 = np.zeros(N)

            # adjust horizontal motor
            self.adjust_device(mirror_x_name, motor, adjustment)

            # propagate through beamline section first without WFS
            wfs.disable()
            self.propagate_between(beam_in, first_device, last_device)

            # centroid measurement
            for i, screen in enumerate(screens):
                m1[i] = screen.cx

            # now propagate with WFS
            wfs.enable()
            self.propagate_between(beam_in, first_device, last_device)
            # retrieve wavefront
            wfs_data = wfs_screen.retrieve_wavefront(wfs)

            plot_data = {'beamline': self}
            for key in wfs_data.keys():
                plot_data[key] = wfs_data[key]

            # plotting time
            calc_object.send_data(plot_data)

            # populate wavefront data
            m1[num_screens] = wfs_data['z2x']
            m1[num_screens+1] = wfs_data['coeff_x'][2]

            # normalized response to motor adjustment
            m_delta = (m1-mx0)/adjustment

            # populate inverse matrix
            Ax_inverse[:, num] = m_delta

            # move motor back
            self.adjust_device(mirror_x_name, motor, -adjustment)

        # print the measurement matrix
        if not self.suppress:
            print(Ax_inverse)

            # make sure the matrix is full rank
            print('rank: %.2f' % np.linalg.matrix_rank(Ax_inverse))

        # invert to get calibration matrix
        Ax = np.linalg.inv(Ax_inverse)

        # print it
        if not self.suppress:
            print(Ax)

        # now calibrate vertical motors
        for num, motor in enumerate(motors_y):

            # initialize measurement
            m1 = np.zeros(N)

            # move motor
            self.adjust_device(mirror_y_name, motor, adjustment)

            # propagate through beamline section first without WFS
            wfs.disable()
            self.propagate_between(beam_in, first_device, last_device)

            # populate measurement with centroid
            for i, screen in enumerate(screens):
                m1[i] = screen.cy

            # now propagate with WFS
            wfs.enable()
            self.propagate_between(beam_in, first_device, last_device)
            # retrieve wavefront
            wfs_data = wfs_screen.retrieve_wavefront(wfs)
            plot_data = {'beamline': self}
            for key in wfs_data.keys():
                plot_data[key] = wfs_data[key]

            # plotting time
            calc_object.send_data(plot_data)

            # add in wavefront data
            m1[num_screens] = wfs_data['z2y']
            m1[num_screens+1] = wfs_data['coeff_y'][2]

            # normalized response to motor adjustment
            m_delta = (m1-my0)/adjustment

            # populate inverse matrix column
            Ay_inverse[:, num] = m_delta

            # move motor back
            self.adjust_device(mirror_y_name, motor, -adjustment)

        # invert matrix to get calibration matrix
        Ay = np.linalg.inv(Ay_inverse)

        # print it
        if not self.suppress:
            print(Ay)

        return Ax, Ay
