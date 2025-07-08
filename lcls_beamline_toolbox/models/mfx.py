import numpy as np
import lcls_beamline_toolbox.xraywavetrace.beam as beam
import lcls_beamline_toolbox.xraywavetrace.optics as optics
import lcls_beamline_toolbox.xraywavetrace.beamline2d as beamline
import lcls_beamline_toolbox.xraywavetrace.motion as motion
import lcls_beamline_toolbox.utility.util as util

class MFX:
    def __init__(self, E0, N=512, ax=0, ay=0):

        self.E0 = E0
        # parameter dictionary. z_source is in LCLS coordinates (20 meters upstream of undulator exit)

        cx = ax*(100-26)
        cy = ay*(100-26)

        self.beam_params = {
            'photonEnergy': E0,
            'N': N,
            'sigma_x': 30e-6,
            'sigma_y': 30e-6,
            'rangeFactor': 5,
            'scaleFactor': 10,
            'z_source': 650 - 26,
            'ax': ax,
            'ay': ay,
            'cx': cx,
            'cy': cy
        }
        self.b1 = beam.Beam(beam_params=self.beam_params, suppress=True)
        self.beamline = self.define_beamline()
        self.prepare_tfs()

        # define motion
        self.mr1L4_pitch = motion.RotationAxis(self.beamline.MR1L4.sagittal,
                                               device_list=[self.beamline.MR1L4],
                                               rotation_center=self.beamline.MR1L4.get_pos(),
                                               initial_position=-552e-6)
        self.mr1l0_pitch = motion.RotationAxis(self.beamline.MR1L0.sagittal,
                                               device_list=[self.beamline.MR1L0],
                                               rotation_center=self.beamline.MR1L0.get_pos(),
                                               initial_position=0)
        self.mr2l0_pitch = motion.RotationAxis(self.beamline.MR2L0.sagittal,
                                               device_list=[self.beamline.MR2L0],
                                               rotation_center=self.beamline.MR2L0.get_pos(),
                                               initial_position=0)
        self.prefocus_y = motion.TranslationAxis(self.beamline.prefocus.yhat,
                                                 device_list=[self.beamline.prefocus])
        self.prefocus_x = motion.TranslationAxis(self.beamline.prefocus.xhat,
                                                 device_list=[self.beamline.prefocus])

        for crl in self.tfs_list:
            temp_x = motion.TranslationAxis(crl.xhat,device_list=[crl])
            temp_y = motion.TranslationAxis(crl.yhat,device_list=[crl])
            setattr(self,crl.name+'_x',temp_x)
            setattr(self,crl.name+'_y',temp_y)

        self.tfs_z = motion.TranslationAxis(self.beamline.tfs_2.zhat,
                                            device_list=self.tfs_list)

    def undulator_pointing(self,ax=0,ay=0):

        cx = ax * (100 - 26)
        cy = ay * (100 - 26)
        self.beam_params['ax'] = ax
        self.beam_params['ay'] = ay
        self.beam_params['cx'] = cx
        self.beam_params['cy'] = cy

        self.b1 = beam.Beam(beam_params=self.beam_params, suppress=True)

    def configure_tfs(self,tfs_config=0):
        # convert tfs_combination into binary
        tfs_binary = np.binary_repr(tfs_config, width=9)

        for i in range(9):
            if int(tfs_binary[i]) == 0:
                self.tfs_list[i].disable()
            else:
                self.tfs_list[i].enable()

    def prepare_tfs(self,E0=None,df=0):

        if E0 is None:
            E0 = self.E0

        f_temp = np.zeros(512)
        for i in range(512):
            self.configure_tfs(tfs_config=i)

            # calculate effective focal length
            s0 = self.beamline.prefocus.z - self.b1.z_source
            s_out = util.Util.lens_image_distance(s0,self.beamline.prefocus.calc_f(E0)) + self.beamline.prefocus.z
            for crl in self.tfs_list:
                if crl.enabled:
                    s_out = util.Util.lens_image_distance(crl.z-s_out,crl.calc_f(E0))+crl.z

            f_temp[i] = s_out - self.beamline.MFX_IP.z

        # find best configuration
        i0 = np.argmin(np.abs(f_temp-df))

        self.configure_tfs(i0)

        return f_temp[i0]


    def propagate(self):
        b2 = self.beamline.propagate_beamline(self.b1)

    def define_beamline(self, tfs_config=0):

        # FEE devices
        im2l0 = optics.PPM('IM2L0',z=735.99,FOV=5e-3,N=256)
        mr1l0 = optics.FlatMirror('MR1L0', z=740,alpha=2.1e-3,length=1)
        im3l0 = optics.PPM('IM3L0',z=746,FOV=5.632e-3,N=256)
        mr2l0 = optics.FlatMirror('MR2L0',z=747.286,alpha=2.1e-3,length=1,orientation=2)
        im4l0 = optics.PPM('IM4L0',z=753.559,FOV=5e-3,N=256)

        fee_devices = [im2l0,mr1l0,im3l0,mr2l0,im4l0]

        # TXI devices
        xpp_s1 = optics.Slit('xpp_s1',z=773.6,x_width=5e-3,y_width=5e-3)
        hx2_shared = optics.PPM('hx2_shared',z=774,FOV=5e-3,N=256)

        txi_devices = [xpp_s1,hx2_shared]

        # XRT
        xcs_s1 = optics.Slit('xcs_s1',z=809.5,x_width=5e-3,y_width=5e-3)
        xcs_yag1 = optics.PPM('xcs_yag1',z=810,FOV=5e-3,N=256)
        mr1l4 = optics.FlatMirror('MR1L4',z=817.1,alpha=2.75e-3,orientation=2)
        mfx_prefocus = optics.CRL('prefocus',z=983,roc=750e-6,diameter=5e-3)
        mfx_dia_yag = optics.PPM('DIA_YAG',z=984.9,FOV=5e-3,N=256)

        xrt_devices = [xcs_s1,xcs_yag1,mr1l4,mfx_prefocus,mfx_dia_yag]

        # MFX
        mfx_dg1_slits = optics.Slit('DG1_slits',z=1019.3,x_width=1e-3,y_width=1e-3)
        mfx_dg1_yag = optics.PPM('DG1_YAG',z=1019.79,FOV=5e-3,N=256)

        # transfocator (center at 1020.44
        z_tfs = 1020.44
        tfs_2 = optics.CRL('tfs_2',z=z_tfs-4*75e-3,roc=500e-6)
        tfs_3 = optics.CRL('tfs_3',z=z_tfs-3*75e-3,roc=300e-6)
        tfs_4 = optics.CRL('tfs_4',z=z_tfs-2*75e-3,roc=250e-6)
        tfs_5 = optics.CRL('tfs_5',z=z_tfs-75e-3,roc=200e-6)
        tfs_6 = optics.CRL('tfs_6',z=z_tfs,roc=125e-6)
        tfs_7 = optics.CRL('tfs_7',z=z_tfs+75e-3,roc=62.5e-6)
        tfs_8 = optics.CRL('tfs_8',z=z_tfs+2*75e-3,roc=50e-6)
        tfs_9 = optics.CRL('tfs_9',z=z_tfs+3*75e-3,roc=50e-6)
        tfs_10 = optics.CRL('tfs_10',z=z_tfs+4*75e-3,roc=50e-6)

        self.tfs_list = [tfs_2,tfs_3,tfs_4,tfs_5,tfs_6,tfs_7,tfs_8,tfs_9,tfs_10]

        # convert tfs_combination into binary
        tfs_binary = np.binary_repr(tfs_config,width=9)

        for i in range(9):
            if int(tfs_binary[i]) == 0:
                self.tfs_list[i].disable()


        mfx_dg2_us_slits = optics.Slit('DG2_us_slit', z=1021.29,x_width=1e-3,y_width=1e-3)
        mfx_dg2_yag = optics.PPM('DG2_YAG',z=1021.74,FOV=1e-3,N=256)
        mfx_dg2_ms_slits = optics.Slit('DG2_ms_slit',z=1022.84,x_width=1e-3,y_width=1e-3)
        mfx_dg2_ds_slits = optics.Slit('DG2_ds_slit',z=1022.99,x_width=1e-3,y_width=1e-3)

        mfx_ip = optics.PPM('MFX_IP',z=1024.84,FOV=100e-6,N=256)

        mfx_dg3_yag = optics.PPM('DG3_YAG',z=1027.84,FOV=1e-3,N=256)

        mfx_devices = [mfx_dg1_slits,mfx_dg1_yag,mfx_dg2_us_slits,mfx_dg2_yag,mfx_dg2_ms_slits,mfx_dg2_ds_slits,mfx_ip,
                       mfx_dg3_yag] + self.tfs_list

        device_list = fee_devices + txi_devices + xrt_devices + mfx_devices

        mfx_beamline = beamline.Beamline(device_list=device_list,ordered=False)

        return mfx_beamline




