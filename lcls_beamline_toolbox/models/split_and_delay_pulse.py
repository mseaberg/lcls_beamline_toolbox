import numpy as np
import matplotlib.pyplot as plt
import lcls_beamline_toolbox.xraywavetrace.beam1d as beam
import lcls_beamline_toolbox.xraywavetrace.optics1d as optics
import lcls_beamline_toolbox.xraywavetrace.beamline1d as beamline
import lcls_beamline_toolbox.xraywavetrace.motion as motion
import xrt.backends.raycing.materials as materials
import scipy.optimize as optimize
import copy
import scipy.spatial.transform as transform

class SND:
    def __init__(self, energy=10000, two_theta=None, delay=0, ax=0, ay=0, cx=0, cy=0):

        if two_theta is not None:
            self.E0 = self.calc_energy(two_theta)
        else:
            self.E0 = energy
        self.cc_branch = Util.define_cc(self.E0)
        self.delay_branch = Util.define_delay(self.E0,delay=delay)
        self.bypass_branch = Util.define_bypass(self.E0)
        self.delay = delay

        self.b2 = None
        self.b3 = None

        # parameter dictionary. z_source is in LCLS coordinates (20 meters upstream of undulator exit)
        self.beam_params = {
            'photonEnergy': self.E0,
            'N': 2048,
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
        self.b1 = beam.Beam(beam_params=self.beam_params,suppress=True)

        self.pulse_delay = beam.Pulse(beam_params=self.beam_params, tau=0.3, time_window=30)
        self.pulse_cc = beam.Pulse(beam_params=self.beam_params, tau=0.1, time_window=10)

        self.t1_tth = motion.RotationAxis(self.delay_branch.c1.sagittal,
                                          [self.delay_branch.c1, self.delay_branch.c2,
                                           self.delay_branch.t1_dh],
                                          rotation_center=self.delay_branch.c1.get_pos(),
                                          initial_position=2*self.delay_branch.c1.bragg,
                                          name='t1_tth')
        self.t1_th1 = motion.RotationAxis(-self.delay_branch.c1.sagittal,
                                          [self.delay_branch.c1],
                                          rotation_center=self.delay_branch.c1.get_pos()+self.delay_branch.c1.normal*1e-3,
                                          initial_position=self.delay_branch.c1.bragg,
                                          name='t1_th1')
        self.t1_th2 = motion.RotationAxis(self.delay_branch.c2.sagittal,
                                          [self.delay_branch.c2],
                                          initial_position=self.delay_branch.c2.bragg,
                                          name='t1_th2')
        self.t1_L = motion.TranslationAxis(self.delay_branch.t1_dh.zhat,
                                           [self.delay_branch.t1_dh,self.delay_branch.c2],
                                           initial_position=self.get_t1_L(), name='t1_L')
        self.t4_tth = motion.RotationAxis(-self.delay_branch.c4.sagittal,
                                          [self.delay_branch.c4,self.delay_branch.c3,
                                           self.delay_branch.t4_dh],
                                          rotation_center=self.delay_branch.c4.get_pos(),
                                          initial_position=2*self.delay_branch.c4.bragg,
                                          name='t4_tth')
        self.t4_th1 = motion.RotationAxis(self.delay_branch.c4.sagittal,
                                          [self.delay_branch.c4],
                                          initial_position=self.delay_branch.c4.bragg,
                                          name='t4_th1')
        self.t4_th2 = motion.RotationAxis(-self.delay_branch.c3.sagittal,
                                          [self.delay_branch.c3],
                                          initial_position=self.delay_branch.c3.bragg,
                                          name='t4_th2')
        self.t4_L = motion.TranslationAxis(-self.delay_branch.t4_dh.zhat,
                                           [self.delay_branch.t4_dh,self.delay_branch.c3],
                                           initial_position=self.get_t4_L(),name='t4_L')
        self.t1_chi1 = motion.RotationAxis(self.delay_branch.c1.tangential,
                                           [self.delay_branch.c1],
                                           name='t1_chi1')
        self.t1_chi2 = motion.RotationAxis(self.delay_branch.c2.tangential,
                                           [self.delay_branch.c2],
                                           name='t1_chi2')
        self.t4_chi1 = motion.RotationAxis(self.delay_branch.c4.tangential,
                                           [self.delay_branch.c4],
                                           name='t4_chi1')
        self.t4_chi2 = motion.RotationAxis(self.delay_branch.c3.tangential,
                                           [self.delay_branch.c3],
                                           name='t4_chi2')
        self.t1_x = motion.TranslationAxis(np.array([1,0,0]),
                                           [self.delay_branch.c1,self.delay_branch.t1_dh,
                                            self.delay_branch.c2],name='t1_x')
        self.t4_x = motion.TranslationAxis(np.array([1,0,0]),
                                           [self.delay_branch.c4,self.delay_branch.t4_dh,
                                            self.delay_branch.c3],name='t4_x')
        self.t1_y1 = motion.TranslationAxis(np.array([0,1,0]),
                                            [self.delay_branch.c1],name='t1_y1')
        self.t1_y2 = motion.TranslationAxis(np.array([0,1,0]),
                                            [self.delay_branch.c2],name='t1_y2')
        self.t4_y1 = motion.TranslationAxis(np.array([0,1,0]),
                                            [self.delay_branch.c4],name='t4_y1')
        self.t4_y2 = motion.TranslationAxis(np.array([0,1,0]),
                                            [self.delay_branch.c3],name='t4_y2')

        self.t1_x.coupled_axes = [self.t1_th1,self.t1_tth,self.t1_th2,self.t1_L,
                                  self.t1_y1,self.t1_y2,self.t1_chi1,self.t1_chi2]
        self.t1_tth.coupled_axes = [self.t1_th1,self.t1_th2,self.t1_y1,self.t1_y2,
                                    self.t1_chi1,self.t1_chi2,self.t1_L]
        self.t4_x.coupled_axes = [self.t4_th1,self.t4_tth,self.t4_th2,self.t4_L,
                                  self.t4_y1,self.t4_y2,self.t4_chi1,self.t4_chi2]
        self.t4_tth.coupled_axes = [self.t4_th1,self.t4_th2,self.t1_y1,self.t1_y2,
                                    self.t1_chi1,self.t1_chi2,self.t4_L]
        self.t1_L.coupled_axes = [self.t1_th2,self.t1_chi2,self.t1_y2]
        self.t4_L.coupled_axes = [self.t4_th2,self.t4_chi2,self.t4_y2]

        self.t2_th = motion.RotationAxis(self.cc_branch.cc1_1.sagittal,
                                         [self.cc_branch.cc1_1,self.cc_branch.cc1_2],
                                         rotation_center=self.cc_branch.cc1_1.get_pos(),
                                         name='t2_th')
        self.t2_x = motion.TranslationAxis(np.array([1,0,0]),
                                           [self.cc_branch.cc1_1,self.cc_branch.cc1_2],
                                           name='t2_x')
        self.t3_th = motion.RotationAxis(-self.cc_branch.cc2_2.sagittal,
                                         [self.cc_branch.cc2_1,self.cc_branch.cc2_2],
                                         rotation_center=self.cc_branch.cc2_2.get_pos(),
                                         name='t3_th')
        self.t3_x = motion.TranslationAxis(np.array([1,0,0]),
                                           [self.cc_branch.cc2_1,self.cc_branch.cc2_2],
                                           name='t3_x')
        self.t2_x.coupled_axes = [self.t2_th]
        self.t3_x.coupled_axes = [self.t3_th]

        self.motor_list = [self.t1_tth,self.t1_th1,self.t1_th2,self.t4_th2,self.t4_th1,
                           self.t4_tth,self.t1_L,self.t4_L,self.t1_chi1,self.t1_chi2,
                           self.t4_chi1,self.t4_chi2,self.t1_x,self.t2_x,self.t3_x,
                           self.t4_x,self.t2_th,self.t3_th,self.t1_y1,self.t1_y2,self.t4_y1,
                           self.t4_y2]

        self.motor_dict = {}
        for motor in self.motor_list:
            self.motor_dict[motor.name] = motor

        self.delay_diagnostic_list = ['di','t1_dh','dd','t4_dh','do','IP']
        self.cc_diagnostic_list = ['di','dcc','do','IP']
        self.delay_beam_stats = {}
        for diagnostic in self.delay_diagnostic_list:
            self.delay_beam_stats[diagnostic] = {
                'intensity': 0,
                'cx': 0,
                'cy': 0
            }


    def calc_energy(self, two_theta):
        test_crystal = materials.CrystalSi(hkl=[2,2,0])
        theta = two_theta/2
        l0 = 2 * test_crystal.d * 1e-10 * np.sin(theta)
        E0 = 1239.84 / (l0 * 1e9)
        dtheta = test_crystal.get_dtheta(E0, alpha=0)
        l0 = 2 * test_crystal.d * 1e-10 * np.sin(theta+dtheta)
        E0 = 1239.84 / (l0 * 1e9)

        return E0

    def propagate_delay(self):
        #self.b2 = self.delay_branch.propagate_beamline(self.b1)
        self.pulse_delay.propagate(beamline=self.delay_branch,
                                   screen_names=self.delay_diagnostic_list)
        for diagnostic in self.delay_diagnostic_list:
            self.delay_beam_stats[diagnostic] = self.pulse_delay.get_beam_stats(diagnostic)

    def propagate_cc(self):
        #self.b3 = self.cc_branch.propagate_beamline(self.b1)
        self.pulse_cc.propagate(beamline=self.cc_branch,
                                screen_names=self.cc_diagnostic_list)

    def propagate_bypass(self):
        #self.bypass_branch.propagate_beamline(self.b1)
        pass

    def adjust_cc(self,delta):
        "Function to adjust both channel-cut crystals together"

        cc11 = self.cc_branch.cc1_1
        cc12 = self.cc_branch.cc1_2
        cc21 = self.cc_branch.cc2_1
        cc22 = self.cc_branch.cc2_2

        pivot1 = Util.get_pos(cc11)
        rot_vec1 = cc11.sagittal * delta
        Util.rotate_about_point(cc11, pivot1, rot_vec1)
        Util.rotate_about_point(cc12, pivot1, rot_vec1)

        pivot2 = Util.get_pos(cc22)
        rot_vec2 = -cc22.sagittal * delta
        Util.rotate_about_point(cc21, pivot2, rot_vec2)
        Util.rotate_about_point(cc22, pivot2, rot_vec2)

    # calculate t1_th1 from c1 orientation
    def get_t1_th1(self):
        vec1 = self.delay_branch.c1.tangential
        angle1 = np.arccos(vec1[2])
        angle2 = self.get_t1_tth()

        return angle2 - angle1

    # move t1_th1 by delta
    def mvr_t1_th1(self, delta):
        pivot = Util.get_pos(self.delay_branch.c1)
        rot_vec = -self.delay_branch.c1.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c1, pivot, rot_vec)

    def mv_t1_th1(self, pos):
        self.mvr_t1_th1(pos-self.get_t1_th1())

    # calculate t1_chi1 from c1 orientation
    def get_t1_chi1(self):
        vec1 = self.delay_branch.c1.normal
        angle1 = np.arcsin(vec1[1])

        return angle1

    def mvr_t1_chi1(self, delta):
        pivot = Util.get_pos(self.delay_branch.c1)
        rot_vec = self.delay_branch.c1.tangential * delta
        Util.rotate_about_point(self.delay_branch.c1, pivot, rot_vec)

    def mv_t1_chi1(self, pos):
        self.mvr_t1_chi1(pos-self.get_t1_chi1())

    def get_t1_tth(self):
        c1_pos = Util.get_pos(self.delay_branch.c1)
        c2_pos = Util.get_pos(self.delay_branch.c2)
        vec = c2_pos - c1_pos
        return np.arccos(vec[2] / np.sqrt(np.sum(vec ** 2)))

    def mvr_t1_tth(self, delta):
        pivot = Util.get_pos(self.delay_branch.c1)
        rot_vec = self.delay_branch.c1.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c2, pivot, rot_vec)
        Util.rotate_about_point(self.delay_branch.c1, pivot, rot_vec)
        Util.rotate_about_point(self.delay_branch.t1_dh, pivot, rot_vec)

    def mv_t1_tth(self, pos):
        self.mvr_t1_tth(pos-self.get_t1_tth())

    def get_t1_L(self):
        c1_pos = Util.get_pos(self.delay_branch.c1)
        c2_pos = Util.get_pos(self.delay_branch.c2)
        vec = c2_pos - c1_pos
        return np.sqrt(np.sum(vec ** 2))

    def mvr_t1_L(self, delta):
        c1_pos = Util.get_pos(self.delay_branch.c1)
        c2_pos = Util.get_pos(self.delay_branch.c2)
        vec = c2_pos - c1_pos
        norm_vec = vec / np.sqrt(np.sum(vec ** 2))
        c2_pos += norm_vec * delta
        self.delay_branch.c2.global_x = c2_pos[0]
        self.delay_branch.c2.global_y = c2_pos[1]
        self.delay_branch.c2.z = c2_pos[2]

        dh_pos = Util.get_pos(self.delay_branch.t1_dh)
        dh_pos += norm_vec * delta
        self.delay_branch.t1_dh.global_x = dh_pos[0]
        self.delay_branch.t1_dh.global_y = dh_pos[1]
        self.delay_branch.t1_dh.z = dh_pos[2]

    def mv_t1_L(self, pos):
        self.mvr_t1_L(pos-self.get_t1_L())

    def get_t1_th2(self):
        vec1 = self.delay_branch.c2.tangential
        angle1 = np.arccos(vec1[2])
        angle2 = self.get_t1_tth()
        return angle2 - angle1

    def mvr_t1_th2(self, delta):
        pivot = Util.get_pos(self.delay_branch.c2)
        rot_vec = self.delay_branch.c2.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c2, pivot, rot_vec)

    def mv_t1_th2(self, pos):
        self.mvr_t1_th2(pos-self.get_t1_th2())

    def get_t1_chi2(self):
        vec1 = self.delay_branch.c2.normal
        angle1 = np.arcsin(vec1[1])

        return angle1

    def mvr_t1_chi2(self, delta):
        pivot = Util.get_pos(self.delay_branch.c2)
        rot_vec = -self.delay_branch.c2.tangential * delta
        Util.rotate_about_point(self.delay_branch.c2, pivot, rot_vec)

    def mv_t1_chi2(self, pos):
        self.mvr_t1_chi2(pos-self.get_t1_chi2())

    def get_t4_th1(self):
        vec1 = self.delay_branch.c4.tangential
        angle1 = np.arccos(vec1[2])
        angle2 = self.get_t4_tth()

        return angle2 - angle1

    def mvr_t4_th1(self, delta):
        pivot = Util.get_pos(self.delay_branch.c4)
        rot_vec = self.delay_branch.c4.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c4, pivot, rot_vec)

    def mv_t4_th1(self, pos):
        self.mvr_t4_th1(pos-self.get_t4_th1())

    def get_t4_chi1(self):
        vec1 = self.delay_branch.c4.normal
        angle1 = np.arcsin(vec1[1])

        return angle1

    def mvr_t4_chi1(self, delta):
        pivot = Util.get_pos(self.delay_branch.c4)
        rot_vec = self.delay_branch.c4.tangential * delta
        Util.rotate_about_point(self.delay_branch.c4, pivot, rot_vec)

    def mv_t4_chi1(self, pos):
        self.mvr_t4_chi1(pos-self.get_t4_chi1())

    def get_t4_tth(self):
        c3_pos = Util.get_pos(self.delay_branch.c3)
        c4_pos = Util.get_pos(self.delay_branch.c4)
        vec = c4_pos - c3_pos
        return np.arccos(vec[2] / np.sqrt(np.sum(vec ** 2)))

    def mvr_t4_tth(self, delta):
        pivot = Util.get_pos(self.delay_branch.c4)
        rot_vec = -self.delay_branch.c4.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c3, pivot, rot_vec)
        Util.rotate_about_point(self.delay_branch.c4, pivot, rot_vec)
        Util.rotate_about_point(self.delay_branch.t4_dh, pivot, rot_vec)

    def mv_t4_tth(self, pos):
        self.mvr_t4_tth(pos-self.get_t4_tth())

    def get_t4_L(self):
        c3_pos = Util.get_pos(self.delay_branch.c3)
        c4_pos = Util.get_pos(self.delay_branch.c4)
        vec = c3_pos - c4_pos
        return np.sqrt(np.sum(vec ** 2))

    def mvr_t4_L(self, delta):
        c3_pos = Util.get_pos(self.delay_branch.c3)
        c4_pos = Util.get_pos(self.delay_branch.c4)
        vec = c3_pos - c4_pos
        norm_vec = vec / np.sqrt(np.sum(vec ** 2))
        c3_pos += norm_vec * delta
        self.delay_branch.c3.global_x = c3_pos[0]
        self.delay_branch.c3.global_y = c3_pos[1]
        self.delay_branch.c3.z = c3_pos[2]

        dh_pos = Util.get_pos(self.delay_branch.t4_dh)
        dh_pos += norm_vec * delta
        self.delay_branch.t4_dh.global_x = dh_pos[0]
        self.delay_branch.t4_dh.global_y = dh_pos[1]
        self.delay_branch.t4_dh.z = dh_pos[2]

    def mv_t4_L(self, pos):
        self.mvr_t4_L(pos-self.get_t4_L())

    def get_t4_th2(self):
        vec1 = self.delay_branch.c3.tangential
        angle1 = np.arccos(vec1[2])
        angle2 = self.get_t4_tth()
        return angle2 - angle1

    def mvr_t4_th2(self, delta):
        pivot = Util.get_pos(self.delay_branch.c3)
        rot_vec = -self.delay_branch.c3.sagittal * delta
        Util.rotate_about_point(self.delay_branch.c3, pivot, rot_vec)

    def mv_t4_th2(self, pos):
        self.mvr_t4_th2(pos-self.get_t4_th2())

    def get_t4_chi2(self):
        vec1 = self.delay_branch.c3.normal
        angle1 = np.arcsin(vec1[1])

        return angle1

    def mvr_t4_chi2(self, delta):
        pivot = Util.get_pos(self.delay_branch.c3)
        rot_vec = -self.delay_branch.c3.tangential * delta
        Util.rotate_about_point(self.delay_branch.c3, pivot, rot_vec)

    def mv_t4_chi2(self, pos):
        self.mvr_t4_chi2(pos - self.get_t4_chi2())

    def get_t2_th(self):
        vec = self.cc_branch.cc1_1.tangential
        return np.arccos(vec[2])

    def mvr_t2_th(self, delta):
        cc11 = self.cc_branch.cc1_1
        cc12 = self.cc_branch.cc1_2

        pivot1 = Util.get_pos(cc11)
        rot_vec1 = cc11.sagittal * delta
        Util.rotate_about_point(cc11, pivot1, rot_vec1)
        Util.rotate_about_point(cc12, pivot1, rot_vec1)

    def mv_t2_th(self, pos):
        self.mvr_t2_th(pos - self.get_t2_th())

    def get_t3_th(self):
        vec = self.cc_branch.cc2_2.tangential
        return np.arccos(vec[2])

    def mvr_t3_th(self, delta):
        cc21 = self.cc_branch.cc2_1
        cc22 = self.cc_branch.cc2_2

        pivot2 = Util.get_pos(cc22)
        rot_vec2 = -cc22.sagittal * delta
        Util.rotate_about_point(cc21, pivot2, rot_vec2)
        Util.rotate_about_point(cc22, pivot2, rot_vec2)

    def mv_t3_th(self, pos):
        self.mvr_t3_th(pos - self.get_t3_th())

    def get_t1_x(self):
        return np.copy(self.delay_branch.c1.global_x)

    def mvr_t1_x(self, delta):
        self.delay_branch.c1.global_x += delta
        self.delay_branch.c2.global_x += delta
        self.delay_branch.t1_dh.global_x += delta

    def mv_t1_x(self, pos):
        self.mvr_t1_x(pos-self.get_t1_x())

    def get_t2_x(self):
        return np.copy(self.cc_branch.cc1_1.global_x)

    def mvr_t2_x(self, delta):
        self.cc_branch.cc1_1.global_x += delta
        self.cc_branch.cc1_2.global_x += delta

    def mv_t2_x(self, pos):
        self.mvr_t2_x(pos - self.get_t2_x())

    def get_t3_x(self):
        return np.copy(self.cc_branch.cc2_2.global_x)

    def mvr_t3_x(self, delta):
        self.cc_branch.cc2_1.global_x += delta
        self.cc_branch.cc2_2.global_x += delta

    def mv_t3_x(self, pos):
        self.mvr_t3_x(pos-self.get_t3_x())

    def get_t4_x(self):
        return np.copy(self.delay_branch.c4.global_x)

    def mvr_t4_x(self, delta):
        self.delay_branch.c3.global_x += delta
        self.delay_branch.c4.global_x += delta
        self.delay_branch.t4_dh.global_x += delta

    def mv_t4_x(self, pos):
        self.mvr_t4_x(pos - self.get_t4_x())

    def get_dcc_x(self):
        return np.copy(self.cc_branch.dcc.global_x)

    def mvr_dcc_x(self, delta):
        self.cc_branch.dcc.global_x += delta

    def mv_dcc_x(self, pos):
        self.mvr_dcc_x(pos - self.get_dcc_x())

    def get_dd_x(self):
        return np.copy(self.delay_branch.dd.global_x)

    def mvr_dd_x(self, delta):
        self.delay_branch.dd.global_x += delta

    def mv_dd_x(self, pos):
        self.mvr_dd_x(pos - self.get_dd_x())

    def get_t1_dh_sum(self):
        #return self.delay_branch.t1_dh.profile.sum()
        return self.delay_beam_stats['t1_dh']['intensity']

    def get_dd_sum(self):
        #return self.delay_branch.dd.profile.sum()
        return self.delay_beam_stats['dd']['intensity']

    def get_t4_dh_sum(self):
        #return self.delay_branch.t4_dh.profile.sum()
        return self.delay_beam_stats['t4_dh']['intensity']

    def get_do_sum(self):
        #return self.delay_branch.do.profile.sum()
        return self.delay_beam_stats['do']['intensity']

    def get_IP_sum(self):
        #return self.delay_branch.IP.profile.sum()
        return self.delay_beam_stats['IP']['intensity']

    def get_dd_cx(self):
        #return self.delay_branch.dd.cx
        return self.delay_beam_stats['dd']['cx']

    def get_dd_cy(self):
        #return self.delay_branch.dd.cy
        return self.delay_beam_stats['dd']['cy']

    def get_do_cx(self):
        #return self.delay_branch.do.cx
        return self.delay_beam_stats['do']['cx']

    def get_do_cy(self):
        #return self.delay_branch.do.cy
        return self.delay_beam_stats['do']['cy']

    def get_do_r(self):
        r = np.sqrt(self.get_do_cx()**2+self.get_do_cy()**2)
        return r

    def get_IP_cx(self):
        #return self.delay_branch.IP.cx
        return self.delay_beam_stats['IP']['cx']

    def get_IP_cy(self):
        #return self.delay_branch.IP.cy
        return self.delay_beam_stats['IP']['cy']

    def get_IP_r(self):
        r = np.sqrt(self.get_IP_cx()**2+self.get_IP_cy()**2)
        return r



class Util:

    # "channel-cut" branch.
    # This branch is simpler in that there are nominally only two degrees of freedom - rotation stages
    # for each channel cut crystal which are used to select the photon energy.
    @staticmethod
    def define_cc(E0):
        """
        Parameters
        ----------
        E0: photon energy (eV)

        Returns
        -------
        Beamline object
        """
        # channel cut branch

        lens00 = optics.CRL('lens00', diameter=5e-3, E0=E0, f=356, z=980, orientation=0)
        lens01 = optics.CRL('lens01', diameter=5e-3, E0=E0, f=356, z=980 + 1e-6, orientation=1)

        s3 = optics.Slit('s3', z=990, x_width=0.5e-3, y_width=1e-3)
        di = optics.PPM('di', z=1000, FOV=4e-3, N=256)
        t1_slit = optics.Slit('t1', y_width=4e-3, dy=2e-3, z=di.z + 0.15)

        cc1_1 = optics.Crystal('cc1_1', hkl=[2, 2, 0], E0=E0, z=t1_slit.z + 2. / 3., orientation=2, length=.02, width=.02,
                               show_figures=False)
        dci = optics.PPM('dci', z=cc1_1.z - 75e-3, N=256)
        gap = 55e-3
        dz = gap * np.sin(np.pi / 2 - 2 * cc1_1.alpha) / np.sin(cc1_1.alpha)
        # print('dz: {}'.format(dz))
        cc1_2 = optics.Crystal('cc1_2', hkl=[2, 2, 0], E0=E0, z=cc1_1.z + dz, orientation=0, length=.06, width=.02,
                               show_figures=False)
        dcc = optics.PPM('dcc', z=t1_slit.z + 1, FOV=4e-3, N=256)

        cc2_2 = optics.Crystal('cc2_2', hkl=[2, 2, 0], E0=E0, z=t1_slit.z + 2 - 2. / 3., orientation=2, length=.02,
                               width=.02)
        dz = gap * np.sin(np.pi / 2 - 2 * cc2_2.alpha) / np.sin(cc2_2.alpha)
        cc2_1 = optics.Crystal('cc2_1', hkl=[2, 2, 0], E0=E0, z=cc2_2.z - dz, orientation=0, length=.06, width=.02,
                               show_figures=False)
        dco = optics.PPM('dco', z=cc2_2.z + 75e-3, FOV=4e-3, N=256)
        t4_slit = optics.Slit('t4', y_width=4e-3, dy=2e-3, z=t1_slit.z + 2)
        do = optics.PPM('do', z=t4_slit.z + 0.15, FOV=4e-3, N=256)
        lens0 = optics.CRL('lens0', E0=E0, f=3, z=do.z + 5, orientation=0, diameter=2e-3)
        lens1 = optics.CRL('lens1', E0=E0, f=3, z=do.z + 5 + 1e-6, orientation=1, diameter=2e-3)

        image = 1 / (1 / 3. - 1 / (lens0.z - 624))
        # print(image)

        IP = optics.PPM('IP', FOV=10e-6, z=lens1.z + image, N=256)

        device_list = [s3, di, t1_slit, dci, cc1_1, cc1_2, dcc, cc2_1, cc2_2, dco, t4_slit, do, lens0, lens1, IP]

        cc_branch = beamline.Beamline(device_list, ordered=True)

        return cc_branch


    # "channel-cut" branch in "bypass" mode, meaning the crystals are moved out of the way.
    # Cheating a bit here in that I simply didn't include the crystals in the Beamline object.
    @staticmethod
    def define_bypass(E0):
        # channel cut branch

        lens00 = optics.CRL('lens00', diameter=5e-3, E0=E0, f=356, z=980, orientation=0)
        lens01 = optics.CRL('lens01', diameter=5e-3, E0=E0, f=356, z=980 + 1e-6, orientation=1)

        s3 = optics.Slit('s3', z=990, x_width=0.5e-3, y_width=1e-3)
        di = optics.PPM('di', z=1000, FOV=4e-3, N=256)
        t1_slit = optics.Slit('t1', y_width=4e-3, dy=2e-3, z=di.z + 0.15)

        cc1_1 = optics.Crystal('cc1_1', hkl=[2, 2, 0], E0=E0, z=t1_slit.z + 2. / 3., orientation=2, length=.02, width=.02,
                               show_figures=False)
        dci = optics.PPM('dci', z=cc1_1.z - 75e-3, N=256)
        gap = 55e-3
        dz = gap * np.sin(np.pi / 2 - 2 * cc1_1.alpha) / np.sin(cc1_1.alpha)
        # print('dz: {}'.format(dz))
        cc1_2 = optics.Crystal('cc1_2', hkl=[2, 2, 0], E0=E0, z=cc1_1.z + dz, orientation=0, length=.06, width=.02,
                               show_figures=False)
        dcc = optics.PPM('dcc', z=t1_slit.z + 1, FOV=4e-3, N=256)

        cc2_2 = optics.Crystal('cc2_2', hkl=[2, 2, 0], E0=E0, z=t1_slit.z + 2 - 2. / 3., orientation=2, length=.02,
                               width=.02)
        dz = gap * np.sin(np.pi / 2 - 2 * cc2_2.alpha) / np.sin(cc2_2.alpha)
        cc2_1 = optics.Crystal('cc2_1', hkl=[2, 2, 0], E0=E0, z=cc2_2.z - dz, orientation=0, length=.06, width=.02,
                               show_figures=False)
        dco = optics.PPM('dco', z=cc2_2.z + 75e-3, FOV=4e-3, N=256)
        t4_slit = optics.Slit('t4', y_width=4e-3, dy=2e-3, z=t1_slit.z + 2)
        do = optics.PPM('do', z=t4_slit.z + 0.15, FOV=4e-3, N=256)
        lens0 = optics.CRL('lens0', E0=E0, f=3, z=do.z + 5, orientation=0, diameter=2e-3)
        lens1 = optics.CRL('lens1', E0=E0, f=3, z=do.z + 5 + 1e-6, orientation=1, diameter=2e-3)

        image = 1 / (1 / 3.0 - 1 / (lens0.z - 624))
        # print(image)

        IP = optics.PPM('IP', FOV=10e-6, z=lens1.z + image)

        device_list = [s3, di, t1_slit, dci, dcc, dco, t4_slit, do, lens0, lens1, IP]

        cc_branch = beamline.Beamline(device_list, ordered=True)

        return cc_branch


    # "delay" branch
    # This is the more complicated branch since it has more degrees of freedom, some that are confined to single
    # crystal motion and some that are coupled.
    # The idea is that the "towers" (t1 and t4) adjust angle based on the photon energy, and adjust the crystal
    # spacing (L) to introduce a delay relative to the channel-cut branch.
    # In terms of defining the optics it's not necessarily more complicated than the channel-cut branch, but the
    # complexity comes when defining the degrees of freedom (in the next cell).
    @staticmethod
    def define_delay(E0, delay=0, bypass=False):
        """
        Parameters
        ----------
        E0: photon energy (eV)
        delay: delay branch delay (ps)

        Returns
        -------
        Beamline object
        """
        s3 = optics.Slit('s3', z=990, x_width=0.5e-3, y_width=1e-3)
        di = optics.PPM('di', z=1000, FOV=4e-3, N=256)
        c1slit = optics.Slit('c1slit', y_width=4e-3, dy=-2e-3, z=di.z + 0.15 - 1e-6)
        c1 = optics.Crystal('c1', hkl=[2, 2, 0], E0=E0, z=di.z + 0.15, orientation=0, width=5e-3, dy=-2.5e-3, length=.02)

        # dummy = optics.Crystal('dummy', hkl=[2, 2, 0], E0=6500)
        dummy = optics.Crystal('dummy', hkl=[2, 2, 0], E0=E0)
        gap = 55e-3
        dz = gap * np.sin(np.pi / 2 - 2 * dummy.alpha) / np.sin(dummy.alpha)
        L0 = dz / np.cos(2 * dummy.alpha)
        # print("L0: {}".format(L0))
        #     dz += delay*np.cos(2*c1.alpha)
        delay_L = delay*1e-12*299792458/2/(1-np.cos(2*c1.alpha))
        dz = (L0 + delay_L) * np.cos(2 * c1.alpha)
        dz2 = (L0 + delay_L - 30e-3) * np.cos(2 * c1.alpha)
        # print('dz: {}'.format(dz))
        c2 = optics.Crystal('c2', hkl=[2, 2, 0], E0=E0, z=c1.z + dz, orientation=2, length=.02, width=.02,
                            show_figures=False)

        t1_dh = optics.PPM('t1_dh', z=c1.z + dz2, FOV=4e-3, N=256)

        #     c2 = optics.Crystal('c2',hkl=[2,2,0],E0=E0,z=c1.z+dz,orientation=2,length=.02,width=.02)
        dd = optics.PPM('dd', z=c1.z + 1, FOV=4e-3, N=256)
        c4 = optics.Crystal('c4', hkl=[2, 2, 0], E0=E0, z=c1.z + 2, orientation=0, length=.02, width=.02)
        c3 = optics.Crystal('c3', hkl=[2, 2, 0], E0=E0, z=c4.z - dz, orientation=2, length=.02, width=.02)
        t4_dh = optics.PPM('t4_dh', z=c4.z - dz2, FOV=4e-3, N=256)

        c4slit = optics.Slit('c4slit', y_width=4e-3, dy=-2e-3, z=c4.z + 1e-6)

        do = optics.PPM('do', z=c4.z + 0.15, FOV=4e-3, N=256)
        lens0 = optics.CRL('lens0', E0=E0, f=3, z=do.z + 5, orientation=0, diameter=2e-3)
        lens1 = optics.CRL('lens1', E0=E0, f=3, z=do.z + 5 + 1e-6, orientation=1, diameter=2e-3)

        d_lens = optics.PPM('d_lens',z=lens1.z+.1, FOV=4e-3, N=256)
        image = 1 / (1 / 3.0 - 1 / (lens0.z - 624))
        # print(image)

        IP = optics.PPM('IP', FOV=200e-6, z=lens1.z + image,suppress=True, N=1024)

        device_list = [s3, di, c1slit, c1, t1_dh, c2, dd, c3, t4_dh, c4, c4slit, do, lens0, lens1, d_lens, IP]

        delay_branch = beamline.Beamline(device_list, ordered=True)

        return delay_branch


    # functions for adjusting the various degrees of freedom

    # get the cartesian coordinates of a given device
    @staticmethod
    def get_pos(device):
        pos_vec = np.zeros((3))
        pos_vec[0] = device.global_x
        pos_vec[1] = device.global_y
        pos_vec[2] = device.z

        return pos_vec

    # function to rotate an optic about a point, using the provided rotation vector
    @staticmethod
    def rotate_about_point(device, point, rot_vec):
        re = transform.Rotation.from_rotvec(rot_vec)
        Re = re.as_matrix()

        device_pos = Util.get_pos(device)
        new_pos = np.matmul(Re, device_pos - point) + point

        if issubclass(type(device), optics.Mirror):
            device.normal = np.matmul(Re, device.normal)
            device.sagittal = np.matmul(Re, device.sagittal)
            device.tangential = np.matmul(Re, device.tangential)
        else:
            device.xhat = np.matmul(Re, device.xhat)
            device.yhat = np.matmul(Re, device.yhat)
            device.zhat = np.matmul(Re, device.zhat)

        device.global_x = new_pos[0]
        device.global_y = new_pos[1]
        device.z = new_pos[2]

        return new_pos