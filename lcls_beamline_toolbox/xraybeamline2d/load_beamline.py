import numpy as np
from . import beamline1d as beamline
from . import optics1d_normal as optics1d


def load(beamline_name, IP_name):

    device_array = []
    positions = []
    screens = []
    xControls = []
    yControls = []
    WFS_sections = []
    xGoals = []
    yGoals = []
    zGoals = {}
    wfs_names = {}
   
    # screens shared by all SXR beamlines

    if beamline_name == 'TMO':
        im1k0 = optics1d.PPM('IM1K0', FOV=9.4e-3, z=699.47)
        im2k0 = optics1d.PPM('IM2K0', FOV=9.4e-3, z=731.61)
        mr1k4 = optics1d.FlatMirror('MR1K4', length=1, alpha=12e-3, z=745.72,
                                    orientation=0)
        # pc2k4 = optics1d.Collimator('PC2K4', diameter=14e-3, z=752.88)
        im1k4 = optics1d.PPM('IM1K4', FOV=9.4e-3, z=750.347)
        # sl2k4 = optics1d.Slit('SL2K4',x_width=10e-3, y_width=10e-3, z=755.1)
        im2k4 = optics1d.PPM('IM2K4', FOV=9.4e-3, z=755.32)
        # pc3k4 = optics1d.Collimator('PC3K4', diameter=14e-3, z=755.93)
        im3k4 = optics1d.PPM('IM3K4', FOV=9.4e-3, z=758.889)
        mr2k4 = optics1d.CurvedMirror('MR2K4', length=0.6, p=115., q=2.25,
                                      alpha=14.e-3, z=759.493, orientation=0)
        mr2k4.enable_motors('dF1', 'dF2')
        mr3k4 = optics1d.CurvedMirror('MR3K4', length=0.6, p=115., q=1.6, alpha=14.e-3,
                                      z=760.143, orientation=1)
        mr3k4.enable_motors('dF1', 'dF2')
        im4k4 = optics1d.PPM('IM4K4', FOV=5.7e-3, z=761.101)
        IP1 = optics1d.PPM('IP1', FOV=100.e-6, z=761.743)
        pinhole = optics1d.Slit('pinhole', x_width=10e-3, y_width=10e-3, z=761.7429999)
        pf1k4 = optics1d.WFS('PF1K4', pitch=37.2e-6, z=763.515, f0=1.772,
                             fraction=2, enabled=False)
        im5k4 = optics1d.PPM('IM5K4', FOV=9.4e-3, z=764.313, blur=True, distort=True)

        mr4k4 = optics1d.CurvedMirror('MR4K4', length=0.6, p=4.45, q=1.4, alpha=21e-3,
                                      z=766.193, orientation=1)
        mr4k4.enable_motors('z')
        mr5k4 = optics1d.CurvedMirror('MR5K4', length=0.5, p=5.05, q=0.8, alpha=21e-3,
                                      z=766.793, orientation=0, delta=0e-6, dx=0e-6)
        mr5k4.enable_motors('z')
        # sl3k4 = optics1d.Slit('sl3k4', x_width=20e-3, y_width=20e-3, z=767.204)
        IP2 = optics1d.PPM('IP2', FOV=50e-6, z=767.593)
        pf2k4 = optics1d.WFS('PF2K4', pitch=28.5e-6, z=768.583, fraction=2, enabled=False)
        im6k4 = optics1d.PPM('IM6K4', FOV=9.4e-3, z=769.061)

        if IP_name == 'test':
            # devices related to beam steering
            device_array = [im1k0, im2k0, mr1k4, im1k4, im2k4, im3k4, mr2k4, mr3k4, im4k4, IP1, pf1k4, im5k4]
            positions = [['IM3K4', 'IM5K4']]
            screens = [['IP1', 'IM5K4', 'IM4K4']]
            xControls = [['MR2K4']]
            yControls = [['MR3K4']]
            WFS_sections = [0]
            
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[0] = {'c0': 0, 'c1': 0, 'z': im5k4.z-IP1.z, 'L3': 0}
            yGoals = xGoals.copy()

            # key is section number that has a wavefront sensor
            wfs_names = {
                    0: 'PF1K4'
                    }
            zGoals = {
                    0: im5k4.z - IP1.z
                    }

        if IP_name == 'IP 1':
            # devices related to beam steering
            device_array = [im1k0, im2k0, mr1k4, im1k4, im2k4, im3k4, mr2k4, mr3k4, im4k4, IP1, pinhole, pf1k4, im5k4]
            positions = [['IM2K0'], ['IM2K0', 'IM3K4'], ['IM3K4', 'IM5K4']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K4', 'IM3K4'], ['IP1', 'IM5K4', 'IM4K4']]
            xControls = [[None], ['MR1K4'], ['MR2K4']]
            yControls = [[None], [None], ['MR3K4']]
            WFS_sections = [2]
            
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[2] = {'c0': 0, 'c1': 0, 'z': im5k4.z-IP1.z, 'L3': 0}
            yGoals = xGoals.copy()

            # key is section number that has a wavefront sensor
            wfs_names = {
                    2: 'PF1K4'
                    }
            zGoals = {
                    2: im5k4.z - IP1.z
                    }

        elif IP_name == 'IP 2':
            device_array = [im1k0, im2k0, mr1k4, im1k4, im2k4, im3k4, mr2k4, mr3k4, im4k4,
                            IP1, pf1k4, im5k4, mr4k4, mr5k4, IP2, pf2k4, im6k4]
            positions = [['IM2K0'], ['IM2K0', 'IM3K4'],
                         ['IM3K4', 'IM5K4'], ['IM5K4', 'IM6K4']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K4', 'IM3K4'],
                       ['IP1', 'IM5K4', 'IM4K4'], ['IP2', 'IM6K4']]
            xControls = [[None], ['MR1K4'], ['MR2K4'], ['MR5K4']]
            yControls = [[None], [None], ['MR3K4'], ['MR4K4']]
            WFS_sections = [2, 3]
            
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[2] = {'c0': 0, 'c1': 0, 'z': im5k4.z-IP1.z, 'L3': 0}
            xGoals[3] = {'c0': 0, 'z': im6k4.z-IP2.z, 'L3': 0}
            yGoals = xGoals.copy()

            wfs_names = {
                    2: 'PF1K4',
                    3: 'PF2K4'
                    }
            zGoals = {
                    2: im5k4.z - IP1.z,
                    3: im6k4.z - IP2.z
                    }

    elif beamline_name == 'NEH 2.2':
        N0 = 50
        N1 = -2.44e-2
        N2 = -1.0e-5
        CFF = 1.09
        M1q = 15.36
        p0 = 1150
        E0 = 1000
        alpha0 = 13.72e-3
        beta0 = np.arccos(np.cos(alpha0)-1239.8/p0*1e-9*N0*1e3)
        im1k0 = optics1d.PPM('IM1K0', FOV=9.4e-3, z=699.47)
        im2k0 = optics1d.PPM('IM2K0', FOV=9.4e-3, z=731.61)
        mr1k1 = optics1d.CurvedMirror('MR1K1', length=0.6, p=88.773, q=M1q, alpha=18.3e-3, z=733.773, orientation=1)
        mr1k1.enable_motors('dF1', 'dF2')
        im1k1 = optics1d.PPM('IM1K1', FOV=9.4e-3, z=737.903)
        mr2k1 = optics1d.FlatMirror('MR2K1', length=1, alpha=.0149, z=739.08, orientation=1)
        mr3k1 = optics1d.Grating('MR3K1', length=.22, alpha=13.72e-3, N0=N0, N1=N1, N2=N2, 
                                 beta0=beta0, z=739.76, orientation=1)
        monoYAG = optics1d.PPM('monoYAG', FOV=40e-3, z=mr3k1.z, view_angle_y=5)
        mono = optics1d.Mono('mono', M2=mr2k1, grating=mr3k1, YAG=monoYAG, f=19.6, delta=0e-6, E0=E0, CFF=CFF)
        im2k1 = optics1d.PPM('IM2K1', FOV=9.4e-3, z=742.15)
        mr1k2 = optics1d.FlatMirror('MR1K2', length=0.4, alpha=25.2e-3, z=742.922, orientation=0)
        EXS = optics1d.Slit('exit_slit', x_width=10e-3, y_width=100e-6, z=mr3k1.z+19.6)
        EXS_YAG = optics1d.PPM('EXS_YAG', FOV=1e-3, z=EXS.z+1e-3)
        im1k2 = optics1d.PPM('IM1K2', FOV=9.4e-3, z=777.93)
        mr2k2 = optics1d.FlatMirror('MR2K2', length=0.7, alpha=30.5e-3, z=779.55, orientation=1)
        im2k2 = optics1d.PPM('IM2K2', FOV=20.6e-3, z=780.425)
        mr3k2 = optics1d.CurvedMirror('MR3K2', length=.55, p=136, q=8.85, alpha=25.2e-3, z=781.15, orientation=0)
        mr3k2.enable_motors('dF1', 'dF2')
        im3k2 = optics1d.PPM('IM3K2', FOV=20.6e-3, z=781.9)
        mr4k2 = optics1d.CurvedMirror('MR4K2', length=.8, p=23.39, q=7.25, alpha=30.53e-3, z=782.75, orientation=1)
        mr4k2.enable_motors('dF1', 'dF2')
        im4k2 = optics1d.PPM('IM4K2', FOV=9.4e-3, z=783.455)
        # ATM = optics1d.PPM('ATM', FOV=9.4e-3, z=788.476)
        ChemRIXS = optics1d.PPM('ChemRIXS', FOV=100e-6, z=790, N=256)
        qRIXS = optics1d.PPM('qRIXS', FOV=100e-6, z=785.25, N=256)
        pf1k2 = optics1d.WFS('PF1K2', pitch=30.3e-6, z=786.918, f0=1.668)
        im5k2 = optics1d.PPM('IM5K2', FOV=9.4e-3, z=787.417, N=2048, blur=True)
        pf2k2 = optics1d.WFS('PF2K2', pitch=30.3e-6, z=791.668+.02, fraction=2, f0=1.668)
        im6k2 = optics1d.PPM('IM6K2', FOV=9.4e-3, z=792.167, N=2048, blur=True)
        
        if IP_name == 'Exit Slit':
            device_array = [im1k0, im2k0, mr1k1, im1k1, mono, im2k1, mr1k2,
                            EXS, EXS_YAG]
            positions = [['IM2K0'], ['IM2K0', 'mono'], ['mono', 'EXS_YAG']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K1', 'mono'], ['IM2K1', 'EXS_YAG']]
            xControls = [[None], [None], ['MR1K2']]
            yControls = [[None], ['MR1K1'], [None]]
            WFS_sections = []
            
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            yGoals = xGoals.copy()

            wfs_names = {}
            zGoals = {}

        elif IP_name == 'ChemRIXS':
            device_array = [im1k0, im2k0, mr1k1, im1k1, mono, im2k1, mr1k2, EXS, EXS_YAG,
                            im1k2, mr2k2, im2k2, mr3k2, im3k2, mr4k2, im4k2, ChemRIXS, im5k2,
                            pf2k2, im6k2]
            positions = [['IM2K0'], ['IM2K0', 'mono'], ['mono', 'EXS_YAG'], ['EXS_YAG', 'IM2K2'], ['IM2K2', 'IM6K2']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K1', 'mono'], ['IM2K1', 'EXS_YAG'],
                       ['IM1K2', 'IM2K2'], ['ChemRIXS', 'IM6K2', 'IM4K2']]
            xControls = [[None], [None], ['MR1K2'], [None], ['MR3K2']]
            yControls = [[None], ['MR1K1'], [None], ['MR2K2'], ['MR4K2']]
            WFS_sections = [4]
            
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[4] = {'c0': 0, 'c1': 0, 'z': im6k2.z-ChemRIXS.z, 'L3': 0}

            yGoals = xGoals.copy()
            wfs_names = {
                    4: 'PF2K2'
                    }
            zGoals = {
                    4: im6k2.z-ChemRIXS.z
                    }

        elif IP_name == 'qRIXS':
            device_array = [im1k0, im2k0, mr1k1, im1k1, mono, im2k1, mr1k2, EXS, EXS_YAG,
                            im1k2, mr2k2, im2k2, mr3k2, im3k2, mr4k2, im4k2, qRIXS, pf1k2, im5k2,
                            im6k2]
            positions = [['IM2K0'], ['IM2K0', 'mono'], ['mono', 'EXS_YAG'], ['EXS_YAG', 'IM2K2'],
                         ['IM2K2', 'IM5K2'], ['IM2K2', 'IM6K2']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K1', 'mono'], ['IM2K1', 'EXS_YAG'],
                       ['IM1K2', 'IM2K2'], ['IM4K2', 'IM5K2'], ['qRIXS', 'IM6K2']]
            xControls = [[None], [None], ['MR1K2'], [None], ['MR3K2'], ['MR3K2']]
            yControls = [[None], ['MR1K1'], [None], ['MR2K2'], ['MR4K2'], ['MR4K2']]
            WFS_sections = [5]

            xGoals = [{'cx1': 0, 'cx2': 0} for position in positions]
            xGoals[5] = {'cx1': 0, 'cx2': 0, 'zf': im6k2.z-qRIXS.z, 'c3': 0}
            yGoals = xGoals.copy()
            wfs_names = {
                    5: 'PF1K2'
                    }
            zGoals = {
                    5: pf1k2.z - qRIXS.z
                    }

    elif beamline_name == 'TXI':
        if IP_name == 'SXR':
            im1k0 = optics1d.PPM('IM1K0', FOV=9.4e-3, z=699.47)
            im2k0 = optics1d.PPM('IM2K0', FOV=9.4e-3, z=731.61)
            mr1k3 = optics1d.FlatMirror('MR1K3', length=1., alpha=-9.85e-3, z=735.422, orientation=0)
            mr2k3 = optics1d.FlatMirror('MR2K3', length=1., alpha=-9.85e-3, z=737.022, orientation=0)
            im1k3 = optics1d.PPM('IM1K3', FOV=9.4e-3, z=740.804)
            mr3k3 = optics1d.CurvedMirror('MR3K3', length=0.9, alpha=7e-3, z=770.66, p=125, q=3.500, orientation=1)
            mr4k3 = optics1d.CurvedMirror('MR4K3', length=0.9, alpha=7e-3, z=771.66, p=125, q=2.5, orientation=0)
            IP_SXR = optics1d.PPM('IP SXR', FOV=200e-6, z=774.16)
            pf1k3 = optics1d.WFS('PF1K3', pitch=32.9e-6, z=777.76, f0=3.6, fraction=2)
            im3k3 = optics1d.PPM('IM3K3', FOV=9.4e-3, z=778.66, blur=True)

            device_array = [im1k0, im2k0, mr1k3, mr2k3, im1k3, mr3k3, mr4k3, IP_SXR, pf1k3, im3k3]
            positions = [['IM2K0'], ['IM2K0', 'IM1K3'], ['IM1K3', 'IM3K3']]
            screens = [['IM1K0', 'IM2K0'], ['IM1K3', 'IM1K3'], ['IP SXR', 'IM3K3']]
            xControls = [[None], ['MR1K3', 'MR2K3'], ['MR4K3']]
            yControls = [[None], [None], ['MR3K3']]
            WFS_sections = [2]
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[2] = {'c0': 0, 'z': im3k3.z-IP_SXR.z, 'L3': 0}
            yGoals = xGoals.copy()
            wfs_names = {2: 'PF1K3'}
            zGoals = {2: im3k3.z-IP_SXR.z}
        elif IP_name == 'HXR':
            im1l0 = optics1d.PPM('IM1L0', FOV=9.4e-3, z=699.55)
            im2l0 = optics1d.PPM('IM2L0', FOV=9.4e-3, z=735.9867)
            mr1l0 = optics1d.FlatMirror('MR1L0', length=1., alpha=7e-3, z=740, orientation=0)
            mr1l1 = optics1d.FlatMirror('MR1L1', length=1., alpha=7e-3, z=741.6, orientation=0)
            im1l1 = optics1d.PPM('IM1L1', FOV=9.4e-3, z=745.4)
            mr2l1 = optics1d.CurvedMirror('MR2L1', length=0.9, z=770.66, p=125, q=3.5, orientation=1)
            mr3l1 = optics1d.CurvedMirror('MR3L1', length=0.9, z=771.66, p=125, q=2.5, orientation=0)
            IP_HXR = optics1d.PPM('IP HXR', FOV=200e-6, z=774.16)
            pf1l1 = optics1d.WFS('PF1L1', pitch=31.85e-6, z=777.76, f0=3.6)
            im3l1 = optics1d.PPM('IM3L1', FOV=9.4e-3, z=778.96, blur=True)

            device_array = [im1l0, im2l0, mr1l0, mr1l1, im1l1, mr2l1, mr3l1, IP_HXR, pf1l1, im3l1]
            positions = [['IM2L0'], ['IM2L0', 'IM1L1'], ['IM1L1', 'IM3L1']]
            screens = [['IM1L0', 'IM2L0'], ['IM1L1', 'IM1L1'], ['IP HXR', 'IM3L1']]
            xControls = [[None], ['MR1L0', 'MR1L1'], ['MR3L1']]
            yControls = [[None], [None], ['MR2L1']]
            WFS_sections = [2]
            xGoals = [{'c0': 0, 'c1': 0} for position in positions]
            xGoals[2] = {'c0': 0, 'z': im3l1.z - IP_HXR.z, 'L3': 0}
            yGoals = xGoals.copy()
            wfs_names = {2: 'PF1K3'}
            zGoals = {2: im3l1.z - IP_HXR.z}

    # put parameters in a dict
    parameters = {
            'positions': positions,
            'screens': screens,
            'xControls': xControls,
            'yControls': yControls,
            'WFS_sections': WFS_sections,
            'xGoals': xGoals,
            'yGoals': yGoals,
            'wfs_names': wfs_names,
            'zGoals': zGoals
            }

    # set up beamline
    beamline_object = beamline.Beamline(device_array)

    return beamline_object, parameters
