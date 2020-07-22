import numpy as np
import os
import csv
import matplotlib.pyplot as plt

class Device:

    def __init__(self,name=None,range=None,material=None,angle=np.pi/2, thickness=None, z=None):

        self.range = range
        self.material = material
        self.angle = angle
        self.name = name
        self.thickness = thickness
        self.z = z

        self.get_atomic_mass()
        self.load_index()
        self.get_melt_temp()
        self.get_absorption_edges()
        self.get_heat_capacity()
        self.get_atoms_per_mol()
        self.get_Z()
        self.electron_penetration()

    def get_beam_parameters(self):
        filename = os.path.join(os.path.dirname(__file__), 'data/%s_source.csv' % self.range)

        beam_data = np.genfromtxt(filename, delimiter=',', skip_header=1)
        beam_photon_energy = beam_data[:,0]
        beam_source = beam_data[:,2]
        # FWHM divergence (microrad converted to rad)
        beam_fwhm_div = beam_data[:,3]*1e-6



    def get_heat_capacity(self):
        # units are J/(mol*K)
        cap_dict = {
            'Au': 25.6,
            'Si': 19.8,
            'C': 6.1,
            'CVD': 6.1,
            'B4C': 54,
            'Ni': 25.97,
            'Rh': 24.98,
            'SiC': 26.74,
            'W': 24.31,
            'YAG': 350
        }
        self.heat_capacity = cap_dict[self.material]

    def get_melt_temp(self):
        temp_dict  = {
            'Au': 1337,
            'Si': 1687,
            'C': 3800,
            'CVD': 3800,
            'B4C': 3036,
            'Ni': 1728,
            'Rh': 2237,
            'SiC': 3003,
            'W': 3695,
            'YAG': 1900
        }
        self.melt_temp = temp_dict[self.material]


    def get_atomic_mass(self):
        mass_dict = {
            'Au': 197,
            'Si': 28,
            'C': 12,
            'CVD': 12,
            'B4C': (10.8*4+12)/5.,
            'Ni': 58,
            'Rh': 102.9,
            'SiC': 40.1/2.,
            'W': 183.8,
            'YAG': (88.91*3 + 26.98*5 + 15.999*12)/20.
        }
        self.mass = mass_dict[self.material]

    def get_atoms_per_mol(self):
        atoms_dict = {
            'Au': 1,
            'Si': 1,
            'C': 1,
            'CVD': 1,
            'B4C': 5,
            'Ni': 1,
            'Rh': 1,
            'SiC': 2,
            'W': 1,
            'YAG': 20
        }
        self.atoms = atoms_dict[self.material]

    def get_Z(self):
        Z_dict = {
            'Au': 79,
            'Si': 14,
            'C': 6,
            'CVD': 6,
            'B4C': (5*4+6)/5,
            'Ni': 28,
            'Rh': 45,
            'SiC': (14+6)/2,
            'W': 74,
            'YAG': (39*3 + 13*5 + 8*12)/20
        }
        self.Z = Z_dict[self.material]

    def get_absorption_edges(self):
        edge_dict = {
            'Au': [2206,2743,11919,13734,14353],
            'Si': [149.7,1839],
            'C': [284],
            'CVD': [284],
            'B4C': [188,284],
            'Ni': [110.8,853,8333],
            'Rh': [81.4,307,3004,3146,3412,23220],
            'SiC': [149.7,284,1839],
            'W': [33.6,1809,1872,2281,2575,2820,10207,11544,12100],
            'YAG': [73, 543, 1559.6, 2080, 2156, 2373, 17038]
        }
        self.absorption_edges = np.array(edge_dict[self.material])

    def electron_penetration(self):
        # g/cm^2/keV
        A = ( ( 0.81 * ( self.Z**(-.38) ) ) +0.18) * .001
        B = 0.21 * (self.Z**(-.055)) + 0.78
        # 1/keV
        C = ( 1.1 * self.Z**.29 + 0.21 ) * .001

        # photo-electron KE (keV)
        self.peKE = np.zeros(self.energy.size)

        for i in range(self.energy.size):

            mask = self.absorption_edges<self.energy[i]

            if np.sum(mask)>0:
                edge = np.max(self.absorption_edges[mask])
            else:
                edge = 0.

            # photo-electron kinetic energy (keV)
            self.peKE[i] = (self.energy[i] - edge)/1000

        # electron penetration depth
        self.penetration = ( ( A * self.peKE ) * ( 1 - ( B / ( 1 + C * self.peKE )))) / self.density * .01 / 10

    def load_index(self):

        self.filename = os.path.join(os.path.dirname(__file__), 'data/%s_%s.csv' % (self.material, self.range))

        #self.filename = './'+self.material+'_'+self.range+'.csv'

        density = 0.

        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            i = 0
            for line in reader:
                if i > 0:
                    continue
                densityText = line[1]
                i1 = densityText.find('=')
                density = float(densityText[i1+1:])
                i += 1

        self.density = density
        # density in atoms/m^3
        self.rho = self.density*100**3/self.mass/1.66e-24

        cxro_data = np.genfromtxt(self.filename, delimiter=',', skip_header=2)
        self.energy = cxro_data[:,0]
        self.wavelength = 1239.8/self.energy*1e-9
        self.delta = cxro_data[:,1]
        self.beta = cxro_data[:,2]

        self.index = 1-self.delta+1j*self.beta

    def transmission(self, thickness=None):
        if thickness is None:
            try:
                thickness = self.thickness
            except AttributeError:
                print("No thickness information. Will assume zero thickness.")
                thickness = 0.0
        T = np.exp(-2 * np.pi * self.beta * thickness / (1240. / self.energy * 1e-9)) ** 2

        return T

    def get_thickness(self, transmission, E0):
        """
        Method to get material thickness corresponding to desired transmission
        """
        # interpolate to find beta at this energy
        beta = np.interp(E0, self.energy, self.beta)

        thickness = -np.log(transmission)/2/2/np.pi/beta*1240./E0*1e-9

        return thickness

    def reflectivity(self):

        kiz = 2 * np.pi / self.wavelength * np.sin(self.angle)
        ktz = 2 * np.pi / self.wavelength * np.sqrt(self.index ** 2 - np.cos(self.angle) ** 2)
        R = np.abs((kiz - ktz) / (kiz + ktz)) ** 2
        return R

    def energy_limit(self, x_FWHM, y_FWHM, factor=2, dose=None):

        # d = (1240/energy*1e-9)/(4*np.pi*beta)
        # Note that studies have found consistency with ~30nm energy transport from electrons!
        # d = np.sqrt(self.attenuation_length()**2 + (30e-9)**2)
        d = np.sqrt(self.attenuation_length()**2 + self.penetration**2)
        # d = self.attenuation_length()
        R = self.reflectivity()


        dose_applied = 0

        if dose is None:
            dose_applied = self.melt_dose()/factor
        else:
            dose_applied=dose

        # multiplied by 1000 to convert to mJ
        Ep = 1.6e-19 * dose_applied * np.pi * (x_FWHM / 1.18) * (y_FWHM / 1.18) * d * self.rho * 1000 / 2 / (1 - R) / np.sin(self.angle)
        # Ep = 1.6e-19*maxDose*np.pi*(FWHM)**2*d*rho*1000/2
        # Ep = 1.6e-19*maxDose*FWHM**2*d*rho*1000

        return Ep

    def absorbed_dose(self, x_FWHM, y_FWHM, Ep):
        d = np.sqrt(self.attenuation_length() ** 2 + self.penetration ** 2)
        # d = self.attenuation_length()
        R = self.reflectivity()

        # eV/atom
        dose_applied = Ep/1.6e-19/np.pi/(x_FWHM / 1.18)/(y_FWHM / 1.18)/(d * self.rho * 1000 / 2 / (
                    1 - R) / np.sin(self.angle))

        return dose_applied

    def plot_dose(self,x_FWHM,y_FWHM,energy_ref=None,axis=None,reflectivity=None, Ep=10):

        if energy_ref:
            x_FWHM = x_FWHM * energy_ref / self.energy
            y_FWHM = y_FWHM * energy_ref/self.energy


        dose = self.absorbed_dose(x_FWHM, y_FWHM, Ep)

        if axis is None:
            fig, axis = plt.subplots()

        axis.semilogy(self.energy,dose,label='Absorbed dose')


        axis.semilogy(self.energy,np.ones(self.energy.size)*self.melt_dose(),label='Melt dose')

        if reflectivity is not None:
            axis.semilogy(self.energy,dose*reflectivity,label='Absorbed dose (accounts for R)')

        axis.legend()
        axis.grid(b=True,which='both')
        axis.set_xlabel('Photon Energy (eV)')
        axis.set_ylabel('Dose (eV/atom)')

        return axis




    def energy_limit_absorb(self, x_FWHM, y_FWHM, factor=2, dose=None):
        # d = (1240/energy*1e-9)/(4*np.pi*beta)
        # Note that studies have found consistency with ~30nm energy transport from electrons!
        # d = np.sqrt(self.attenuation_length() ** 2 + (30e-9) ** 2)
        d = np.sqrt(self.attenuation_length()**2 + self.penetration**2)
        # d = self.attenuation_length()
        R = 0.

        dose_applied = 0

        if dose is None:
            dose_applied = self.melt_dose() / factor
        else:
            dose_applied = dose

        # multiplied by 1000 to convert to mJ
        Ep = 1.6e-19 * dose_applied * np.pi * (x_FWHM / 1.18) * (y_FWHM / 1.18) * d * self.rho * 1000 / 2 / (
                    1 - R) / np.sin(self.angle)
        # Ep = 1.6e-19*maxDose*np.pi*(FWHM)**2*d*rho*1000/2
        # Ep = 1.6e-19*maxDose*FWHM**2*d*rho*1000

        return Ep

    def plot_reflectivity(self,axis=None):
        if axis is None:
            fig, axis = plt.subplots()

        axis.plot(self.energy,self.reflectivity())
        axis.grid(which='both')
        axis.set_xlabel('Photon Energy (eV)')
        axis.set_ylabel('Reflectivity')
        return axis

    def save_energy_limit(self,filename,x_FWHM,y_FWHM,energy_ref,reflectivity):

        limit = self.energy_limit(x_FWHM * energy_ref / self.energy, y_FWHM * energy_ref/self.energy, factor=10)

        data = np.zeros((np.size(self.energy),3))
        data[:,0] = self.energy
        data[:,1] = limit
        data[:,2] = limit/reflectivity

        np.savetxt(filename,data,fmt='%.2e',delimiter=',',
                   header='Photon Energy (eV), Energy limit (mJ), Energy Limit with reflectivity (mJ)')

        limit_R = data[:,2]

        return limit, limit_R

    def plot_energy_limit_full_absorption(self,x_FWHM,y_FWHM,energy_ref,axis=None,incident=None,reflectivity=None, factor=10):

        # d = np.sqrt(self.attenuation_length() ** 2 + (30e-9) ** 2)
        d = np.sqrt(self.attenuation_length() ** 2 + self.penetration**2)
        # d = self.attenuation_length()
        R = 0

        dose_applied = self.melt_dose() / factor

        x_FWHM = x_FWHM * energy_ref / self.energy
        y_FWHM = y_FWHM * energy_ref / self.energy

        limit = 1.6e-19 * dose_applied * np.pi * (x_FWHM / 1.18) * (y_FWHM / 1.18) * d * self.rho * 1000 / 2 / (
                    1 - R) / np.sin(self.angle)

        if axis is None:
            fig, axis = plt.subplots()

        axis.semilogy(self.energy, limit, label='Damage Limit')

        if incident is not None:
            axis.semilogy(self.energy, incident, label='Max Incident')

        if reflectivity is not None:
            axis.semilogy(self.energy, limit / reflectivity, label='Max allowed (accounts for R)')

        axis.legend()
        axis.grid(b=True, which='both')
        axis.set_xlabel('Photon Energy (eV)')
        axis.set_ylabel('Energy limit (mJ)')
        axis.set_title('Max energy with full absorption')

        return axis

    def MPS_limits(self,filename,x_FWHM,y_FWHM,energy_ref,reflectivity):

        limit = self.energy_limit(x_FWHM * energy_ref / self.energy, y_FWHM * energy_ref / self.energy, factor=10)

        limit_absorption = self.energy_limit_absorb(x_FWHM * energy_ref / self.energy,
                                                    y_FWHM * energy_ref / self.energy, factor=10)

        if self.range == 'SXR':
            N = (2050 - 250) / 100 + 1

            E_MPS = np.linspace(250, 2050, N)
        else:
            N = (25000-1000)/1000 + 1

            E_MPS = np.linspace(1000,25000, N)

        MPS_limit_noR = np.interp(E_MPS, self.energy, limit)
        MPS_limit = np.interp(E_MPS, self.energy, limit/reflectivity)

        MPS_absorb_noR = np.interp(E_MPS, self.energy, limit_absorption)
        MPS_absorb = np.interp(E_MPS, self.energy, limit_absorption/reflectivity)

        data = np.zeros((np.size(E_MPS), 3))
        data[:, 0] = E_MPS
        data[:, 1] = self.safe_limit(E_MPS,MPS_limit_noR,MPS_absorb_noR)
        data[:, 2] = self.safe_limit(E_MPS,MPS_limit,MPS_absorb)

        np.savetxt(filename, data, fmt='%.2e', delimiter=',',
                   header='Photon Energy (eV), Energy limit (mJ), Energy Limit with reflectivity (mJ)')

        return data

    def safe_limit(self,energy,limit,limit_absorb):

        cons_limit = np.copy(limit)
        cons_absorb = np.copy(limit_absorb)

        cons_limit[0] = np.floor(np.min(limit[0:2])*10)/10
        cons_absorb[0] = np.floor(np.min(limit_absorb[0:2])*10)/10

        for i in range(1, limit.size - 1):
            cons_limit[i] = np.floor(np.min(limit[i - 1:i + 2])*10)/10
            cons_absorb[i] = np.floor(np.min(limit_absorb[i - 1:i + 2])*10)/10

        cons_limit[-1] = np.floor(np.min(limit[-2:])*10)/10
        cons_absorb[-1] = np.floor(np.min(limit_absorb[-2:])*10)/10

        delta_E = energy[1]-energy[0]
        for i in range(limit.size):
            for edge in self.absorption_edges:
                if np.abs(edge-energy[i])<delta_E:
                    cons_limit[i] = cons_absorb[i]


        return cons_limit

    def plot_energy_limit(self,x_FWHM,y_FWHM,energy_ref,axis=None,incident=None,reflectivity=None, factor=10):

        limit = self.energy_limit(x_FWHM * energy_ref / self.energy, y_FWHM * energy_ref/self.energy, factor=factor)

        if axis is None:
            fig, axis = plt.subplots()

        axis.loglog(self.energy,limit,label='Damage Limit')

        if incident is not None:
            axis.loglog(self.energy,incident,label='Max Incident')

        if reflectivity is not None:
            axis.loglog(self.energy,limit/reflectivity,label='Max allowed (accounts for R)')

        axis.legend()
        axis.grid(b=True,which='both')
        axis.set_xlabel('Photon Energy (eV)')
        axis.set_ylabel('Energy limit (mJ)')

        return axis

    def fluence_limit(self, dose=None):

        # d_total = np.sqrt(self.attenuation_length()**2 + (30e-9)**2)
        d_total = np.sqrt(self.attenuation_length()**2 + self.penetration**2)
        # d_total = self.attenuation_length()

        R = self.reflectivity()

        dose_applied = 0
        if dose is None:
            dose_applied = self.melt_dose()
        else:
            dose_applied = dose

        F_th = 1.6e-19 * dose_applied * d_total * self.rho / (1-R) / np.sin(self.angle) * 1e-6

        return F_th

    def temp_rise(self, FWHM, pulse_energy=1.):

        kB = 8.617e-5
        #d = np.sqrt(self.attenuation_length() ** 2 + (30e-9) ** 2)
        d = np.sqrt(self.attenuation_length() ** 2 + self.penetration**2)
        # d = self.attenuation_length()
        R = self.reflectivity()

        dose = pulse_energy/1.6e-19/np.pi/(FWHM/1.18)**2/d/self.rho/1000*2*(1-R)*np.sin(self.angle)

        deltaT = dose/3/kB
        return deltaT


    def attenuation_length(self):

        d = self.wavelength / 4 / np.pi / np.imag(self.index * np.sqrt(1 - (np.cos(self.angle) / self.index) ** 2))
        return d

    def index_from_energy(self, energy0):

        i0 = np.argmax(self.energy > energy0)

        return self.index[i0]

    def melt_dose(self):

        deltaT = self.melt_temp - 293

        # eV/atom
        dose = self.heat_capacity*deltaT/6.022e23/1.602e-19/self.atoms

        # deltaT = self.melt_temp-293
        # kB = 8.617e-5
        # dose = 3*kB*deltaT

        return dose


class Mirror(Device):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Crystal(Device):

    def __init__(self, hkl=None, alpha=0, asymmetry_type='incidence', **kwargs):
        """
        Initiailize a Crystal. Only supports hard x-rays for now.
        Parameters
        ----------
        hkl: Miller indices
        alpha
        asymmetry_type
        kwargs
        """
        super().__init__(**kwargs)

        self.hkl = hkl
        self.alpha = alpha
        self.asymmetry_type=asymmetry_type




class DeviceCollection:

    def __init__(self, *devices, **mono):

        # name devices if they're not named yet, set as attributes
        # also put devices in a list
        self.devices = devices
        for num, device in enumerate(devices):
            if device.name is None:
                device.name = 'Device%s' % num
            setattr(self, device.name, device)

        if 'mono' in mono.keys():
            setattr(self, 'mono_reflectivity', mono['mono'])

    def calc_total_transmission(self):

        try:
            energy = self.devices[0].energy
            total_T = np.ones_like(energy)
        except IndexError:
            energy = 0
            total_T = 1

        for num, device in enumerate(self.devices):
            if isinstance(device, Mirror):
                total_T *= device.reflectivity()
            else:
                total_T *= device.transmission()

        if hasattr(self, 'mono_reflectivity'):
            total_T *= getattr(self, 'mono_reflectivity')

        return energy, total_T

    def plot_total_transmission(self, axis=None):

        energy, total_T = self.calc_total_transmission()

        if axis is None:
            fig, axis = plt.subplots()

        axis.plot(energy, total_T)
        axis.set_xlabel('Photon Energy (eV)')
        axis.set_ylabel('Cumulative beamline transmission')
        axis.grid(b=True, which='both')

        return axis

    def add_devices(self, *devices):

        # get total number of devices we already have
        num0 = len(self.devices)

        # add devices to list
        self.devices.extend(devices)
        # set as attributes
        for num, device in enumerate(devices):
            if device.name is None:
                device.name = 'Device%s' % (num + num0 - 1)
            setattr(self, device.name, device)
