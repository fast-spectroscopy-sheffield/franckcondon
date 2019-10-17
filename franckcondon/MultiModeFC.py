import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class MultiModeFC:
    
    def __init__(self):
        self.spectrum_type = 'pl'  # or 'abs'
        self.num_modes = None
        self.num_replicas = 6

    def initialise(self, num_modes):
        self.num_modes = num_modes
        self._calculate_mi_permutations()
    
    @staticmethod
    def _gaussian_lineshape(x, xc, w):
        return np.exp(-((x-xc)**2)/(2*w*w))
    
    @staticmethod
    def _franck_condon_factor(S, m):
        return (S**m)/np.math.factorial(m)
    
    def _calculate_mi_permutations(self):
        p = product(range(self.num_replicas), repeat=self.num_modes)
        self._permutations = np.array(list(p))
    
    def _calculate_vibrational_energy(self, m_i, vib_energies):
        E_vib = 0
        for i in range(len(m_i)):
            m = m_i[i]
            hw = vib_energies[i]
            E_vib += m*hw
        return E_vib
    
    def _calculate_intensity(self, m_i, hr_params):
        I = 1
        for i in range(len(m_i)):
            m = m_i[i]
            S = hr_params[i]
            fcf = self._franck_condon_factor(S, m)
            I *= fcf
        return I
            
    def _calculate_peak(self, m_i, x, vib_energies, hr_params, energy_00, broadening):
        I = self._calculate_intensity(m_i, hr_params)
        E_vib = self._calculate_vibrational_energy(m_i, vib_energies)
        sign = 1 if self.spectrum_type == 'abs' else -1
        peak = I*self._gaussian_lineshape(x, energy_00+(sign*E_vib), broadening)
        return peak
    
    def calculate_fc_progression(self, x, vib_energies, hr_params, energy_00, broadening):
        model = np.zeros_like(x)
        for m_i in self._permutations:
            peak = self._calculate_peak(m_i, x, vib_energies, hr_params, energy_00, broadening)
            model += peak
        return model
    
    def plot_modes(self, x, model, vib_energies, hr_params, energy_00, broadening):
        fig, ax = plt.subplots()
        ax.plot(x, model, 'k-')
        for m_i in self._permutations:
            if sum(m_i) <= 1:
                peak = self._calculate_peak(m_i, x, vib_energies, hr_params, energy_00, broadening)
                ax.plot(x, peak)
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('PL (arb.)' if self.spectrum_type == 'pl' else 'Absorbance (arb.)')
        ax.set_xlim([min(x), max(x)])
        return fig, ax
 
       
if __name__ == '__main__':
    # setup parameters
    hr_params = [0.67, 0.1, 0.52, 0.46]
    vib_energies = [0.054, 0.089, 0.165, 0.188]
    energy_00 = 3.050
    broadening = 0.027
    x = np.linspace(2.5, 3.1, 1000)
    
    # do the calculation for the 4 modes
    mmfc = MultiModeFC()
    mmfc.initialise(4)
    model = mmfc.calculate_fc_progression(x, vib_energies, hr_params, energy_00, broadening)
    
    # plot the results
    mmfc.plot_modes(x, model, vib_energies, hr_params, energy_00, broadening)
