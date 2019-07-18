import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class MultiModeFC:
    
    def __init__(self):
        self._initialise_params()

    def _initialise_params(self):
        self.hr_params = [0.67, 0.1, 0.52, 0.46]
        self.vib_energies = [0.054, 0.089, 0.165, 0.188]
        self.energy_00 = 3.050
        self.broadening = 0.027
        self.num_replicas = 6
        self.update()
        self.x = np.linspace(2.5, 3.1, 1000)
        
    def update(self):
        self.num_modes = len(self.hr_params)
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
    
    def _calculate_vibrational_energy(self, m_i):
        E_vib = 0
        for i in range(len(m_i)):
            m = m_i[i]
            hw = self.vib_energies[i]
            E_vib += m*hw
        return E_vib
    
    def _calculate_intensity(self, m_i):
        I = 1
        for i in range(len(m_i)):
            m = m_i[i]
            S = self.hr_params[i]
            fcf = self._franck_condon_factor(S, m)
            I *= fcf
        return I
            
    def _calculate_peak(self, m_i):
        I = self._calculate_intensity(m_i)
        E_vib = self._calculate_vibrational_energy(m_i)
        peak = I*self._gaussian_lineshape(self.x, self.energy_00-E_vib, self.broadening)
        return peak
    
    def calculate_fc_progression(self):
        model = np.zeros_like(self.x)
        self.peaks_to_plot = {}
        for count, m_i in enumerate(self._permutations):
            peak = self._calculate_peak(m_i)
            if sum(m_i) <= 1:
                self.peaks_to_plot[count] = peak
            model += peak
        self.model = model
 
       
if __name__ == '__main__':
    mmfc = MultiModeFC()
    mmfc.calculate_fc_progression()
    plt.figure()
    plt.plot(mmfc.x, mmfc.model, 'k-')
    for peak in mmfc.peaks_to_plot.values():
        plt.plot(mmfc.x, peak)
    plt.ylim([0, 1.3])
    plt.xlabel('energy (eV)')
    plt.ylabel('PL (arb.)')
