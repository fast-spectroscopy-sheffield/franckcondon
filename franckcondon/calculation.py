import numpy as np
import matplotlib.pyplot as plt
from itertools import product


class MultiModeFC:
    """
    Class for calculating Franck-Condon Progressions.
    
    Refer to the examples for more detailed guidelines.
    
    Attributes
    ----------
    spectrum_type : str 
        either 'pl' (photoluminescence) or 'abs' (absorption)
    num_modes : int
        the number of vibrational modes to be included
    num_replicas : int
        the number of vibronic replicas to be calculated
        
    """
    
    def __init__(self):
        """Initialise MultiModeFC."""
        self.spectrum_type = 'pl'  # or 'abs'
        self.num_modes = 1
        self.num_replicas = 6

    def initialise(self, num_modes):
        """
        Set the number of vibrational modes and calculate the permutations of vibrational quanta.
        
        Parameters
        ----------
        num_modes : int
            the number of vibrational modes to be included
            
        Returns
        -------
        None.
            
        """
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
        """
        Calculate the multi-mode Franck-Condon progression.
        
        Parameters
        ----------
        x : numpy.ndarray
            1D array containing the energy values to use in the calculation.
        vib_energies : list[float]
            The vibrational energies of the modes.
        hr_params : list[float]
            The Huang-Rhys parameters of the modes.
        energy_00 : float
            The energy of the 0-0 transition.
        broadening : float
            The linewidth broadening.
            
        Returns
        -------
        numpy.ndarray
            1D array containing the calculated Franck-Condon progression.
            
        """
        model = np.zeros_like(x)
        for m_i in self._permutations:
            peak = self._calculate_peak(m_i, x, vib_energies, hr_params, energy_00, broadening)
            model += peak
        return model
    
    def plot_modes(self, x, model, vib_energies, hr_params, energy_00, broadening):
        """
        Create a plot showing the total Franck-Condon progression and mode-resolved 0-0 and 0-1 peaks.
        
        Parameters
        ----------
        x : numpy.ndarray
            1D array containing the energy values to use in the calculation.
        vib_energies : list[float]
            The vibrational energies of the modes.
        hr_params : list[float]
            The Huang-Rhys parameters of the modes.
        energy_00 : float
            The energy of the 0-0 transition.
        broadening : float
            The linewidth broadening.
            
        Returns
        -------
        matplotlib.figure.Figure
            A figure handle for the generated plot.
        matplotlib.axes.Axes
            An axes object for the generated plot.
            
        """
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
 