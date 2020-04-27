from franckcondon.calculation import MultiModeFC
from scipy.optimize import least_squares
import numpy as np


class MultiModeFCFitting(MultiModeFC):
    """
    Class for fitting Franck-Condon progressions to experimental data.
    
    Refer to the examples for more detailed guidelines.
    
    Attributes
    ----------
    spectrum_type : str 
        either 'pl' (photoluminescence) or 'abs' (absorption)
    num_modes : int
        the number of vibrational modes to be included
    num_replicas : int
        the number of vibronic replicas to be calculated
    refractive_index : int
        refractive index of the surroundings (arbitrary in practice)
    fit_vibrational_energies : {str, list[int]}
        how many vibrational energies to fit. Either 'all', 'None' or a list of indicies
    fit_hr_params : {str, list[int]}
        how many Huang-Rhys parameters to fit. Either 'all', 'None' or a list of indicies
    fit_broadening : book
        whether to fit the linewidth
    fit_energy_00 : bool
        whether to fit the energy of the 0-0 transition
    fit_scaling_factor :bool
        whether to fit the scaling factor (highly recommended)
    algorithm : str
        least_squares algorithm to use (see scipy.optimize.least_squares)
    enforce_non_negativity : bool
        whether to force all fitted parameters to have positive values (recommended)
    
    """
    
    def __init__(self):
        """Initialise MultiModeFCFitting.
        
        By default everything except vibrational energies ae fitted. The
        algorithm is trf and non-negativity is enforced.
        
        """
        super().__init__()
        self.refractive_index = 1
        self.fit_vibrational_energies = 'None'
        self.fit_hr_params = 'all'
        self.fit_broadening = True
        self.fit_energy_00 = True
        self.fit_scaling_factor = True
        self.algorithm = 'trf'
        self.enforce_non_negativity = True
        
    def correct_for_phonon_dos(self):
        """Correct the experimental data for the phonon density of states.
        
        Returns
        -------
        None.
            
        """
        self.y /= (self.x*self.refractive_index)**3
        
    def input_data(self, x, y):
        """
        Load the experimental data.

        Parameters
        ----------
        x : numpy.ndarray
            1D array containing the energy values.
        y : numpy.ndarray
            1D array containing the spectrum values.

        Returns
        -------
        None.

        """
        self.x = x
        self.y = y
        
    def input_parameters(self, vib_energies, hr_params, energy_00, broadening, scaling_factor):
        """
        Load the Franck-Condon parameters.
        
        They are either fixed or used as starting points for the fitting.

        Parameters
        ----------
        vib_energies : list[float]
            The vibrational energies of the modes.
        hr_params : list[float]
            The Huang-Rhys parameters of the modes.
        energy_00 : float
            The energy of the 0-0 transition.
        broadening : float
            The linewidth broadening.
        scaling_factor : float
            A factor to scale the data in order to match the fit (which is 
            normalised to approximately 1)

        Returns
        -------
        None.

        """
        self.vib_energies = vib_energies
        self.hr_params = hr_params
        self.energy_00 = energy_00
        self.broadening = broadening
        self.scaling_factor = scaling_factor
        
    def _create_tofit_lists(self):
        if self.fit_vibrational_energies == 'None':
            self.fit_vibrational_energies = []
        elif self.fit_vibrational_energies == 'all':
            self.fit_vibrational_energies = list(range(self.num_modes))
        if self.fit_hr_params == 'None':
            self.fit_hr_params = []
        elif self.fit_hr_params == 'all':
            self.fit_hr_params = list(range(self.num_modes))
        
    def _get_params_tofit_from_list(self, list_tofit):
        tofit = [True if (i in list_tofit) else False for i in range(self.num_modes)]
        return np.array(tofit, dtype=bool)
    
    def _construct_parameter_array(self):
        return np.array([self.scaling_factor, self.energy_00, self.broadening, *self.vib_energies, *self.hr_params])
    
    def _construct_parameters_tofit_mask(self):
        vib_energies_mask = self._get_params_tofit_from_list(self.fit_vibrational_energies)
        hr_params_mask = self._get_params_tofit_from_list(self.fit_hr_params)
        return np.array([self.fit_scaling_factor, self.fit_energy_00, self.fit_broadening, *vib_energies_mask, *hr_params_mask], dtype=bool)
        
    def _calculate_model(self, params):
        scaling_factor, energy_00, broadening, vib_energies, hr_params = self._unpack_parameter_array(params)
        model = self.calculate_fc_progression(self.x, vib_energies, hr_params, energy_00, broadening)
        return model, scaling_factor
    
    def _unpack_parameter_array(self, params):
        scaling_factor = params[0]
        energy_00 = params[1]
        broadening = params[2]
        vib_energies = params[3:3+self.num_modes]
        hr_params = params[-self.num_modes:]
        return scaling_factor, energy_00, broadening, vib_energies, hr_params
    
    def _calculate_residuals(self, params_tofit, params_tohold, mask):
        params = self._get_full_param_array(params_tofit, params_tohold, mask)
        model, scaling_factor = self._calculate_model(params)
        return scaling_factor*self.y - model
    
    @staticmethod
    def _get_full_param_array(params_tofit, params_tohold, mask):
        params = np.zeros_like(mask, dtype=np.float64)
        f, h = 0, 0
        for i, b in enumerate(mask):
            if b:
                params[i] = params_tofit[f]
                f += 1
            else:
                params[i] = params_tohold[h]
                h += 1
        return params
            
    def perform_fit(self):
        """
        Do the fitting.

        Returns
        -------
        None.

        """
        self._create_tofit_lists()
        params = self._construct_parameter_array()
        mask = self._construct_parameters_tofit_mask()
        initial_guess = params[mask]
        params_tohold = params[np.invert(mask)]
        if self.algorithm == 'lm' or not self.enforce_non_negativity:
            fitted_params = least_squares(lambda params_tofit: self._calculate_residuals(params_tofit, params_tohold, mask), initial_guess, method=self.algorithm).x
        else:
            fitted_params = least_squares(lambda params_tofit: self._calculate_residuals(params_tofit, params_tohold, mask), initial_guess, method=self.algorithm, bounds=(0, np.inf)).x
        params = self._get_full_param_array(fitted_params, params_tohold, mask)
        self.scaling_factor, self.energy_00, self.broadening, self.vib_energies, self.hr_params = self._unpack_parameter_array(params)
        self.model = self.calculate_fc_progression(self.x, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        self.calculate_reorganisation_energy()
        
    def check_initial_guess(self):
        """
        Create a plot showing the data and model.
        
        Useful for checking how good the initial parameter guess is.

        Returns
        -------
        None.

        """
        self.model = self.calculate_fc_progression(self.x, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        self.plot_result(save=False)
        
    def plot_result(self, save=True):
        """
        Create a plot of the data and the fitted model.
        
        Parameters
        ----------
        save : bool, optional
            Whether to save the plot. If True, it will be saved to 
            **fc_fit.png**. The default is True.

        Returns
        -------
        None.

        """
        fig, ax = self.plot_modes(self.x, self.model, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        ax.plot(self.x, self.scaling_factor*self.y, 'r-', zorder=0)
        if save:
            fig.savefig('fc_fit.png', format='png', dpi=300, bbox_inches='tight')
    
    def print_result(self, tofile=False):
        """
        Print the parameter values to console or file.

        Parameters
        ----------
        tofile : bool, optional
            If True the values are printed to file. If False they are printed 
            to console. The default is False.

        Returns
        -------
        None.

        """
        if tofile:
            f = open('fc_parameters.txt', 'w')
            stream = f
        else:
            stream = None  # sys.stdout
        print('parameters (* - fitted)\n', file=stream)
        print('scaling factor: {0:.3f}{1}'.format(self.scaling_factor, '*' if self.fit_scaling_factor else ''), file=stream)
        print('broadening (eV): {0:.3f}{1}'.format(self.broadening, '*' if self.fit_broadening else ''), file=stream)
        print('0-0 energy (eV): {0:.3f}{1}'.format(self.energy_00, '*' if self.fit_energy_00 else ''), file=stream)
        for index, S in enumerate(self.hr_params):
            print('HR parameter {0}: {1:.3f}{2}'.format(index, S, '*' if index in self.fit_hr_params else ''), file=stream)
        for index, E_vib in enumerate(self.vib_energies):
            print('vib energy {0}: {1:.3f}{2}'.format(index, E_vib, '*' if index in self.fit_vibrational_energies else ''), file=stream)
        print('\nreorganisation energy (eV): {0:.3f}'.format(self.E_reorg), file=stream)
        if tofile:
            f.close()
        
    def calculate_reorganisation_energy(self):
        """
        Compute the reorganisation energy based on the fitted parameters.

        Returns
        -------
        None.

        """
        self.E_reorg = 0
        for i in range(self.num_modes):
            E_vib = self.vib_energies[i]
            S = self.hr_params[i]
            self.E_reorg += S*E_vib
            
    def save(self):
        """
        Save the fitted model.
        
        The total progression as well as the 0-0 and 0-1 contributions will be
        saved to **fc_fit.csv**.

        Returns
        -------
        None.

        """
        results = np.vstack((self.x, self.scaling_factor*self.y, self.model))
        for m_i in self._permutations:
            if sum(m_i) <= 1:
                peak = self._calculate_peak(m_i, self.x, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
                results = np.vstack((results, peak))
        np.savetxt('fc_fit.csv', np.transpose(results), delimiter=',', header='columns: energy (eV), data, fit, 0-0 and 0-1 peaks')
        self.print_result(tofile=True)
