from MultiModeFC import MultiModeFC
from scipy.optimize import least_squares
import numpy as np


class MultiModeFCFitting(MultiModeFC):
    
    def __init__(self):
        super(MultiModeFCFitting, self).__init__()
        self.refractive_index = 1
        self.fit_vibrational_energies = 'None'
        self.fit_hr_params = 'all'
        self.fit_broadening = True
        self.fit_energy_00 = True
        self.fit_scaling_factor = True
        
    def correct_for_phonon_dos(self):
        self.y /= (self.x*self.refractive_index)**3
        
    def input_data(self, x, y):
        self.x = x
        self.y = y
        
    def input_parameters(self, vib_energies, hr_params, energy_00, broadening, scaling_factor):
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
    
    def calculate_residuals(self, params_tofit, params_tohold, mask):
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
        self._create_tofit_lists()
        params = self._construct_parameter_array()
        mask = self._construct_parameters_tofit_mask()
        initial_guess = params[mask]
        params_tohold = params[np.invert(mask)]
        fitted_params = least_squares(lambda params_tofit: self.calculate_residuals(params_tofit, params_tohold, mask), initial_guess, bounds=(0, np.inf)).x
        params = self._get_full_param_array(fitted_params, params_tohold, mask)
        self.scaling_factor, self.energy_00, self.broadening, self.vib_energies, self.hr_params = self._unpack_parameter_array(params)
        self.model = self.calculate_fc_progression(self.x, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        self.calculate_reorganisation_energy()
        
    def check_initial_guess(self):
        self.model = self.calculate_fc_progression(self.x, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        self.plot_result()
        
    def plot_result(self):
        fig, ax = self.plot_modes(self.x, self.model, self.vib_energies, self.hr_params, self.energy_00, self.broadening)
        ax.plot(self.x, self.scaling_factor*self.y, 'r-', zorder=0)
    
    def print_result(self):
        print('parameters (* - fitted)\n')
        print('scaling factor: {0:.3f}{1}'.format(self.scaling_factor, '*' if self.fit_scaling_factor else ''))
        print('broadening (eV): {0:.3f}{1}'.format(self.broadening, '*' if self.fit_broadening else ''))
        print('0-0 energy (eV): {0:.3f}{1}'.format(self.energy_00, '*' if self.fit_energy_00 else ''))
        for index, S in enumerate(self.hr_params):
            print('HR parameter {0}: {1:.3f}{2}'.format(index, S, '*' if index in self.fit_hr_params else ''))
        for index, E_vib in enumerate(self.vib_energies):
            print('vib energy {0}: {1:.3f}{2}'.format(index, E_vib, '*' if index in self.fit_vibrational_energies else ''))
        print('\nreorganisation energy (eV): {0:.3f}'.format(self.E_reorg))
        
    def calculate_reorganisation_energy(self):
        self.E_reorg = 0
        for i in range(self.num_modes):
            E_vib = self.vib_energies[i]
            S = self.hr_params[i]
            self.E_reorg += S*E_vib
        
         
if __name__ == '__main__':
    # setup parameters
    hr_params = [0.67, 0.1, 0.52, 0.46]
    vib_energies = [0.054, 0.089, 0.165, 0.188]
    energy_00 = 3.050
    broadening = 0.027
    x = np.linspace(2.5, 3.1, 1000)
    
    # do the calculation for the 4 modes
    mmfc = MultiModeFC()
    mmfc.initialise(4)  # there are 4 modes
    model = mmfc.calculate_fc_progression(x, vib_energies, hr_params, energy_00, broadening)
    
    # add some noise to the model to use as test data
    y = model + np.random.normal(0, 0.03, len(x))
    
    # example of useage
    mmfcf = MultiModeFCFitting()
    
    # enter the data and parameters
    scaling_factor = 20
    mmfcf.input_data(x, y)
    mmfcf.correct_for_phonon_dos()
    mmfcf.input_parameters(vib_energies, hr_params, energy_00, broadening, scaling_factor)
    mmfcf.initialise(4)  # there are four modes
    
    # check initial guess with a plot if desired
    mmfcf.check_initial_guess()
    
    # choose which parameters to fit
    mmfcf.fit_broadening = False  # don't fit the broadening
    mmfcf.fit_hr_params = 'all'  # fit all of the HR parameters
    mmfcf.fit_vibrational_energies = [0, 2]  # fit the first and third vibrational energies
    
    # do the fitting and time it
    import time
    start = time.time()
    mmfcf.perform_fit()
    end = time.time()
    print('\ncomputation time: {0:.4f} seconds\n'.format(end-start))
    
    # look at the results
    mmfcf.print_result()
    mmfcf.plot_result()