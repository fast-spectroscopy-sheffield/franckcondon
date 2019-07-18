from MultiModeFC import MultiModeFC
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt


class MultiModeFCFitting(MultiModeFC):
    
    def __init__(self):
        super(MultiModeFCFitting, self).__init__()
        self.y = None
        self.refractive_index = 1
        self.scaling_factor = 1
        
    def correct_for_phonon_dos(self):
        self.y /= (self.x*self.refractive_index)**3
        
    def calculate_model(self, params):
        self.unpack_params(params)
        self.calculate_fc_progression()
        
    def unpack_params(self, params):
        self.scaling_factor = params[0]
        self.broadening = params[1]
        self.energy_00 = params[2]
        self.hr_params = params[3:]
    
    def calculate_residuals(self, params):
        self.calculate_model(params)
        return self.scaling_factor*self.y - self.model
    
    def perform_fit(self):
        initial_guess = [self.scaling_factor, self.broadening, self.energy_00, *self.hr_params]
        fitted_params = least_squares(lambda params: self.calculate_residuals(params), initial_guess, bounds=(0, 1000)).x
        self.unpack_params(fitted_params)
        self.calculate_reorganisation_energy()
        
    def plot_result(self):
        plt.figure()
        plt.plot(self.x, self.scaling_factor*self.y, 'k-')
        plt.plot(self.x, self.model, 'r-')
        plt.xlabel('energy (eV)')
        plt.ylabel('PL (arb.)')
    
    def print_result(self):
        print('fitted parameters\n')
        print('scaling factor: {0:.3f}'.format(self.scaling_factor))
        print('broadening: {0:.3f} eV'.format(self.broadening))
        print('0-0 energy: {0:.3f} eV'.format(self.energy_00))
        for index, S in enumerate(self.hr_params):
            print('HR parameter {0}: {1:.3f}'.format(index, S))
        print('reorganisation energy: {0:.3f} eV'.format(self.E_reorg))
        
    def calculate_reorganisation_energy(self):
        self.E_reorg = 0
        for i in range(self.num_modes):
            E_vib = self.vib_energies[i]
            S = self.hr_params[i]
            self.E_reorg += S*E_vib
        
         
if __name__ == '__main__':
    # example of useage
    mmfcf = MultiModeFCFitting()
    
    # create some noisy data using the default values
    mmfcf.calculate_fc_progression()
    mmfcf.y = mmfcf.model + np.random.normal(0, 0.04, len(mmfcf.x))
    mmfcf.correct_for_phonon_dos()
    
    # specify the vibrational energies
    mmfcf.vib_energies = [0.054, 0.089, 0.165, 0.188]
    
    # set up the initial parameters
    mmfcf.hr_params = [0.7, 0.1, 0.5, 0.4]
    mmfcf.broadening = 0.025
    mmfcf.scaling_factor = 20
    mmfcf.energy_00 = 3.0
    mmfcf.update()
    
    # do the fitting
    mmfcf.perform_fit()
    
    # look at the results
    mmfcf.print_result()
    mmfcf.plot_result()