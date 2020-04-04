from franckcondon import MultiModeFCFitting


import pandas as pd
# load data
data = pd.read_csv('example_data.csv', header=None, index_col=0, squeeze=True)
x = data.index.values
y = data.values

# setup parameters
hr_params = [1, 0.5]
vib_energies = [0.19, 0.067]
energy_00 = 3.77
broadening = 0.027


# example of useage
mmfcf = MultiModeFCFitting()

# enter the data and parameters
scaling_factor = 150
mmfcf.input_data(x, y)
mmfcf.correct_for_phonon_dos()
mmfcf.input_parameters(vib_energies, hr_params, energy_00, broadening, scaling_factor)
mmfcf.initialise(2)  # there are two modes

# check initial guess with a plot if desired
mmfcf.check_initial_guess()

# choose which parameters to fit
mmfcf.fit_broadening = True
mmfcf.fit_hr_params = 'all'
mmfcf.fit_vibrational_energies = 'all'

# do the fitting and time it
import time
start = time.time()
mmfcf.perform_fit()
end = time.time()
print('\ncomputation time: {0:.4f} seconds\n'.format(end-start))

# look at the results
mmfcf.print_result(tofile=False)
mmfcf.plot_result(save=False)  # saves the graph by default

# save the data
#mmfcf.save()  # the data, the fit, the 0-0 and 0-1 peaks and parameters
