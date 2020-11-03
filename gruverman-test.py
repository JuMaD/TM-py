from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter


# get data from file
voltage, currents = data_from_csv('testdata.csv', sep='\t', min_voltage=0.01, max_voltage=1, current_start_column=1,
                                  voltage_column=0)
current = currents[1]

#instantiate Model
GruvermanBarrier = GruvermanModel()

# create a parameter object for a gruverman model that exists outside the fit function.
# Advantage: can be passed to functions and can be more easily modified by a GUI



gruverman_params = lmfit.Parameters()
gruverman_params.add('area', value=2.25e-10, vary=False)
gruverman_params.add('phi1', value=1, min=2, max=5)
gruverman_params.add('phi2', value=2, min=2, max=5)
gruverman_params.add('d', value=1, min=0.5, max=3)
gruverman_params.add('massfactor', value=1, min=0.1, max=1, vary=False)
gruverman_params.add('weight', value=1, min=0.1, max=1, vary=True)
gruverman_params.add('absolute', value=1, vary=False)
gruverman_params.add('J', value=0, vary=False)

gruverman_brute, gruverman_trials, gruverman_fit = brute_then_local(GruvermanBarrier, current, voltage, 50,'cobyla', gruverman_params)

print(fit_param_to_df(gruverman_trials))



plt.figure()

plt.plot(voltage, np.abs(gruverman_fit.best_fit), 'bo', label=f'Brute->local')
plt.plot(voltage, current, 'ro', label='data')
plt.legend(loc='best')
plt.yscale('log')

plt.ylabel('current (A)')
plt.xlabel('voltage (V)')
plt.title('best fit')
plt.show()