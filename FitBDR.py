from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter


# get data from file
voltage, currents = data_from_csv('testdata.csv', sep='\t', min_voltage=0.01, max_voltage=1, current_start_column=1,
                                  voltage_column=0)
current = currents[1]

#instantiate Model
BDRBarrier = BDRModel()

# create a parameter object for a BDR model that exists outside the fit function.
# Advantage: can be passed to functions and can be more easily modified by a GUI


BDR_params = lmfit.Parameters()
BDR_params.add('area', value=2.25e-10, vary=False)
BDR_params.add('phi_avg', value=1, min=2, max=5)
BDR_params.add('phi_interfacial', value=2, min=2, max=5)
BDR_params.add('d', value=1, min=0.5, max=3)
BDR_params.add('massfactor', value=1, min=0.1, max=1, vary=False)
BDR_params.add('weight', value=1, min=0.1, max=1, vary=True)
BDR_params.add('absolute', value=1, vary=False)
BDR_params.add('J', value=0, vary=False)

BDR_brute, BDR_trials, BDR_fit = brute_then_local(BDRBarrier, current, voltage, 50, 'cobyla', BDR_params)

print(BDR_fit.best_fit)
plt.figure()

plt.plot(voltage, np.abs(BDR_fit.best_fit), 'bo', label=f'Brute->local')
plt.plot(voltage, current, 'ro', label='data')
plt.legend(loc='best')
plt.yscale('log')

plt.ylabel('current (A)')
plt.xlabel('voltage (V)')
plt.title('best fit')
plt.show()