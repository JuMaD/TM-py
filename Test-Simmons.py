from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter


# get data from file
voltage, currents = data_from_csv('testdata.csv', sep='\t', min_voltage=0.01, max_voltage=0.51, current_start_column=1,
                                  voltage_column=0)
current = currents[1]

#instantiate Model
SimmonsBarrier = SimmonsModel()

# create a parameter object for a simmons model that exists outside the fit function.
# Advantage: can be passed to functions and can be more easily modified by a GUI

simmons_params = lmfit.Parameters()
simmons_params.add('area', value=2.25e-10, vary=False)
simmons_params.add('alpha', value=1, min=0, max=1, vary=False)
simmons_params.add('phi', value=2, min=1, max=8)
simmons_params.add('d', value=1, min=0.5, max=3)
simmons_params.add('weight', value=1, min=0.1, max=1, vary=False)
simmons_params.add('beta', value=1, min=0.1, max=1, vary=True)
simmons_params.add('absolute', value=1, vary=False)
simmons_params.add('J', value=0, vary=False)

simmons_fit = SimmonsBarrier.fit(current, v=voltage, params=simmons_params, method='cobyla')

plt.figure()

plt.plot(voltage, np.abs(simmons_fit.best_fit), 'bo', label='Local')
plt.plot(voltage, current, 'ro', label='data')
plt.legend(loc='best')
plt.yscale('log')

plt.ylabel('current (A)')
plt.xlabel('voltage (V)')
plt.title('best fit')
plt.show()