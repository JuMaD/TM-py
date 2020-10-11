from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter

#good fit fuctions so far: basinhopping, cobyla, dual_annealing, nelder, ampgo
v = np.linspace(-1, 1, 146)

np.random.seed(123)
noise = np.random.normal(size=len(v), scale=10e-10)

SimmonsBarrier = SimmonsModel()
#true_params = SimmonsBarrier.make_params(area=2.5e-9, d=1.2, alpha=0.8, phi=3, weight=1)
#true = SimmonsBarrier.eval(v=v, area=2.5e-9, alpha=1, phi=3, d=1.5, weight=1, beta=1)
data = SimmonsBarrier.eval(v=v, area=2.5e-9, alpha=1, phi=3, d=1.5, weight=1, beta=1)+noise


### get data from file
voltage, currents = DataFromCSV('testdata.csv', sep='\t', min_voltage=0.01, max_voltage=0.51, current_start_column=1, voltage_column=0)
current = currents[1]
fit = SimmonsBarrier.fit(current, v=voltage,
                         area=Parameter('area', value=2.25e-10, vary=False),
                         alpha=Parameter('alpha', value=1, min=0, max=1),
                         phi=Parameter('phi', value=2, min=1, max=8),
                         d=Parameter('d', value=1, min=0.5, max=3),
                         weight=Parameter('weight', value=1, min=0.01, max=1, vary=True),
                         beta=Parameter('beta', value=1, min=0.01, max=1, vary=True),
                         absolute=Parameter('absolute', value=True),
                         method='basinhopping',
                         )

print(fit.fit_report())

plt.figure()
#plt.plot(v, np.abs(data), 'ro')
plt.plot(voltage, np.abs(fit.init_fit), label='intial fit')
plt.plot(voltage, np.abs(fit.best_fit),'bo', label='best fit')
plt.plot(voltage, current, 'ro', label='data')
plt.legend(loc='best')
plt.yscale('log')
#plt.xlim(-0.5, 0.5)

plt.ylabel('I(A)')
plt.xlabel('V(V)')
plt.title('best fit')
plt.show()

