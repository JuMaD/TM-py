from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter

#good fit fuctions so far: basinhopping, cobyla, dual_annealing
v = np.linspace(-1, 1, 101)

np.random.seed(123)
noise = np.random.normal(size=len(v), scale=8e-10)

SimmonsBarrier = SimmonsModel()
true_params = SimmonsBarrier.make_params(area=2.5e-9, d=1.2, alpha=0.8, phi=3, weight=1)
true = SimmonsBarrier.eval(v=v, area=2.5e-9, alpha=1, phi=3, d=1.5, weight=1, beta=1)
data = SimmonsBarrier.eval(v=v, area=2.5e-9, alpha=1, phi=3, d=1.5, weight=1, beta=1)+noise

v2= np.linspace(-0.5,0,5,101)
print(data)
# add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
fit = SimmonsBarrier.fit(data,v=v,
                         area=Parameter('area', value=2.5e-9, vary=False),
                         alpha=Parameter('alpha', value=1, min=0, max=1),
                         phi=Parameter('phi', value=3.2, min=2.5, max=3.5),
                         d=Parameter('d', value=1, min=0.5, max=2),
                         weight=Parameter('weight', value=1, vary=False),
                         beta=Parameter('beta', value=1, min=0.01, max=1), method='basinhopping'
                         )

print(fit.fit_report())

plt.figure()
plt.plot(v, np.abs(data), 'ro')
plt.plot(v, np.abs(fit.init_fit), label='intial fit')
plt.plot(v, np.abs(fit.best_fit), label='best fit')

plt.legend(loc='best')
plt.yscale('log')

plt.ylabel('I(A)')
plt.xlabel('V(V)')
plt.title('simulated measurement')
plt.show()

