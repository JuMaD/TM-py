from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np

SimmonsBarrier = SimmonsModel()
SimmonsBarrier2 = SimmonsModel()
SimmonsBarrier3 = SimmonsModel()
SimmonsBarrier4 = SimmonsModel()

for w in np.arange(0.1, 0.9, 0.1):
    true_params = SimmonsBarrier.make_params(area=1, d=1.2, alpha=0.8, phi=1, weight=w)
    true_params2 = SimmonsBarrier2.make_params(area=1, d=1.2, alpha=0.8, phi=3, weight=1-w)

    true_params3 = SimmonsBarrier.make_params(area=1, d=1.2, alpha=0.8, phi=1, weight=1)
    true_params4 = SimmonsBarrier.make_params(area=1, d=1.2, alpha=0.8, phi=3, weight=1)

    v = np.linspace(-0.1, 0.1, 100)
    true = SimmonsBarrier.eval(params=true_params, v=v)
    true2 = SimmonsBarrier2.eval(params=true_params2, v=v)
    true3 = SimmonsBarrier3.eval(params=true_params3, v=v)
    true4 = SimmonsBarrier4.eval(params=true_params4, v=v)

combined = combine_same_model(SimmonsBarrier, SimmonsBarrier2)
combinedparams = true_params+true_params2
truecombined = combined.eval(params=combinedparams, v=v)

plt.figure()
plt.plot(v, np.absolute(true2), v, np.absolute(true3), v, np.absolute(truecombined))

plt.ylabel('I(A)')
plt.xlabel('V(V)')
plt.title('simulated measurement')
plt.show()
