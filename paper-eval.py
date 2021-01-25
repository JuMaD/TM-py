from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameter



param_names = ['area', 'phi1', 'phi2', 'd', 'weight', 'massfactor', 'J', 'absolute']

#Simulation Window
max_voltage = 1

#Energy Levels
syn_homo = 5.73399
syn_lumo = 1.25700
syn_fermi = 4.02270

anti_homo = 5.73099
anti_lumo = 1.25800
anti_fermi = 4.0195

#poisson_delta1 = 0.009/2
#poisson_delta2 = 0.252/2

massfactor = 1

syn_electron_barrier = -(syn_lumo-syn_fermi)
anti_electron_barrier = -(anti_lumo-syn_fermi)

syn_hole_barrier = -(syn_fermi - syn_homo)
anti_hole_barrier = -(anti_fermi - anti_homo)

alox_CB = 0.79
alox_VB = 6.76
al_wf = 4.3

print("=== Barriers ===")
print(f'syn electrons: {syn_electron_barrier}')
print(f'anti electrons: {anti_electron_barrier}')
print(f'syn holes: {syn_hole_barrier}')
print(f'anti holes: {anti_hole_barrier}')

range_delta1 = [0.0, 0.45, 0.05]
range_factor = [1, 10, 1]

model_id = "VB-plus-minus_5nm"

""" def loop_gruverman(range_delta1, range_factor, massfactor):

    for poisson_delta1 in np.arange(range_delta1[0],range_delta1[1], range_delta1[2]):
        ratios = {}

        for a in np.arange(range_factor[0], range_factor[1], range_factor[2]):
            data = []
            poisson_delta2 = poisson_delta1*a
            #area=2.25e-10
            data.append([1, alox_VB-al_wf, alox_VB-al_wf-poisson_delta1, 5, 1, massfactor, 1, 1])
            data.append([1, alox_VB-al_wf, alox_VB-al_wf+poisson_delta2, 5, 1, massfactor, 1, 1])

            df = pd.DataFrame(data, columns = param_names)
            v = np.linspace(-max_voltage, max_voltage, 200)
            Gruverman = GruvermanModel()
            gruverman_d= eval_from_df(v, df, Gruverman, ["phi1","phi2"], semilogy=True, plot=False)

            ratio = gruverman_d[1]/gruverman_d[0]
            ratios[f"{a}"] = ratio

        dict = pd.DataFrame.from_dict(ratios)

        plt.plot(v, dict)
        plt.legend(dict.keys())
        plt.title(f"ON/OFF ratios for poisson_delta1 = {poisson_delta1}")
        plt.semilogy()
        plt.savefig(f"graphs/{model_id}_delta1_{poisson_delta1}.png")
        plt.close() """

data = []

#per molecule calculation -- peer
#poisson_delta1 = 0.008
#poisson_delta2 = -0.276

#10V calculation --> Falk
#poisson_delta1 = 0.273
#poisson_delta2 = -0.934

#1V calculation, symmetric?
# This model assumes an "effective barrier height" and "effective barrierwidth" that combines alox +
poisson_delta1 = 0.0389
poisson_delta2 = -0.0895




data = []
data.append([1, alox_VB-al_wf, syn_hole_barrier-poisson_delta2*2, 7.8, 1, 1, 0, 1])
data.append([1, alox_VB-al_wf, anti_hole_barrier-poisson_delta1*2, 7.8, 1, 1, 0, 1])

df = pd.DataFrame(data, columns = param_names)
v = np.linspace(-max_voltage, max_voltage, 200)
Gruverman = GruvermanModel()
gruverman_d= eval_from_df(v, df, Gruverman, ["phi1","phi2"], semilogy=True, plot=False)

ratio = gruverman_d[1]/gruverman_d[0]
plt.plot(v, ratio)
plt.semilogy()
plt.show()
plt.close()

#loop_gruverman(range_delta1, range_factor, massfactor)

# ~3% current difference