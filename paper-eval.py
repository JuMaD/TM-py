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


syn_electron_barrier = -(syn_lumo-syn_fermi)
anti_electron_barrier = -(anti_lumo-syn_fermi)

syn_hole_barrier = -(syn_fermi - syn_homo)
anti_hole_barrier = -(anti_fermi - anti_homo)

#constants
alox_CB = 2
alox_VB = 6.3
al_wf = 4.3
massfactor = 1



#################################
# Calculation of poisson deltas #
#################################

#constants
epsilon_SAM = 3
d_SAM = 2.82

epsilon_Alox = 9.34 #check with marc's reference
d_Alox = 5 # assumption - why?
V_max = 3 #maximum sweep voltage for the "end state" of the SAM

print("=== Input ===")
print(f'epsilon_SAM={epsilon_SAM}')
print(f'd_SAM={d_SAM}')
print(f'epsilon_Alox={epsilon_Alox}')
print(f'd_Alox={d_Alox}')
print(f'V_max={V_max}')
print(f'alox_CB={alox_CB}')
print(f'alox_VB={alox_VB}')


print("=== Barriers ===")
print(f'syn electrons: {syn_electron_barrier}')
print(f'anti electrons: {anti_electron_barrier}')
print(f'syn holes: {syn_hole_barrier}')
print(f'anti holes: {anti_hole_barrier}')

# Calculation of field applied to SAM wit given V_max
E_SAM = epsilon_Alox / (epsilon_Alox * d_SAM + epsilon_SAM * d_Alox) * V_max
print("=== Field SAM ===")
print(f"E_SAM={E_SAM}")
# Poisson delta calculation from linear interpolation of MD Values
debeye_factor = 10**(-19) / (constants.c*100)
mu1 = 0.07 * E_SAM * debeye_factor
mu2 = (-0.05526 - 0.24029 * E_SAM) * debeye_factor
print("=== mu (D) ===")
print(f"mu1 = {0.07 * E_SAM}")
print(f"mu2 = {(-0.05526 - 0.24029 * E_SAM)}")
surface_density = 3.6 * 10**18
poisson_delta1 = surface_density * mu1 / (epsilon_SAM * constants.epsilon_0)
poisson_delta2 = surface_density * mu2 / (epsilon_SAM * constants.epsilon_0)
#poisson_delta1 = 0.0389
#poisson_delta2 = -0.0895
print("=== Poisson Deltas ===")
print(f"poisson_delta1={poisson_delta1}")
print(f"poisson_delta2={poisson_delta2}")



data = []

#holes
phi_1 =  alox_VB - al_wf
phi_2_syn = syn_hole_barrier - poisson_delta2
phi_2_anti = anti_hole_barrier - poisson_delta1

#electrons
#phi_1 =  al_wf-alox_CB
#phi_2_syn = syn_electron_barrier - poisson_delta2
#phi_2_anti = anti_electron_barrier - poisson_delta1

data.append([1, phi_1, phi_2_syn, d_SAM + d_Alox, massfactor, 1, 0, 1])
data.append([1, phi_1, phi_2_anti, d_SAM + d_Alox, massfactor, 1, 0, 1])
df = pd.DataFrame(data, columns = param_names)
v = np.linspace(-max_voltage, max_voltage, 200)
Gruverman = GruvermanModel()
gruverman_d= eval_from_df(v, df, Gruverman, ["phi1","phi2"], semilogy=True, plot=False)
print(gruverman_d)
ratio = gruverman_d[1]/gruverman_d[0]
results = {"Voltage (V)":v, "Ratio":ratio}
df = pd.DataFrame(results)
print("=== Ratios ===")
print(df)
#plt.plot(v, ratio)
#plt.semilogy()
#plt.show()
#plt.close()







# --------------- dump
#per molecule calculation -- peer
#poisson_delta1 = 0.008
#poisson_delta2 = -0.276

#10V calculation --> Falk
#poisson_delta1 = 0.273
#poisson_delta2 = -0.934

# ~3% current difference
#poisson_delta1 = 0.009/2
#poisson_delta2 = 0.252/2