from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameter


param_names = ['area', 'phi1', 'phi2', 'd',  'massfactor', 'weight', 'J', 'absolute']

#Simulation Window
max_voltage = 1

#Energy Levels
#syn_homo = 5.73399
syn_homo = 5.73
syn_lumo = 1.25700
#syn_fermi = 4.02270
syn_fermi = 4.25 # Pb & Al WF

#anti_homo = 5.73099
anti_homo = 5.73
anti_lumo = 1.25800
#anti_fermi = 4.0195
anti_fermi = 4.25 # Pb & Al WF

syn_electron_barrier = -(syn_lumo-syn_fermi)
anti_electron_barrier = -(anti_lumo-syn_fermi)

syn_hole_barrier = -(syn_fermi - syn_homo)
anti_hole_barrier = -(anti_fermi - anti_homo)

#constants
alox_CB = 2
alox_BG = 6.5
alox_VB = 5.2
al_wf = 4.25


V_max = 3 #maximum sweep voltage for the "end state" of the SAM


massfactor = 1

epsilon_SAM = 3
d_SAM = 2.81#2.81

epsilon_Alox = 9
d_Alox = 3

surface_density = 3.1 * 10**18


#################################
# Calculation of poisson deltas #
#################################



print("=== Input ===")
print(f'epsilon_SAM=\t\t{epsilon_SAM}')
print(f'd_SAM=\t\t\t\t{d_SAM} nm')
print(f'epsilon_Alox=\t\t{epsilon_Alox}')
print(f'd_Alox=\t\t\t\t{d_Alox} nm')
print(f'surface density =\t{surface_density}  m^-2')
print(f'V_max=\t\t\t\t{V_max} V')

print("")
print(f'massfactor=\t\t{massfactor}')

print("")
print(f'al_wf=\t\t\t\t{al_wf} eV')
print(f'alox_VB=\t\t\t{alox_VB} eV')


#print("=== Barriers ===")
#print(f'syn electrons: {syn_electron_barrier}')
#print(f'anti electrons: {anti_electron_barrier}')
#print(f'syn holes: {syn_hole_barrier}')
#print(f'anti holes: {anti_hole_barrier}')

# Calculation of field applied to SAM wit given V_max
#E_SAM = epsilon_Alox / (epsilon_Alox * d_SAM + epsilon_SAM * d_Alox) * V_max
E_SAM = V_max  * ( ( ( epsilon_Alox ) / (epsilon_SAM*d_Alox + epsilon_Alox*d_SAM) ) )
#E_SAM = 2.5
print("")
print("=== Field across SAM ===")
print(f"E_SAM = \t\t\t{E_SAM} V/nm")
# Poisson delta calculation from linear interpolation of MD Values
debeye_factor = 3.33564*10**(-30)
mu1 = 0.07 * E_SAM  * debeye_factor
mu2 = (-0.05526 - 0.24029 * E_SAM) * debeye_factor
print("")
print("=== mu (Cm) ===")
print(f"mu1(V_max) = \t\t{mu1} Cm")
print(f"mu2(V_max) = \t\t{mu2} Cm")

helmholtz_delta1 = surface_density * mu1 / (epsilon_SAM * constants.epsilon_0)
helmholtz_delta2 = surface_density * mu2 / (epsilon_SAM * constants.epsilon_0)
#poisson_delta1 = 0.0389
#poisson_delta2 = -0.0895
print("=== Helmholtz Deltas ===")
print(f"helmholtz_delta1 =\t{helmholtz_delta1} eV")
print(f"helmholtz_delta2 =\t{helmholtz_delta2} eV")


data = []

#Hole Barriers
phi_1 =  alox_VB - al_wf
phi_2_syn = syn_hole_barrier - helmholtz_delta2
phi_2_anti = anti_hole_barrier - helmholtz_delta1

print("")
print("=== Barriers (J) ===")
print(f"phi_1=\t\t\t\t{phi_1*constants.elementary_charge} J")
print(f"phi_2_syn=\t\t\t{phi_2_syn*constants.elementary_charge} J")
print(f"phi_2_anti=\t\t\t{phi_2_anti*constants.elementary_charge} J")


print("=== Barriers (eV) ===")
print(f"phi_1=\t\t\t\t{phi_1} eV")
print(f"phi_2_syn=\t\t\t{phi_2_syn} eV")
print(f"phi_2_anti=\t\t\t{phi_2_anti} eV")


data.append([1, phi_1, phi_2_syn, d_SAM + d_Alox, massfactor, 1, 0, 1])
data.append([1, phi_1, phi_2_anti, d_SAM + d_Alox, massfactor, 1, 0, 1])
df = pd.DataFrame(data, columns = param_names)
v = np.linspace(-max_voltage, max_voltage, max_voltage*200+1)
#print(f'voltage check {v[0]}, {v[200]}')
Gruverman = GruvermanModel()

print("")
print("=== alpha ===")
gruverman_d= eval_from_df(v, df, Gruverman, ["phi1","phi2"], semilogy=True, plot=False)
print("")
print("=== current densities (A/m^2 ?) ===")
print("\t\t-1V\t\t\t\t\t\t1V")
print(f'HRS:\t{gruverman_d[0][0]}\t {gruverman_d[0][1]}')
print(f'LRS:\t{gruverman_d[1][0]}\t {gruverman_d[1][1]}')
ratio = gruverman_d[1]/gruverman_d[0]
results = {"Voltage (V)":v, "Ratio":ratio}
df = pd.DataFrame(results)
print("=== Ratios ===")

print(df.loc[[0, max_voltage*200], :])
#print(df)




#plt.plot(v,gruverman_d[0])
#plt.plot(v,gruverman_d[1])
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