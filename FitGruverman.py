from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter, fit_report
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import sys

root = tk.Tk()
root.withdraw()
filename = filedialog.askopenfilename(initialdir = sys.path[0])

# get data from file
voltage, currents = data_from_csv(f'{filename}', sep='\t', min_voltage=0.01, max_voltage=0.4, current_start_column=1,
                                  voltage_column=0)

# create a parameter object for a gruverman model that exists outside the fit function.
# Advantage: can be passed to functions and can be more easily modified by a GUI
gruverman_params = lmfit.Parameters()
gruverman_params.add('area', value=2.25e-10, vary=False)
gruverman_params.add('phi1', value=1, min=2, max=5, vary=True, brute_step=0.5)
gruverman_params.add('phi2', value=2, min=2, max=5, vary=True, brute_step=0.5)
gruverman_params.add('d', value=1, min=0.5, max=3, vary=True, brute_step=0.2)
gruverman_params.add('massfactor', value=1, min=0.1, max=20, vary=True, brute_step=1)
gruverman_params.add('weight', value=1, min=0.1, max=1, vary=True, brute_step=0.2)
gruverman_params.add('absolute', value=1, vary=False)
gruverman_params.add('J', value=1, vary=False)

for i in range(1,len(currents)):
    #i=2 #placeholder for loop
    current = currents[i]

    #instantiate Model
    GruvermanBarrier = GruvermanModel()
    gruverman_brute, gruverman_trials, gruverman_fit = brute_then_local(GruvermanBarrier, current, voltage, 50,'cobyla', gruverman_params)

    plt.figure()
    plt.plot(voltage, np.abs(gruverman_fit.best_fit), '-', label=f'Brute->local')
    plt.plot(voltage, current, 'ro', label='data')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.ylabel('current (A)')
    plt.xlabel('voltage (V)')
    plt.title('best fit')
    plt.ioff()
    plt.savefig(f"gruverman-fitresult_{i}.png")
    plt.clf()

    df = pd.DataFrame(list(zip(currents[1], gruverman_fit.best_fit)), columns =['Data', 'Fit'], index=voltage)
    df.to_csv(f"gruverman-fitresult_{i}.csv", sep='\t')

    trials_df = fit_param_to_df(gruverman_trials)
    trials_df.to_csv(f"gruverman-best_trials_{i}.csv", sep="\t")

