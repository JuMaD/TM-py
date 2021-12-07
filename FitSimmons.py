import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import os
import sys
from TunnelingModels import *
from lmfit import Parameter, fit_report
from tkinter import filedialog, messagebox
from tqdm import tqdm


while True:
    # open file dialog
    root = tk.Tk()
    root.withdraw()
    dirname = filedialog.askdirectory()

    # create directory for fits if it does not exist already
    if not os.path.exists(os.path.join(dirname, 'Fits')):
        os.makedirs(os.path.join(dirname, 'Fits'))

    #  iterate over all files in directory
    for file in tqdm(os.listdir(dirname)):

        if file.endswith("_density.csv"):
            #############################
            # Make Dataframes from file #
            #############################

            # open file and get data, skip the 0 value because there might be a duplicate
            filename = os.path.join(dirname, file)
            voltage, currents = data_from_csv(filename, sep='\t', min_voltage=0.01, max_voltage=0.51, current_start_column=2,
                                              voltage_column=0)

            # create a parameter object for a simmons model that exists outside the fit function.
            # Here you can adjust which parameters to fit and set bounds.
            # If no brute_step is set, a linear grid with 20 values is assumed for that parameter

            simmons_params = lmfit.Parameters()
            simmons_params.add('area', value=1, vary=False)
            simmons_params.add('alpha', value=1, min=0.5, max=1, vary=True, brute_step=0.1)
            simmons_params.add('phi', value=2, min=1, max=5, brute_step=0.5)
            simmons_params.add('d', value=1, min=1, max=5, brute_step=0.4)
            simmons_params.add('weight', value=0.1, min=0.1, max=1, vary=False)
            simmons_params.add('beta', value=1, min=0.1, max=1.5, vary=True, brute_step=0.1)
            simmons_params.add('absolute', value=1, vary=False)
            simmons_params.add('J', value=1, vary=False)

            for i in range(1,len(currents)):
                current = currents[i]

                #instantiate Model
                SimmonsBarrier = SimmonsModel()
                simmons_brute, simmons_trials, simmons_fit = brute_then_local(SimmonsBarrier, current, voltage, 50, 'leastsq', simmons_params)

                plt.figure()
                plt.plot(voltage, np.abs(simmons_fit.best_fit), '-', label=f'Brute->local')
                plt.plot(voltage, current, 'ro', label='data')
                plt.legend(loc='best')
                plt.yscale('log')
                plt.ylabel('current (A)')
                plt.xlabel('voltage (V)')
                plt.title('best fit')
                plt.ioff()
                fig_path = os.path.join(dirname, f"{file}-simmons-fitresult_{i}.png")
                plt.savefig(fig_path)
                plt.clf()

                df = pd.DataFrame(list(zip(currents[i], simmons_fit.best_fit)), columns =['Data', 'Fit'], index=voltage)

                trials_df = fit_param_to_df(simmons_trials)
                # save fit report to a file:
                param_save_path = os.path.join(dirname, f'{file}-fit_result.txt')
                with open(param_save_path, 'w') as fh:
                    fh.write(simmons_fit.fit_report())

                result_to_csv(df, filename, f"{file}-simmons-fitresult_{i}")
                result_to_csv(trials_df, filename, f"{file}-simmons-best_trials_{i}")

    again = messagebox.askyesno('Finished!', f'Finished wrangling files in {dirname}!\n Select another directory?')

    if again:
        continue
    else:
        break