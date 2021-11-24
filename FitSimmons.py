from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Parameter, fit_report

root = tk.Tk()
root.withdraw()
filename = filedialog.askopenfilename(initialdir = sys.path[0])

# get data from file
voltage, currents = data_from_csv(f'{filename}', sep='\t', min_voltage=0.01, max_voltage=0.51, current_start_column=1,
                                  voltage_column=0)

# create a parameter object for a simmons model that exists outside the fit function.
# Advantage: can be passed to functions

simmons_params = lmfit.Parameters()
simmons_params.add('area', value=2.25e-10, vary=False)
simmons_params.add('alpha', value=1, min=0, max=1, vary=False)
simmons_params.add('phi', value=2, min=1, max=5)
simmons_params.add('d', value=1, min=0.5, max=3)
simmons_params.add('weight', value=0.1, min=0.1, max=1, vary=False)
simmons_params.add('beta', value=1, min=0.1, max=1, vary=True)
simmons_params.add('absolute', value=1, vary=False)
simmons_params.add('J', value=0, vary=False)

for i in range(1,len(currents)):
    #i=2 #placeholder for loop
    current = currents[i]

    #instantiate Model
    SimmonsBarrier = SimmonsModel()
    simmons_brute, simmons_trials, simmons_fit = brute_then_local(SimmonsBarrier, current, voltage, 50, 'cobyla', simmons_params)

    plt.figure()
    plt.plot(voltage, np.abs(simmons_fit.best_fit), '-', label=f'Brute->local')
    plt.plot(voltage, current, 'ro', label='data')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.ylabel('current (A)')
    plt.xlabel('voltage (V)')
    plt.title('best fit')
    plt.ioff()
    plt.savefig(f"simmons-fitresult_{i}.png")
    plt.clf()

    df = pd.DataFrame(list(zip(currents[1], simmons_fit.best_fit)), columns =['Data', 'Fit'], index=voltage)
    df.to_csv(f"simmons-fitresult_{i}.csv", sep='\t')

    trials_df = fit_param_to_df(simmons_trials)
    trials_df.to_csv(f"simmons-best_trials_{i}.csv", sep="\t")