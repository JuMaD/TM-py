from lmfit import Model
# from lmfit import CompositeModel
from scipy import constants
from numpy import exp, sqrt, sinh
from matplotlib.colors import LogNorm
import lmfit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import copy
import re

# global definition of physical constants
hbar = constants.hbar
h = constants.h
m_e = constants.electron_mass
e = constants.elementary_charge
pi = constants.pi

# todo: Models: tyler expansion, multi-barrier model, tsu esaki, distribution over area, ..
# todo: add DOIs of references to the models

def simmons(v, area, alpha, phi, d, weight=1, beta=1, J=0, absolute=1):

    d = d * 10 ** (-9)

    if J:
        area = 0

    J_0 = e / (2 * pi * h * (beta * d) ** 2)
    A = 4 * pi * beta * d * sqrt(2 * m_e) / h

    I = - weight * area * J_0 * (e*phi * exp(-A * alpha * sqrt(e * phi)) - (e * phi + e * v) *
                                 exp(-A * alpha * sqrt(e * phi) + e * v))
    if absolute:
        I = abs(I)

    return I

class SimmonsModel(Model):
    _doc__ = "Simmons Model" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(simmons, *args, **kwargs)
        self.set_param_hint('beta', min=0.01)  # Enforce that beta is positive.
        self.set_param_hint('alpha', min=0.01)  # Enforce that beta is positive.
        self.set_param_hint('phi', min=0.01)  # Enforce that beta is positive


def gruverman(v, area, phi1, phi2, d, massfactor=1, weight=1, J=0, absolute=1):
    """

    :param v:
    :param area:
    :param phi1:
    :param phi2:
    :param phi_diff:
    :param d:
    :param massfactor:
    :param weight:
    :param J:
    :param absolute:
    :return:
    """

    if J:
        area = 1
    #todo: think about constraints and implement them in model
    m = massfactor*m_e
    C = - 32 * pi * e * m / (9 * h ** 3)
    phi_1 = phi1 * e
    phi_2 = phi2 * e
    alpha = 8 * pi * d * 10**(-9) * sqrt(2 * m) / (3 * h * (phi_1 + e * v - phi_2))

    arg_1 = phi_1 + (e*v)/2
    arg_2 = phi_2 - (e*v)/2
    a = area * C * exp(alpha * (arg_2**1.5-arg_1**1.5)) / (alpha**2 * (sqrt(arg_2) - sqrt(arg_1))**2)
    b = sinh(3*e*v / 4 * alpha * (sqrt(arg_2) - sqrt(arg_1)))


    I = weight * a * b

    if absolute:
        I = abs(I)

    return I

class GruvermanModel(Model):
    _doc__ = "Gruverman" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(gruverman, *args, **kwargs)


def bdr(v, area, phi_avg, phi_interfacial, d, J=0, absolute=1, massfactor=1):
    """
    Expanded BDR model from Miller 2009 [DOI: 10.1063/1.3122600]
    Model valid for biases less than one-third of the barrier height. both phi are in Volts
    :param v:
    :param area: area of the junction
    :param phi_avg: average barrier height
    :param phi_interfacial: interfacial barrier height difference
    :param d: thickness in nm
    :param J:
    :param absolute:
    :return:
    """
    d = d * 10**(-9)
    G_0 = (e / h)**2 * sqrt(2 * massfactor * m_e * e * phi_avg / (d)**2 ) * exp( -2*d / hbar * sqrt(2 * massfactor * m_e * e * phi_avg ))
    if J:
        area = 1
    # todo: check whetether gelta phi has to be under the sqrt
    I = area * G_0 * ( v + d * sqrt( 2 * massfactor * m_e / e) * phi_interfacial * v**2 / ( 24 * hbar * phi_avg**(3/2))) + ( d**2 * massfactor * m_e * e * v**3 ) / ( 12 * hbar**2 * phi_avg )

    if abs:
        I = abs(I)

    return I

class BDRModel(Model):
    _doc__ = "BDR" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(bdr, *args, **kwargs)


# Utility Functions

def data_from_csv(filename, sep, current_start_column, min_voltage, max_voltage, voltage_column=0, comments='#'):
    """
    This function loads current and voltage values from a csv File and returns the current as np array and
    the currents as array of np arrays. Lines that start with the "comments" character are ignored,
    the first line is assumed to be the column name and is also disregarded.
    :param max_voltage:  maximum voltage to evaluate
    :param filename: name and location of the file where the data is stored
    :param sep: csv seperator
    :param voltage_column: column number in which the voltage data is stored (first column: 0)
    :param current_start_column: first column in which current data is stored
    :param min_voltage: minimum voltage (absolute) to evaluate
    :param comments: character that signals a comment line in the csv file
    :return: voltage, currents
    """
    dataFromFile = pd.read_csv(filename, sep=sep, comment=comments)
    dataFromFile = dataFromFile.dropna()
    dataFromFile = dataFromFile.reset_index(drop=True)

    indicesToDrop = dataFromFile[(abs(dataFromFile.iloc[:, voltage_column].values) > max_voltage)].index
    indicesToDrop.append(dataFromFile[(abs(dataFromFile.iloc[:, voltage_column].values) < min_voltage)].index)
    new = dataFromFile[(abs(dataFromFile.iloc[:, voltage_column].values) < min_voltage)].index
    dataFromFile.drop(indicesToDrop, inplace=True)
    dataFromFile.drop(new, inplace=True)
    print(dataFromFile.iloc[:, voltage_column].values)

    voltage = dataFromFile.iloc[:, voltage_column].values
    currents = []
    [currents.append(dataFromFile.iloc[:, n].values) for n in range(current_start_column, len(dataFromFile.columns))]
    return voltage, currents

def plot_results_brute(result, best_vals=True, varlabels=None,
                       output=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Mostly copied from the "brute-fit"  example of the lmfit documentation
    https://lmfit.github.io/lmfit-py/examples/example_brute.html).

    :param result:      Contains the lmfit results from the `brute` method.
    :param best_vals:   Whether to show the best values from the grid search (default is True).
    :param arlabels:    If None (default), use `result.var_names` as axis labels, otherwise use the names
                        specified in `varlabels`.
    :param output:      Name of the output PDF file (default is 'None')
    """
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                elif i == npars-1:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 7, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 3, dtype='int')
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                elif j == npars-1:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output)

def combine_same_model(a, b):
    if a.name == b.name:
        newparamnames = []

        for name in a.param_names:
            newparamnames.append(name+'2')
        a._param_names = newparamnames
    return a + b


def brute_then_local(model, current, voltage, n_solutions, local_method, parameters):
    """
    Performs a brute grid search first and then performs local_method optimization
    on the n_solutions best solutions from it.

    Inspired by an example from the lmfit documentation (https://lmfit.github.io/lmfit-py/examples/example_brute.html)
    :param model:
    :param n_solutions:
    :param local_method:
    :param parameters:      parameters to start from
    :return:                result_brute, trials, best_result --> Returns the result of the first brute,
                            all local fits of the best trials of brute and the best overall result.
    """

    # todo: implement alternative measures to chisqr. See MinimizerResult: https://lmfit.github.io/lmfit-py/fitting.html

    result_brute = model.fit(current, v=voltage, params=parameters, method='brute', keep=n_solutions)
    best_result = copy.deepcopy(result_brute)

    trials = []

    for candidate in result_brute.candidates:
        trial = model.fit(current, v=voltage, params=candidate.params, method=local_method)
        trials.append(trial)
        if trial.chisqr < best_result.chisqr:
            best_result = trial


    return result_brute, trials, best_result


def fit_param_to_df(list_of_results):
    """
    Takes a list of fit results and creates a dataframe listing the values of all parameters
    and errors (chi-square) for all results

    :param list_of_results: A list of fit results from any fit.
    :param df_params:    Parameters to be added to the df. If list is empty, all parameters will be added
    :return:             Pandas dataframe
    """
    # todo: implement alternative measures to chisqr. See MinimizerResult: https://lmfit.github.io/lmfit-py/fitting.html

    dict_list = []
    for result in list_of_results:
        to_print = {}

        for param in result.params:
                to_print[result.params[param].name] = result.params[param].value
        to_print["chisqr"] = result.chisqr
        to_print["redchi"] =result.redchi
        to_print["aic"] = result.aic
        to_print["bic"] = result.bic
        dict_list.append(to_print)
    df = pd.DataFrame(dict_list)

    return df

def eval_from_df(v, df, model, label_params, semilogy):
    """
    Evaluates the model d with parameter sets that are given in df and plots the result
    :param v:
    :param df:
    :param model:
    :return:
    """


    names = model.param_names
    results = []
    for i in range(0,df.shape[0]):
        params = lmfit.Parameters()
        for name in names:
            params.add(name, value=df.at[i, name])
        result = model.eval(v=v, params=params)
        results.append(result)

    for i in range(len(results)):
        labels = {}
        for label in label_params:
            labels[label] = np.round(df.at[i,label],6)


        if semilogy:
            plt.semilogy(v, results[i], label=re.sub(':',' =',re.sub('[{}\']', '', str(labels))))
        else:
            plt.plot(v, results[i], label=re.sub(':',' =',re.sub('[{}\']', '', str(labels))))
    plt.legend()
    plt.show()
    return results


def calc_TVS(current, voltage, alpha=2):
    """
    Calculates the transition voltage spectroscopy spectrum from I-V data
    :param current:
    :param voltage:
    :param alpha:
    :return: spectrum
    """
    x = 1 / voltage
    ln = np.log(current / ( voltage ** alpha) )

    return x, ln

# todo: calc_NDC

