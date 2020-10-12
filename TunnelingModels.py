from lmfit import Model
# from lmfit import CompositeModel
from scipy import constants
from numpy import exp, sqrt, sinh
import lmfit
import pandas as pd
# import numpy as np

# global definition of physical constants
hbar = constants.hbar
h = constants.h
m_e = constants.electron_mass
e = constants.elementary_charge
pi = constants.pi


def simmons(v, area, alpha, phi, d, weight=1, beta=1, J=False, absolute=True):

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


def gruverman(v, area, phi1, phi2, d, massfactor=1, weight=1, J=False):
    if J:
        area = 1
    m = massfactor*m_e
    C = 32 * pi * e * m / (9 * h ** 3)
    phi_1 = phi1 * e
    phi_2 = phi2 * e
    alpha = 8 * pi * d * 10**(-9) * sqrt(2 * m) / (3 * h * phi_1 + e * v - phi_2)

    a = area * C * exp(alpha * ((phi_2 - e * v / 2)**(3/2) - (phi_1 + e * v / 2)**(3/2))) / \
        (alpha**2 * (sqrt(phi_2 - e * v / 2) - sqrt(phi_1 + e * v / 2)))**2
    b = sinh(3 * e * v / 4 * alpha((sqrt(phi_2 - e * v / 2) - sqrt(phi_1 + e * v / 2))))

    I = weight * a * b

    return I


class GruvermanModel(Model):
    _doc__ = "Gruverman" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(gruverman, *args, **kwargs)

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
    :return: volatage, currents
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


def combine_same_model(a, b):
    if a.name == b.name:
        newparamnames = []

        for name in a.param_names:
            newparamnames.append(name+'2')
        a._param_names = newparamnames
    return a + b


