"""lmfit supports
- multiple fit methods in serial (e.g. brute --> least squares)
- compound models e.g. a*ModelA+b*ModelB
- global minimization techniques such as APMGO and basinhopping"""

"""Thie file contains the definition of different tunneling models. 
The models are first discribed via function and then implemented as sub-classes of the Model class of lmfit."""

from lmfit import Model
from lmfit import CompositeModel
import lmfit

#from mpmath import *

from scipy import constants

import numpy as np
from numpy import exp, sqrt

#global definition of physical constants
hbar = constants.hbar
h = constants.h
m_e = constants.electron_mass
e = constants.elementary_charge
pi = constants.pi

#todo: implement Gruverman Model

def Simmons(v, area, alpha, phi, d, weight=1, beta=1):

    d = d * 10 ** (-9)

    J_0 = e / (2 * pi * h * (beta * d) ** 2)
    A = 4 * pi * beta * d * sqrt(2 * m_e) / h

    I = - weight * area * J_0 * (e*phi *exp(-A * alpha * sqrt(e*phi)) - (e*phi + e * v)*exp(-A * alpha * sqrt(e*phi) + e * v))


    return I

class SimmonsModel(Model):
        _doc__ = "Simmons Model" + lmfit.models.COMMON_DOC

        def __init__(self, *args, **kwargs):
            # pass in the defining equation
            super().__init__(Simmons, *args, **kwargs)
            self.set_param_hint('beta', min=0.01)  # Enforce that beta is positive.
            self.set_param_hint('alpha', min=0.01)  # Enforce that beta is positive.
            self.set_param_hint('phi', min=0.01)  # Enforce that beta is positive


def Simmons0K(v, area, alpha, phi, d, weight=1):
    """Simmons Function. Calculates the tunneling current of a symmetric barrier at 0K.
   :param v: Voltage. Dependent variable
   :param A:
   :param B:
   :param phi: height of barrier in eV
   :param alpha: ideality factor of effective electron mass. 0<= alpha <=1
   :param d: barrier width
   :param alpha: barrier height

   :rtype: float
   """
    A = 6.1657*10**10
    B = 1.02463
    return weight * area * A * ((phi - v / 2) / exp(d * B * sqrt(alpha * (phi - v / 2))) - (phi + v / 2) / exp(d * B * sqrt(alpha * (phi + v / 2)))) / (d ** 2)

class Simmons0KModel(Model):
    _doc__ = "SimmonsOkModel" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(Simmons0K, *args, **kwargs)
        self.set_param_hint('beta', min=0)  # Enforce that beta is positive.

def Gruverman(v, area, alpha, phi, d, weight=1, beta=1):

    return 1

# Utility Functions


def combineSameModel(A,B):
    if A.name == B.name:
        newparamnames = []

        for name in B.param_names:
            newparamnames.append(name+'2')
        B._param_names = newparamnames
    return A+B


