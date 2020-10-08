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
from numpy import exp, sqrt, sinh

#global definition of physical constants
hbar = constants.hbar
h = constants.h
m_e = constants.electron_mass
e = constants.elementary_charge
pi = constants.pi


def Simmons(v, area, alpha, phi, d, weight=1, beta=1, J=False):

    d = d * 10 ** (-9)

    if J:
        area = 0

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

def Gruverman(v, area, phi1, phi2, d, massfactor=1, weight=1, J=False):
    if J:
        area = 1
    m = massfactor*m_e
    C = 32 * pi * e * m / (9 * h ** 3)
    phi_1 = phi1 * e
    phi_2 = phi2 * e
    alpha = 8*pi*d*10**(-9)*sqrt(2*m) / (3*h*phi_1+e*v-phi_2)

    a = area * C * exp(alpha*((phi_2-e*v/2)**(3/2)-(phi_1+e*v/2)**(3/2))) / (alpha**2*(sqrt(phi_2-e*v/2)-sqrt(phi_1+ev/2)))**2
    b = sinh(3*e*v/4 * alpha((sqrt(phi_2-e*v/2)-sqrt(phi_1+ev/2))))

    I = a*b

    return I

class GruvermanModel(Model):
    _doc__ = "Gruverman" + lmfit.models.COMMON_DOC

    def __init__(self, *args, **kwargs):
        # pass in the defining equation
        super().__init__(Gruverman, *args, **kwargs)



# Utility Functions

def combineSameModel(A,B):
    if A.name == B.name:
        newparamnames = []

        for name in B.param_names:
            newparamnames.append(name+'2')
        B._param_names = newparamnames
    return A+B


