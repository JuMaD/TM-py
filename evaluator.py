from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameter


param_names = ['area', 'alpha', 'phi', 'd', 'weight', 'beta', 'J', 'absolute']
data = []

for i in range(10):
    data.append([1e-9, 1, 1, 1+1/10*(i+1), 1, 1, 0, 1])

df = pd.DataFrame(data, columns = param_names)
v = np.linspace(-1.0, 1.0, 50)

Simmons = SimmonsModel()
eval_from_df(v, df, Simmons, ["phi","d"])