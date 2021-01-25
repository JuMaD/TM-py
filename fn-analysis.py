from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Parameter
from numpy import random


param_names = ['area', 'alpha', 'phi', 'd', 'weight', 'beta', 'J', 'absolute']
data = []

data.append([1e-9, 1, 1, 1.4, 1, 1, 0, 1])

df = pd.DataFrame(data, columns = param_names)
v = np.linspace(-1.0, 1.0, 50)

Simmons = SimmonsModel()
result = eval_from_df(v, df, Simmons, ["phi","d"], semilogy=False)

result_df = pd.DataFrame({"voltage": v, "current": result[0]})
noise = random.normal(scale=0.7215, size=v.size)
noisy = result[0] + noise
noisy_df = pd.DataFrame({"voltage": v, "current": noisy})
noisy_df.plot()

