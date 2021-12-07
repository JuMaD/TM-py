from TunnelingModels import *
import matplotlib.pyplot as plt
import numpy as np


# todo: iterate over a list of list of dicts --> multiple combined models
# todo: think about the difference between evaluating and fitting here :) --> turn dict into Parameters objects??

model_name = "simmons"
a = 0.2
dict1 = {"area":1, "alpha":1, "phi":1, "d":1, "weight":a, "beta":1, "J":0, "absolute":1}
dict2 = {"area":1, "alpha":1, "phi":1.2, "d":1, "weight":1-a, "beta":1, "J":0, "absolute":1}


list_of_param_dicts = [dict1, dict2]
no_fcns = len(list_of_param_dicts)
list_of_models = []
list_of_param_names = []
list_of_param_values = []

for i in range(1, no_fcns+1):
    model_i = Model(eval(model_name), prefix=f'f{i}_')
    # print(model_i.name)
    list_of_models.append(model_i)
    for param in list_of_param_dicts[i-1]:
        list_of_param_names.append(f'f{i}_{param}')
        list_of_param_values.append(list_of_param_dicts[i-1][param])

combined = None
for model in list_of_models:
    if combined == None:
        combined = model
    else:
        combined = combined + model

# make several list_of_param_values here
data = [list_of_param_values]
df = pd.DataFrame(data, columns = list_of_param_names)
v = np.linspace(-1.0, 1.0, 50)

#parameterize what is used in legend
result = eval_from_df(v, df, combined, ["f1_phi","f2_phi"], semilogy=True)

print(result)
