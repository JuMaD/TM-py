# Tunneling Models Py (TM-py)
TM-py aims to simplify fitting experimental J(V) or I(V) data to theoretical tunneling models, 
comparing tunneling models with one another and evaluating different fit methods. 
The python module lmfit is used to define tunneling models and perform fits. 
With this:
- Multiple option for optimization functions are available (see [here](https://lmfit.github.io/lmfit-py/fitting.html)) and can easily be combined (e.g. first a grid search, then Levenberg-Marquardt)
- models can be combined to compound models (esp. interesting for ensemble molecular junctions with parameters varying over junction area)

In the long term there should be a GUI and data analysis techniques such as noise analysis and [outlier detection](https://lmfit.github.io/lmfit-py/examples/example_detect_outliers.html#sphx-glr-examples-example-detect-outliers-py) should be featured
Models and utility functions are implemented in TunnelingModels.py

FitGruverman.py and FitSimmons.py showcase fitting experimental data to the Gruverman or the Simmons model, using a grid search and then local optimization on the 50 best results. The following output files are generated:
- [modelname]-fitresult_{i}.png: Best fit result of the i-th column in the provided data plotted together with the data
- [modelname]-fitresult_{i}.csv: csv with data and best fit result at the evaluated voltages
- [modelname]-best_trials_{i}csv: Parameter sets of the best trials (so the user can check whether 'more physical' combinations may yield reasonable fits as well)

FitBDR.py is similar to those, but with a much simpler interface.

evaluator.py showcases the usage of the evaluation function to simulate tunneling current from given parameters and a given model.

papereval.py was a practical implementation to calculate on/off ratios of memristive devices where the switching was caused by a shift in tunneling barrier height on one side.
# Utility functions
## brute then local
## fn-analysis
# Models
## Simmons
'Classical' Simmons model, that describes tunneling through a thin, trapezoidal, symmetric barrier (same barrier height on both sides)
DOI:10.1063/1.1702682

## Gruverman
'Classical' Gruverman model, that describes tunneling through a thin, trapezoidal, asymmetric barrier (barrier height different, depending on the side), specifically developped for ferroelectric tunneling junctions.
Derived from the BDR model and WKB approximation given small voltages (eV/2 < barrier heights) and "thick" barriers (d ((2m/hbar)^2*phi)^1/2 >> 1)
DOI:10.1021/nl901754t

## Brinkman, Dynes and Rowell (BDR)
Tunneling through a trapezoidal asymetric barrier
DOI: 10.1063/1.1659141

# Authors
- [Julian Dlugosch](http://github.com/jumad) - initial work