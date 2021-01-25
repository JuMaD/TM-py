# Tunneling Models Py (TM-py)
TM-py aims to simplify fitting experimental J(V) or I(V) data to theoretical tunneling models, 
comparing tunneling models with one another and evaluating different fit methods. 
The python module lmfit is used to define tunneling models and perform fits. 
With this:
- Multiple option for optimization functions are available (see [here](https://lmfit.github.io/lmfit-py/fitting.html)) and can easily be combined (e.g. first a grid search, then Levenberg-Marquardt)
- models can be combined to compound models (esp. interesting for ensemble molecular junctions with parameters varying over junction area)


In the long term there should be a GUI and data analysis techniques such as noise analysis and [outlier detection](https://lmfit.github.io/lmfit-py/examples/example_detect_outliers.html#sphx-glr-examples-example-detect-outliers-py) should be featured

A more comprehensive documentation will follow as this project grows.

# Utility functions
## brute then local
## fn-analysis
# Models
## Simmons
Description of Simmons Model
DOI:

## Gruverman
Description of Gruverman Model
DOI:

## Brinkman, Dynes and Rowell (BDR)
Description of the BDR Model
DOI:

# Authors
- [Julian Dlugosch](http://github.com/jumad) - initial work