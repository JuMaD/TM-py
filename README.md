#Tunneling Models Py (TM-py)
TM-py aims to simplify fitting experimential J(V) or I(V) data to theoretical tunneling models, 
comparing tunneling models with one another and evaluating different fit methods. 
The python module lmfit is used to define tunneling models and perform fits. 
With this:
- The multi-parameter optimization can be varyied / performed in multiple steps (e.g. first a grid search, then )
- models can be combined to compound models (esp. interesting for ensemble molecular junctions with parameters varying over junction area)


In the long term there should be a GUI and data analysis techniques such as noise analysis and [outlier detection](https://lmfit.github.io/lmfit-py/examples/example_detect_outliers.html#sphx-glr-examples-example-detect-outliers-py) should be featured

A more comprehensive documentation will follow as this project grows.
#Authors
- [Julian Dlugosch]() - initial work