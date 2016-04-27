# MHC-I pepdite binding prediction

## Data:
- http://tools.iedb.org/main/datasets/
- basically pepdite seq, hla allele names, and a target value
- see xplore/explore.ipynb for details

## Modelling:
- details in modelling folder 

### xgboost:
- http://xgboost.readthedocs.org/en/latest/
- gradient boosted ensemble of decision trees
	- http://xgboost.readthedocs.org/en/latest/model.html
	- http://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
- using dense variables

### neural networks:
- using the keras (theano) library
	- http://keras.io
- sparse representation, one-hot encoding
	- something with this blosum matrix, will be checked later
- 2 layer and deep model too
- trained in gpu

### other simple models:
- using the scikit-learn library
	- http://scikit-learn.org/stable/
- linear regression
- k nearest neighbor regression

