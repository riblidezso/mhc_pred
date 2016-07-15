# MHC-I peptide binding prediction

## Data:
- http://tools.iedb.org/main/datasets/
- basically pepdite seq, hla allele names, and a target IC50 value
- see xplore/explore.ipynb for details

## Modelling:
- details in modelling folders:
	- modelling_reg: regression models
	- modelling_class: classification models
	- modelling_benchmark: recreating a benchmark from an article to compare with state of the art models

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


## Benchmark with state of the art

- based on this article: http://www.ncbi.nlm.nih.gov/pubmed/25017736
- Recreated the benchmark with new models, and taken the numbers form the supplementary

---
