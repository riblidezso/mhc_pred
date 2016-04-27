# MHC-I peptide binding prediction

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


## Benchmark with state of the art

- based on this article: http://www.ncbi.nlm.nih.gov/pubmed/25017736
- Recreated the benchmark with new models, and taken the numbers form the supplementary

Method | CV | Blind test
--- | --- | ---
SMMPMBEC | 0.8989 | 0.8474
NetMHC | 0.8930 | 0.8833
NetMHCpan |0.9176 | 0.8830
new: xgboost | 0.9212 | 0.9042
new: 2layer net | **0.9259** | **0.9049**

- The new models have passed the state of the art

---
