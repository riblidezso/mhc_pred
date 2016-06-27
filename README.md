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

Method | CV | Blind test
--- | --- | ---
SMMPMBEC | 0.8989 | 0.8474
NetMHC | 0.8930 | 0.8833
NetMHCpan |0.9176 | 0.8830
new: xgboost dense | 0.9212 | 0.9042
new: 2layer net | 0.9259 | 0.9049
new: xgboost sparse | **0.9283** | **0.9057**

- The new models have passed the state of the art
	- Please note that i did not tune the models excessively to fit this particular dataset
		- 2layer net was not tuned at all, 512 is the first and only number of units i have tried
		- xgboost was only tried with 2-3 depth values, no extensive hyperparameter tuning was used
	- by tuning the models one could improve the scores on this dataset, but this would not be completely fair, because i think the other models have not tuned themself
		- altough self tuning with hyperparameter search could/should part of a best performing model 

---
