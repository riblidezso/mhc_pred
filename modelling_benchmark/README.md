## Benchmarking binding classification

My models are evaluated similarly to this article: http://www.ncbi.nlm.nih.gov/pubmed/25017736 .
The notebooks in this folder try to guess binding status (IC50<500 means binding).
Results are evaluated with cross validation.
I used all the available datapoints from both group of data. 
I used all data points even if they only have (<,> measurement values).

### Best results:
- stacking XGBoost and Neural network
- using one-hot encoding
- encoding HLA alleles, serotypes
