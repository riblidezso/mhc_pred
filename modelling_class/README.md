## Binding classification

The notebooks in this folder try to guess binding status (IC50<500 means binding).
Results are evaluated with cross validation.
I used all the available datapoints from both group of data. 
I used all data points even if they only have (<,> measurement values).

### Best results:
- XGBoost
- using one-hot encoding
- encoding HLA alleles, serotypes
