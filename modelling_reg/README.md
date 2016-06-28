## IC50 value regression

The notebooks in this folder try to regress IC50 values.
Results are evaluated with cross validation.
I used all the available datapoints from both group of data. 
I used all data points even if they only have (<,> measurement values).

### Best results:
- XGBoost
- using one-hot encoding
- encoding HLA alleles, serotypes


### More thorough preformance evaluation

xgboost_sample_result.py evaluate a training on different random CV splits.
This might take long time, but because of the high variance of the performance its essential.
