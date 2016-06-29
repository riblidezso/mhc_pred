#!/usr/env python

"""
This script runs CV on MHC binding regression with multiple data splits.

Multiple splits allows to estimate the mean and variance of the performance.
Number of folds are hard coded! (To force standard results)

Now the seeds are going from 0,1,2,3... for reproducibility.
"""

#load modules
import os,sys
from scipy.stats import pearsonr

#command line arguments
representations=sys.argv[1]
n_repeats=sys.argv[2]
work_dir=sys.argv[3] #'/data/data1/ribli/mhc/'

#load my functions
sys.path.append('../')
from utils_xgb import load_all_data,my_xgb_cv_predict

#go to working dir
os.chdir(work_dir)


#load data
x,y,_=load_all_data(
    hla_representation=representations,
    species_representation=representations,
    seq_representation=representations)


#model params
params = {'max_depth':20,
         'eta':0.05,
         'min_child_weight':5,
         'colsample_bytree':1,
         'subsample':1,
         'silent':1,
         'objective': "reg:linear",
         'eval_metric': 'rmse',
         'nthread':4}

#loop
for i in xrange(n_repeats):
    #train
    y_pred=my_xgb_cv_predict(params,x,y,n_folds=5,seed=i)
    #evaluate
    r=pearsonr(y,y_pred)[0]
    
    print i,r
    sys.stdout.flush()