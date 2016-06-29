#!/usr/env python

"""
This script runs CV on MHC binding regression with multiple data splits.

Multiple splits allows to estimate the mean and variance of the performance.
Number of folds are hard coded! (To force standard results)

Now the seeds are going from 0,1,2,3... for reproducibility.
"""

#load modules
import os,sys
from sklearn.metrics import roc_auc_score

#command line arguments
representations=sys.argv[1]
n_repeats=int(sys.argv[2])
work_dir=sys.argv[3] #'/data/data1/ribli/mhc/'

#load my functions
sys.path.append('../')
from utils_xgb import load_all_data,my_xgb_cv_predict

#go to working dir
os.chdir(work_dir)

#load data
x,_,y_c=load_all_data(
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
         'objective': "binary:logistic",
         'eval_metric': 'auc',
         'nthread':22}

#loop
for i in xrange(n_repeats):
    #train
    y_pred=my_xgb_cv_predict(params,x,y_c,n_folds=5,seed=i)
    #evaluate
    auc=roc_auc_score(y_c,y_pred)
    
    print i,auc
    sys.stdout.flush()
