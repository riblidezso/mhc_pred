import time

import numpy as np
import pandas as pd

from sklearn import preprocessing

from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split

from keras.callbacks import ModelCheckpoint,EarlyStopping

import xgboost as xgb

from sklearn import metrics

def load_data(hla_representation='simple',species_representation='simple',
              seq_representation='simple'):
    start=time.time()
    
    #load train data
    print 'Reading from file...'
    dataf='benchmark_mhci_reliability/binding/bd2013.1/bdata.20130222.mhci.public.1.txt'
    data=pd.read_csv(dataf,sep='\t')
     
    print 'Creating representation...'
    #encode hla
    x_hla=encode_hla(data.mhc.values,hla_representation)
    
    #encode species
    x_species=encode_hla(data.species.values,species_representation)

    #encode amino acids
    x_seq=encode_seq(data.sequence.values,seq_representation)
    
    #stack columns together
    x=np.column_stack([x_species,x_hla,x_seq,data.peptide_length.values])

    #predict log10 value
    y=np.log10(data.meas.values)

    #permute arrays as they are ordered!!!
    perm=np.random.permutation(len(x))
    x,y=x[perm],y[perm]
    
    #classify y
    y[y<np.log10(500)]=0
    y[y>=np.log10(500)]=1
    
    print 'Using ',len(x),'data points'
    print 'Done, It took:',time.time()-start,'s'
    return x,y

def encode_hla(x_in,representation):
    """Encode string HLA type to numbers."""
    if representation=='simple':
        le = preprocessing.LabelEncoder()
        x_out=le.fit_transform( x_in )
    elif representation=='one-hot':
        lb = preprocessing.LabelBinarizer()
        x_out=lb.fit_transform( x_in )
    return x_out

def encode_species(x_in,representation):
    """Encode string HLA type to numbers."""
    if representation=='simple':
        le = preprocessing.LabelEncoder()
        x_out=le.fit_transform( x_in )
    elif representation=='one-hot':
        lb = preprocessing.LabelBinarizer()
        x_out=lb.fit_transform( x_in )
    return x_out

def encode_seq(x_in,representation):
    """Encode string amino acid sequences to numbers."""
    #make peptids 'equal' length
    maxlen=np.max(map(len,x_in))
    x_temp=np.array([list(x.zfill(maxlen)) for x in x_in])
    
    if representation=='simple':
        le = preprocessing.LabelEncoder()
        le.fit(x_temp.flatten())
        x_out=np.column_stack([le.transform(x_temp[:,i]) for i in range(maxlen)])
    elif representation=='one-hot':
        lb = preprocessing.LabelBinarizer()
        lb.fit(x_temp.flatten())
        x_out=np.column_stack([lb.transform(x_temp[:,i]) for i in range(maxlen)])
    elif representation=='blosum':
        #read blosum matrix
        blosum62 = pd.read_csv('BLOSUM62.txt',header=None,skiprows=7,delim_whitespace=True,index_col=0)
        blosum_dict=blosum62.transpose().to_dict(orient='list')
        blosum_dict['0']=blosum_dict['*']
        #transform
        x_out=np.column_stack([[blosum_dict[x_temp[i,j]] for i in xrange(len(x_in)) ] for j in range(maxlen)])

    return x_out



"""Cross validation functions for keras.

People usually do not cross validate when they have to train for days, but
"""

def my_keras_fit_predict(get_model,X_train,y_train,X_test,
                      validation_split=0.1,patience=1,nb_epoch=100,**kwargs):
    """Fit model on train test set with early stopping."""
    #get model
    model=None
    model=get_model(X_train.shape[1])
    
    #callbacks
    best_model=ModelCheckpoint('best_model',save_best_only=True,verbose=1)
    early_stop=EarlyStopping(patience=patience,verbose=1)

    #train it
    callb_hist=model.fit(X_train,y_train,nb_epoch = nb_epoch,
                            validation_split=validation_split,
                            callbacks=[best_model,early_stop],**kwargs)
    #predict
    model.load_weights('best_model')
    y_pred_test=model.predict(X_test).ravel()
    
    return y_pred_test


def my_keras_cv_predict(get_model,x,y,n_folds=3,**kwargs):
    """Evaluate model with cross validation."""
    #res
    y_pred=np.zeros(len(y))
    #folds
    for train_index,test_index in KFold(len(x),n_folds=n_folds):
        #data split
        x_train,y_train,x_test=x[train_index],y[train_index],x[test_index]
        #fit predict
        y_pred[test_index]=my_keras_fit_predict(get_model,x_train,y_train,x_test,**kwargs)
    return y_pred



"""Cross valdiation with xgb too."""

def my_xgb_fit_predict(params,X_train,y_train,X_test,
                       num_boost_round=5000,verbose_eval=500,
                       early_stopping_rounds=200,
                       validation_size=0.1,**kwargs):
    """Fit model on train test set with early stopping."""
    #validation data for early stopping
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=validation_size)
    
    #convert to data format for xgb
    dtrain = xgb.DMatrix( X_train, label=y_train)
    dvalid = xgb.DMatrix( X_valid, label=y_valid)
    dtest = xgb.DMatrix( X_test )
    
    #printed evals
    evallist  = [(dtrain,'train'),(dvalid,'eval')]

    #lets train
    bst = xgb.train(params,dtrain,evals=evallist,
                    num_boost_round=num_boost_round,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval)
    
    y_pred_test=bst.predict(dtest)
    return y_pred_test


def my_xgb_cv_predict(params,x,y,n_folds=3,**kwargs):
    """Evaluate model with cross validation."""
    #res
    y_pred=np.zeros(len(y))
    #folds
    for train_index,test_index in KFold(len(x),n_folds=n_folds):
        #data split
        x_train,y_train,x_test=x[train_index],y[train_index],x[test_index]
        #fit predict
        y_pred[test_index]=my_xgb_fit_predict(params,x_train,y_train,x_test,**kwargs)
    return y_pred


import matplotlib.pyplot as plt
import seaborn as sns

"""evaluation."""
def plot_roc(y,probs):
    """Plot ROC curve and print auc score."""
    fpr, tpr, thresholds = metrics.roc_curve(y,probs)
    auc=metrics.roc_auc_score(y,probs)
    
    plt.figure(figsize=(6,6))
    plt.plot(fpr,tpr,lw=2)
    plt.plot([0,1],[0,1],lw=2)
    plt.xlim(-0.01,1.01)
    plt.ylim(-0.01,1.01)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    print 'AUC:',auc