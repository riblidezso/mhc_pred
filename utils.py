import os
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.utils import resample
from keras.callbacks import ModelCheckpoint,EarlyStopping
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns



"""Data looader functions"""

def load_all_data(hla_representation='simple',species_representation='simple',
                  seq_representation='simple'):
    """Load all data."""
    #load data with train test 
    x_train,y_train,y_train_c,x_test,y_test,y_test_c=load_data(
        hla_representation=hla_representation,
        species_representation=species_representation,
        seq_representation=seq_representation)

    #concatenate the sets
    all_x=np.concatenate([x_train,x_test],axis=0)
    all_y=np.concatenate([y_train,y_test],axis=0)
    all_y_c=np.concatenate([y_train_c,y_test_c],axis=0)

    #shuffle them
    rng=np.random.RandomState(42)
    perm=rng.permutation(len(all_y))
    x=all_x[perm]
    y=all_y[perm]
    y_c=all_y_c[perm]
    
    return all_x,all_y,all_y_c


def load_data(hla_representation='simple',species_representation='simple',
              seq_representation='simple',verbose=False):
    """Load train/blind test data as in the benchmark."""
    start=time.time()
    
    #load train data
    dataf='benchmark_mhci_reliability/binding/bd2009.1/bdata.2009.mhci.public.1.txt'
    t_data=pd.read_csv(dataf,sep='\t')
    
    #load test data
    dataf='benchmark_mhci_reliability/binding/blind.1/bdata.2013.mhci.public.blind.1.txt'
    b_data=pd.read_csv(dataf,sep='\t')
    
    #change hla encoding in test, its different from train
    from  re import sub
    b_data['mhc']=[sub(':','',sub('\*','-',hla_type))  for hla_type in b_data['mhc'].values]
    
    #concat them
    data=pd.concat([t_data,b_data])
     
    #encode hla
    x_train_hla=encode(data.mhc.values,t_data.mhc.values,hla_representation)
    x_test_hla=encode(data.mhc.values,b_data.mhc.values,hla_representation)
    
    #encode hla loci (A,B,no C?)
    x_train_hla_loci=encode_hla_loci(data.mhc.values,t_data.mhc.values,hla_representation)
    x_test_hla_loci=encode_hla_loci(data.mhc.values,b_data.mhc.values,hla_representation)
    
    #encode hla subtype
    x_train_hla_sero=encode_hla_sero(data.mhc.values,t_data.mhc.values,hla_representation)
    x_test_hla_sero=encode_hla_sero(data.mhc.values,b_data.mhc.values,hla_representation)
    
    #encode species
    x_train_species=encode(data.species.values,t_data.species.values,species_representation)
    x_test_species=encode(data.species.values,b_data.species.values,species_representation)

    #encode amino acids
    x_train_seq=encode_seq(data.sequence.values,t_data.sequence.values,seq_representation)
    x_test_seq=encode_seq(data.sequence.values,b_data.sequence.values,seq_representation)
    
    #stack columns together
    x_train=np.column_stack([x_train_species,x_train_hla,x_train_hla_loci,
                             x_train_hla_sero,x_train_seq,t_data.peptide_length.values])
    x_test=np.column_stack([x_test_species,x_test_hla,x_test_hla_loci,
                            x_test_hla_sero,x_test_seq,b_data.peptide_length.values])

    #predict log10 value
    y_train=np.log10(t_data.meas.values)
    y_test=np.log10(b_data.meas.values)

    #permute arrays as they are ordered!!!
    rng=np.random.RandomState(42)
    perm=rng.permutation(len(x_train))
    x_train,y_train=x_train[perm],y_train[perm]
    
    rng=np.random.RandomState(42)
    perm=rng.permutation(len(x_test))
    x_test,y_test=x_test[perm],y_test[perm]
    
    #classify y
    y_train_c=np.ones(len(y_train))
    y_train_c[y_train>=np.log10(500)]=0
    y_test_c=np.ones(len(y_test))
    y_test_c[y_test>=np.log10(500)]=0
    
    #report if asked for
    if verbose:
        print 'Using ',len(x_train),' training data points'
        print 'Using ',len(x_test),' testing data points'
        print 'Done, It took:',time.time()-start,'s'
        
    return x_train,y_train,y_train_c,x_test,y_test,y_test_c


def encode(x_all,x_in,representation):
    """Encode string to numbers."""
    if representation=='simple':
        le = preprocessing.LabelEncoder()
        le.fit(x_all)
        x_out=le.transform( x_in )
    elif representation=='one-hot':
        lb = preprocessing.LabelBinarizer()
        lb.fit(x_all)
        x_out=lb.transform( x_in )
    return x_out


def encode_hla_loci(x_all,x_in,representation):
    """Encode string HLA loci to numbers."""
    #get 1st subtype
    new_x_all=[x.split('-')[1][0] for x in x_all]
    new_x_in=[x.split('-')[1][0] for x in x_in]
    
    return encode(new_x_all,new_x_in,representation)


def get_serotype(hla_type):
    """Parse serotype from hla type."""
    if len(hla_type.split('-'))<3:
        return '0'
    else:
        return hla_type.split('-')[2][:2]

    
def encode_hla_sero(x_all,x_in,representation):
    """Encode string HLA serotype to numbers."""
    #get 1st subtype
    new_x_all=[get_serotype(x) for x in x_all]
    new_x_in=[get_serotype(x) for x in x_in]
    
    return encode(new_x_all,new_x_in,representation)


def encode_seq(x_all,x_in,representation):
    """Encode string amino acid sequences to numbers."""
    #make peptids 'equal' length
    maxlen=np.max(map(len,x_all))
    x_temp_all=np.array([list(x.zfill(maxlen)) for x in x_all])
    x_temp_in=np.array([list(x.zfill(maxlen)) for x in x_in])
    
    if representation=='simple':
        le = preprocessing.LabelEncoder()
        le.fit(x_temp_all.flatten())
        x_out=np.column_stack([le.transform(x_temp_in[:,i]) for i in range(maxlen)])
    elif representation=='one-hot':
        lb = preprocessing.LabelBinarizer()
        lb.fit(x_temp_all.flatten())
        x_out=np.column_stack([lb.transform(x_temp_in[:,i]) for i in range(maxlen)])
    elif representation=='blosum':
        #read blosum matrix
        blosum62 = pd.read_csv('BLOSUM62.txt',header=None,skiprows=7,delim_whitespace=True,index_col=0)
        blosum_dict=blosum62.transpose().to_dict(orient='list')
        blosum_dict['0']=blosum_dict['*']
        #transform
        x_out=np.column_stack([[blosum_dict[x_temp_in[i,j]] for i in xrange(len(x_in)) ] for j in range(maxlen)])

    return x_out




"""Cross validation functions for keras.

People usually do not cross validate when they have to train for days, but
"""

def my_keras_fit_predict(get_model,X_train,y_train,X_test,
                      validation_split=0.1,patience=1,nb_epoch=100,**kwargs):
    """
    Fit model on train test set with early stopping.
    
    You need to specify the get_model() function which returns the keras model.
    """
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


def my_keras_cv_predict(get_model,x,y,n_folds=3,shuffle=True,seed=42,**kwargs):
    """
    Evaluate model with cross validation.
    
    You need to specify the get_model() function which returns the keras model.
    """
    #res
    y_pred=np.zeros(len(y))
    #folds
    for train_index,test_index in KFold(len(x),n_folds=n_folds,
                                        shuffle=shuffle,random_state=np.random.RandomState(seed)):
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


def my_xgb_cv_predict(params,x,y,n_folds=3,shuffle=True,seed=42,**kwargs):
    """Evaluate model with cross validation."""
    #res
    y_pred=np.zeros(len(y))
    #folds
    for train_index,test_index in KFold(len(x),n_folds=n_folds,
                                        shuffle=shuffle,random_state=np.random.RandomState(seed)):
        #data split
        x_train,y_train,x_test=x[train_index],y[train_index],x[test_index]
        #fit predict
        y_pred[test_index]=my_xgb_fit_predict(params,x_train,y_train,x_test,**kwargs)
    return y_pred



"""Evaluation functions."""

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
    
    
    
def corr_plot(y_reg,y_reg_pred):
    """Plot the correlation of predicted and original values, print coefficient."""
    plt.figure(figsize=(9,9))
    plt.plot(y_reg,y_reg_pred,'.',ms=3,alpha=0.5)
    plt.plot(y_reg,y_reg,lw=3,alpha=0.7)
    plt.xlim(xmin=-0.5)
    plt.xlabel('measured ic50')
    plt.ylabel('predicted ic50')

    from scipy.stats import pearsonr
    print 'Correlation:',pearsonr(y_reg,y_reg_pred)[0]
    
    
    
def bootstrap_auc(y_c,y_pred,N=100):
    """Bootstrap the AUC score."""
    scores=[]
    for i in xrange(N):
        res_y=resample(np.column_stack([y_c,y_pred]))
        scores.append(roc_auc_score(res_y[:,0],res_y[:,1]))
        
    print 'Score is :', '%.4f' % np.mean(scores),
    print '+-','%.4f' % np.std(scores)