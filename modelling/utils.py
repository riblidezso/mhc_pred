import time

import numpy as np
import pandas as pd

from sklearn import preprocessing


def load_data(hla_representation='simple',seq_representation='simple'):
    start=time.time()
    
    #load train data
    print 'Reading from file...'
    dataf='benchmark_mhci_reliability/binding/bd2013.1/bdata.20130222.mhci.public.1.txt'
    data=pd.read_csv(dataf,sep='\t')
    
    #select only human and exact measurements (no >,<)
    data=data[(data.species=='human') &(data.inequality=='=') ][['mhc','sequence','meas','peptide_length']]
     
    print 'Creating representation...'
    #encode hla
    x_hla=encode_hla(data.mhc.values,hla_representation)

    #encode amino acids
    x_seq=encode_seq(data.sequence.values,seq_representation)
    
    #stack columns together
    x=np.column_stack([x_hla,x_seq,data.peptide_length.values])

    #predict log10 value
    y=np.log10(data.meas.values)

    #permute arrays as they are ordered!!!
    perm=np.random.permutation(len(x))
    x,y=x[perm],y[perm]
    
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