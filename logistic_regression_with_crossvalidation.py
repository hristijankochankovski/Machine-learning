import numpy as np
import pickle, os
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

np.random.seed()

#Setting the initial parameters
L = 40 # linear system size
J = -1.0 # Ising interaction
T = np.linspace(0.25,4.0,16) # set of temperatures


# url to data
url_main = 'https://physics.bu.edu/~pankajm/ML-Review-Datasets/isingMC/';

######### LOAD DATA
# The data consists of 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25):
data_file_name = "Ising2DFM_reSample_L40_T=All.pkl" 
# The labels are obtained from the following file:
label_file_name = "Ising2DFM_reSample_L40_T=All_labels.pkl"


data = pickle.load(urlopen(url_main + data_file_name)) # pickle reads the file and returns the Python object (1D array, compressed bits)
data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
data=data.astype('int')
data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)
labels = pickle.load(urlopen(url_main + label_file_name)) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

#Splitting the data
num_classes = 2

X_ordered=data[:85000,:]
Y_ordered=labels[:85000]
X_disordered=data[85000:,:]
Y_disordered=labels[85000:]
X=np.concatenate((X_ordered,X_disordered))
Y=np.concatenate((Y_ordered,Y_disordered))

#Setting the parameter for cross-validation
k = 10
kfold = KFold(n_splits = k)
#Performing cross validation
logreg=LogisticRegression()
predicted = cross_val_score(logreg, X, Y, cv=kfold)
print(predicted)
print(np.mean(predicted))




