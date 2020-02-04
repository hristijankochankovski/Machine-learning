import numpy as np
import pickle, os
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


np.random.seed()

#Setting the initial parameters
L = 40 # linear system size
J = -1.0 # Ising interaction
T = np.linspace(0.25,4.0,16) # set of temperatures
T_c = 2.26 # Onsager critical temperature in the TD limit

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

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size= 0.2)


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
print("Test set accuracy: {:.2f}".format(logreg.score(X_test,Y_test)))
print("Training set accuracy: {:.2f}".format(logreg.score(X_train,Y_train)))


# Scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
logreg.fit(X_train_scaled, Y_train)
print("Test set accuracy scaled data: {:.2f}".format(logreg.score(X_test_scaled,Y_test)))
print("Train set accuracy scaled data: {:.2f}".format(logreg.score(X_train_scaled,Y_train)))



