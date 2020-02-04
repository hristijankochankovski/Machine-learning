import numpy as np
import pickle
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import seaborn as sns


np.random.seed()

#Setting the initial parameters
L = 40 # linear system size
J = -1.0 # Ising interaction
T = np.linspace(0.25,4.0,16) # set of temperatures
T_c = 2.26 # Onsager critical temperature in the TD limit

#Setting the parameters for the neural network
eta_vals = np.logspace(-2, 0, 3)
lmbd_vals = np.logspace(-2, 0, 3)
n_hidden_neurons = 50
epochs = 10
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


DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)


#Main part of the program. It is made of two loops over the different values of
# the learning rate and hyperparameter lambda. Using the function MLPClassifier
#the logistic regression is done
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(50), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs, solver = 'adam')
       

        
        dnn.fit(X_train, Y_train)
        
        DNN_scikit[i][j] = dnn
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", dnn.score(X_test, Y_test))
        print()
        
        
sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

#Gettin the matrix from the accuracy test
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        
        train_pred = dnn.predict(X_train) 
        test_pred = dnn.predict(X_test)

        train_accuracy[i][j] =  accuracy_score(Y_train, train_pred)
        test_accuracy[i][j] =  accuracy_score(Y_test, test_pred)

#Plotting the results     
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Training Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")

plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
plt.yticks(lmbd_vals)

sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
