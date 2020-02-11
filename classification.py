from __future__ import division
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score

#Reading the data
data = pd.read_csv('pulsar_stars.csv')

#Checking data's structures
print('Number of features: %s' %data.shape[1])
print('Number of examples: %s' %data.shape[0])


#Naming different feautures in the data
data.columns = ['m-profile', 'std-profile', 'kur-profile', 'skew-profile', 'mean-dmsnr',
               'std-dmsnr', 'kurtosis-dmsnr', 'skew-dmsnr', 'target']

#Taking out the labels - we need them for supervised training
Y = data['target'].values
x_data = data.drop(['target'],axis=1)
#Renormalization of the data
X = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

#Setting train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state=1)

#Performing logistic regression
LogRegression = LogisticRegression()
LogRegression.fit(X_train,Y_train)
LogRegression_prediction = LogRegression.predict(X_test)
Log_score = LogRegression.score(X_test,Y_test)
#Making the confusion matrix and classification report
Cm_Log = confusion_matrix(Y_test,LogRegression_prediction)
Cm_Log = pd.DataFrame(Cm_Log)
Cm_Log["total"]=Cm_Log[0]+Cm_Log[1]
Cr_Log = classification_report(Y_test,LogRegression_prediction)
#Printing obtained results
print('Classification report for Logistic Regression: \n',Cr_Log)
print(Cm_Log)



#Performing classification with decision tree
Decision_tree = DecisionTreeClassifier()
Decision_tree.fit(X_train,Y_train)
prediction_dt = Decision_tree.predict(X_test)
#Making the confusion matrix and classification report
dt_cm=confusion_matrix(Y_test,prediction_dt)
dt_cm=pd.DataFrame(dt_cm)
dt_cm["total"]=dt_cm[0]+dt_cm[1]
Decission_score = Decision_tree.score(X_test,Y_test)
#Printing obtained results
Cr_dt=classification_report(Y_test,prediction_dt)
print('Classification report for Decision tree: \n',Cr_dt)
print(dt_cm)






#Same for the random forest
Random_forest = RandomForestClassifier(n_estimators = 200,random_state=1)
Random_forest.fit(X_train,Y_train)
prediction_rf = Random_forest.predict(X_test)
Random_score = Random_forest.score(X_test,Y_test)
rf_cm=confusion_matrix(Y_test,prediction_rf)
Cr_rf=classification_report(Y_test,prediction_rf)
print('Classification report for Random Forest: \n',Cr_rf)
print(rf_cm)


#Same for the neural network. 
from sklearn.neural_network import MLPClassifier
Neural_network = MLPClassifier(hidden_layer_sizes=(22), activation='logistic', max_iter = 150, solver = 'adam')
Neural_network.fit(X_train, Y_train)
prediction_nn = Neural_network.predict(X_test)
Neural_score = Neural_network.score(X_test,Y_test)
nn_cm=confusion_matrix(Y_test,prediction_rf)
Cr_nn = classification_report(Y_test,prediction_nn)
print('Classification report for Neural Networks: \n', Cr_nn)
print(nn_cm)


#Bar chart with the accuracy of the classification process obtained with different models
algorithms = ("Logistic Regression","Decision Tree","Random Forest","Neural network")
scores = (Log_score,Decission_score,Random_score,Neural_score)
y_pos = np.arange(1,5)
colors = ("blue","green","red","orange")
plt.figure(figsize=(24,12))
plt.xticks(y_pos,algorithms,fontsize=18)
plt.yticks(np.arange(0.00, 1.01, step=0.01))
plt.ylim(0.90,1.00)
plt.bar(y_pos,scores,color=colors)
plt.grid()
plt.suptitle("Accuracy of different models",fontsize=24)
plt.show()


#ROC curves for each of the methods, using again sklearn package
plt.figure(figsize=(24,12))
fpr_dt, tpr_dt, thresholds = roc_curve(Y_test, prediction_dt)
fpr_rf, tpr_rf, thresholds = roc_curve(Y_test, prediction_rf)
fpr_lr, tpr_lr, thresholds = roc_curve(Y_test, LogRegression_prediction)
fpr_nn, tpr_nn, thresholds = roc_curve(Y_test, prediction_nn)
plt.plot([0, 1], [0, 1], 'k--',color="black")
plt.plot(fpr_dt, tpr_dt, color="green", label = 'Decision Trees')
plt.plot(fpr_lr, tpr_lr, color="blue", label = 'Logistic Regression')
plt.plot(fpr_rf, tpr_rf, color="red", label = 'Random Forest')
plt.plot(fpr_nn, tpr_nn, color="orange", label = 'Neural Network')
plt.legend(fontsize = 16)
plt.title('ROC curves for different methods', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 16)
plt.ylabel('True Positive Rate', fontsize = 16)

#ROC score for every method
roc_score_lr = roc_auc_score(Y_test, LogRegression_prediction)
roc_score_dt = roc_auc_score(Y_test, prediction_dt)
roc_score_rf = roc_auc_score(Y_test, prediction_rf)
roc_score_nn = roc_auc_score(Y_test, prediction_nn)

print('ROC score logistic regression: \n',roc_score_lr)
print('ROC score decission trees: \n',roc_score_dt)
print('ROC score random forest: \n',roc_score_rf)
print('ROC score neural networks: \n',roc_score_nn)