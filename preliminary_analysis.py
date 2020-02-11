from __future__ import division
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')



#Reading data
data = pd.read_csv('pulsar_stars.csv')

#Looking at the data's structure
print('Number of features: %s' %data.shape[1])
print('Number of examples: %s' %data.shape[0])

#Labeling the different columns that represent the features of the data
data.columns = ['m-profile', 'std-profile', 'kur-profile', 'skew-profile', 'mean-dmsnr',
               'std-dmsnr', 'kurtosis-dmsnr', 'skew-dmsnr', 'target']

#Taking the labels in a separete variable
Y = data['target'].values
#Renormalazing the data
x_data = data.drop(['target'],axis=1)
X = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


data.isnull().sum()
#We will plot a correlation matrix of the different feautures
plt.figure(figsize=(15,5))
#Here we define the bar
#cbar_kws whenever there is a bar or not
# annot - To have numbers inside the map or not
cbar_kws1 = { 'ticks' : [-1,-0.75, -0.5,-0.25, 0, 0.25, 0.5, 0.75, 1], 'orientation': 'horizontal'}
#Defining the heat map.
sns.heatmap(data.corr(), cbar_kws = cbar_kws1, cmap='seismic', annot=True, linewidths = 0.2)
plt.xticks(rotation=0)
plt.title("Pearson correlations", fontsize = 24 )



#Here we redefine the data. This means that we take all the features an we put in a matrix
#and we seperate the column with the information whenever the star is a pulsar or not
#this will be important when we try to classify the data, since now we have our target data
#that classifies the stars into ones that are pulsars and ones that are not    
data1 = data.groupby('target')[['m-profile', 'std-profile', 'kur-profile', 'skew-profile', 'mean-dmsnr',
               'std-dmsnr', 'kurtosis-dmsnr', 'skew-dmsnr', 'target']].mean().reset_index(drop = True)

data1= data1.transpose().reset_index()
data1.columns = ['features', 'Not Pulsar Star', 'Pulsar Star']

#Plottng the average values of the features for pulsars and for no pulsars
plt.figure(figsize=(10,8))
plt.subplot(211)
vis2=sns.pointplot(data=data1.iloc[1:], x='features', y='Not Pulsar Star',color='y', label='Not Pulsar Star')
vis3=sns.pointplot(data=data1.iloc[1:], x='features', y='Pulsar Star', color='b', label='Pulsar Star')

plt.title('Mean values of features for every target class', fontsize=15)
plt.xlabel('Feature', fontsize=13)
plt.ylabel('Values', fontsize=13)
plt.xticks(rotation=30)




#Pair plot for different feautures. Pulsars and not pulsars are labbeled with different colors
plt.figure(figsize=(8,15))
sns.scatterplot(data=data[data['target']==1], x='kur-profile', y='skew-profile', label='Not Pulsar Star')
sns.scatterplot(data=data[data['target']==0], x='kur-profile', y='skew-profile', label='Pulsar Star')
plt.title('Profile', fontsize=15)
plt.xlabel('Kurtosis profile', fontsize=13)
plt.ylabel('Skweness profile', fontsize=13)

plt.figure(figsize=(8,15))
sns.scatterplot(data=data[data['target']==0], x='kurtosis-dmsnr', y='skew-dmsnr', label='Pulsar Star')
sns.scatterplot(data=data[data['target']==1], x='kurtosis-dmsnr', y='skew-dmsnr', label='Not Pulsar Star')
plt.title('DMSNR', fontsize=15)
plt.xlabel('Kurtosis DMSNR', fontsize=13)
plt.ylabel('Skweness DMSNR', fontsize=13)

plt.figure(figsize=(8,15))
sns.scatterplot(data=data[data['target']==0], x='m-profile', y='kur-profile', label='Pulsar Star')
sns.scatterplot(data=data[data['target']==1], x='m-profile', y='kur-profile', label='Not Pulsar Star')
plt.title('Profile', fontsize=15)
plt.xlabel('Kurtosis DMSNR', fontsize=13)
plt.ylabel('Skweness DMSNR', fontsize=13)

plt.figure(figsize=(8,15))
sns.scatterplot(data=data[data['target']==0], x='mean-dmsnr', y='std-dmsnr', label='Pulsar Star')
sns.scatterplot(data=data[data['target']==1], x='mean-dmsnr', y='std-dmsnr', label='Not Pulsar Star')
plt.title('DMSNR', fontsize=15)
plt.xlabel('Mean DMSNR', fontsize=13)
plt.ylabel('Standard DMSNR', fontsize=13)










