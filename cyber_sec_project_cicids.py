#!/usr/bin/env python

import matplotlib
import sklearn.metrics as m
import warnings
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
import numpy as np
# import time
# import sklearn
# import imblearn

# Ignore warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataframe1=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
dataframe2=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
dataframe3=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv")
dataframe4=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv")
dataframe5=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
dataframe6=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
dataframe7=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv")
dataframe8=pd.read_csv("./MachineLearningCSV/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv")

# import xgboost as xgb

# Concat all days of data
dataframe = pd.concat([dataframe1,dataframe2])
del dataframe1,dataframe2
dataframe = pd.concat([dataframe,dataframe3])
del dataframe3
dataframe = pd.concat([dataframe,dataframe4])
del dataframe4
dataframe = pd.concat([dataframe,dataframe5])
del dataframe5
dataframe = pd.concat([dataframe,dataframe6])
del dataframe6
dataframe = pd.concat([dataframe,dataframe7])
del dataframe7
dataframe = pd.concat([dataframe,dataframe8])
del dataframe8

dataframe.info()
dataframe.head()


# Remove infinity and NaN values.
for name in dataframe.columns:
    dataframe = dataframe[dataframe[name] != "Infinity"]
    dataframe = dataframe[dataframe[name] != np.nan]
    dataframe = dataframe[dataframe[name] != ",,"]
    dataframe = dataframe[dataframe[name] != np.Infinity]
    dataframe = dataframe[dataframe[name] != -np.Infinity]

# Changing relevant columns to numeric data.
dataframe[['Flow Bytes/s', ' Flow Packets/s']] = dataframe[['Flow Bytes/s', ' Flow Packets/s']].apply(pd.to_numeric) 

    
# Date preprocesing, removing the least useful columns beforehand
dataframe.drop([' Bwd PSH Flags'], axis=1, inplace=True)
dataframe.drop([' Bwd URG Flags'], axis=1, inplace=True)
dataframe.drop(['Fwd Avg Bytes/Bulk'], axis=1, inplace=True)
dataframe.drop([' Fwd Avg Packets/Bulk'], axis=1, inplace=True)
dataframe.drop([' Fwd Avg Bulk Rate'], axis=1, inplace=True)
dataframe.drop([' Bwd Avg Bytes/Bulk'], axis=1, inplace=True)
dataframe.drop([' Bwd Avg Packets/Bulk'], axis=1, inplace=True)
dataframe.drop(['Bwd Avg Bulk Rate'], axis=1, inplace=True)

dataframe.info()
dataframe.head()
# This helps if we get nan value issues sometimes.
dataframe = dataframe.reset_index()


#Create training and testing data
from sklearn.model_selection import train_test_split
trainData, testData=train_test_split(dataframe,test_size=0.3, random_state=10)

trainData.describe()
testData.describe()


# We need to scale all numeric values to same mean and std
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

# select numerical columns and scale them.
data_columns = trainData.select_dtypes(include=['float64','int64']).columns
scaled_traindata = standard_scaler.fit_transform(trainData.select_dtypes(include=['float64','int64']))
scaled_testdata = standard_scaler.fit_transform(testData.select_dtypes(include=['float64','int64']))

# convert to dataframe
scaled_train_dataframe = pd.DataFrame(scaled_traindata, columns = data_columns)
scaled_test_dataframe = pd.DataFrame(scaled_testdata, columns = data_columns)



from sklearn.preprocessing import OneHotEncoder 

one_hot_enc = OneHotEncoder() 

traindata_dep = trainData[' Label'].values.reshape(-1,1)
traindata_dep = one_hot_enc.fit_transform(traindata_dep).toarray()
testdata_dep = testData[' Label'].values.reshape(-1,1)
testdata_dep = one_hot_enc.fit_transform(testdata_dep).toarray()



train_X=scaled_train_dataframe
train_y=traindata_dep[:,0]

test_X=scaled_test_dataframe
test_y=testdata_dep[:,0]



# Now we will try to fit data to random forest classifier
# to find out most useful features
from sklearn.ensemble import RandomForestClassifier
random_forest_cls = RandomForestClassifier()

random_forest_cls.fit(train_X, train_y)

# create score from classifier's feature importance values and sort the columns
feature_score = np.round(random_forest_cls.feature_importances_,3)
feature_scores_dataframe = pd.DataFrame({'feature':train_X.columns,'importance':feature_score})
feature_scores_dataframe = feature_scores_dataframe.sort_values('importance',ascending=False).set_index('feature')

# plot feature scores
plt.rcParams['figure.figsize'] = (11, 4)
feature_scores_dataframe.plot.bar()



# Eliminate features recursively
from sklearn.feature_selection import RFE
import itertools

random_forest_cls = RandomForestClassifier()

# select 20 attributes
rfe = RFE(random_forest_cls, n_features_to_select=20)
rfe = rfe.fit(train_X, train_y)

feature_map = [(i, v) for i, v in itertools.zip_longest(rfe.get_support(), train_X.columns)]
selected_features = [v for i, v in feature_map if i==True]

selected_features

# select important features
a = [i[0] for i in feature_map]
train_X = train_X.iloc[:,a]
test_X = test_X.iloc[:,a]



# Partition the datasets in x,y train, test
X_train,X_test,Y_train,Y_test = train_test_split(train_X,train_y,train_size=0.70, random_state=2)

# Classifiers
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.naive_bayes import BernoulliNB 

# Train Decision Tree Model
decision_tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
decision_tree_classifier.fit(X_train, Y_train)

# Train Gaussian Naive Baye Model
naive_bayes_classifier = BernoulliNB()
naive_bayes_classifier.fit(X_train, Y_train)

# Random Forest Model
random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, Y_train)




# Model evaluation
from sklearn import metrics

models = []
models.append(('Random Forest Classifier', random_forest_classifier))
models.append(('Naive Bayes Classifier', naive_bayes_classifier))
models.append(('Decision Tree Classifier', decision_tree_classifier))

# calculate evaluation metrics of different models
for name, model in models:
    scores = cross_val_score(model, X_train, Y_train, cv=10)
    accuracy = metrics.accuracy_score(Y_train, model.predict(X_train))
    confusion_matrix = metrics.confusion_matrix(Y_train, model.predict(X_train))
    classification = metrics.classification_report(Y_train, model.predict(X_train))
    print()
    print('---------------------------------------- {} Model Evaluation ----------------------------------'.format(name))
    print()
    print("Cross Validation Mean Score:" "\n", scores.mean())
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()



#Validate Models
for name, model in models:
    accuracy = metrics.accuracy_score(Y_test, model.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, model.predict(X_test))
    classification = metrics.classification_report(Y_test, model.predict(X_test))
    print()
    print('---------------------------------------- {} Model Test Results ----------------------------------------'.format(name))
    print()
    print ("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification) 
    print()        



# predicting for test data
predict_NB = naive_bayes_classifier.predict(test_X)
predict_dt = decision_tree_classifier.predict(test_X)
predict_rf = random_forest_classifier.predict(test_X)

