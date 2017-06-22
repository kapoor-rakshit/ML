""" SOURCE : https://pythonprogramming.net/training-testing-machine-learning-tutorial/"""

"""Regression is a form of supervised machine learning, which is where the scientist teaches the machine by showing it features and then showing it what the correct answer is, over and over, to teach the machine.
"Then test" the machine on some unseen data, where the scientist still knows what the correct answer is, but the machine doesn't. """

"""feature is input; label is output.
  Example : if you're trying to predict the type of pet someone will choose, your input features might include age, home region, family income, etc. 
  The label is the final choice, such as dog, fish, iguana, rock, etc.

Once you've trained your model, you will give it sets of new input containing those features; 
it will return the predicted "label" (pet type) for that person."""

import pandas as pd
import quandl                    # quandl.com is financial dataset site 
from math import *
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import pickle
 
dataframe=quandl.get("nse/wipro",authtoken="Yiu9TpqKozssyt4ja9tA",start_date="2017-01-01",end_date="2017-02-05",limit=10)

# wipro NSE code 
# https://www.quandl.com/data/NSE-National-Stock-Exchange-of-India    for more company stock codes
# https://docs.quandl.com/docs/parameters-2                           for params to be filtered

# limit=5 displays the last 5 records from end_date, default end_date is current date

""" print(dataframe) """
# each column is a feature

dataframe["HLChange"]=(dataframe["High"]-dataframe["Low"])/(dataframe["Close"])*100     # adds new features
dataframe["COChange"]=(dataframe["Close"]-dataframe["Open"])/(dataframe["Open"])*100

dataframe=dataframe[["HLChange","COChange","Open","Close"]]                    # Choose which features to use

forecast_on="Close"                                             # feature to predict value on
dataframe.fillna(-99999,inplace=True)                           # if data is NaN fill it with -99999
forecast_for_nextdays=int(ceil(0.3*len(dataframe)))             # if data for 100 days then calculate for (0.3) 30% days = 30 days

print(forecast_for_nextdays)                                                    # get value of days to work on 
dataframe["FutureClose"]=dataframe[forecast_on].shift(-forecast_for_nextdays)   # value in next days : label

print(dataframe)

features=np.array(dataframe.drop(dataframe["FutureClose"],1))   # 1 axis, specify to remove the feature (column)     # all features values (inputs), except FutureClose feature are features

features=preprocessing.scale(features)                                      # make features values as in range -1 to 1

features_lately = features[-forecast_for_nextdays:]                         # last (forecast_for_nextdays) days data
features = features[:-forecast_for_nextdays]                                # remaining are used as features

dataframe.dropna(inplace=True)                                              # if any row has any feature as NaN drop that
labels=np.array(dataframe["FutureClose"])                                   # FutureClose values is used as label (outputs)

features_train,features_test,labels_train,labels_test=cross_validation.train_test_split(features,labels,test_size=0.2)  # 80% is training data and 20% is testing data
                                                                            # create tests and training features and labels

classifier=LinearRegression(n_jobs=-1)           ########                # a classifier chosen with parameters

# n_jobs : an algorithm that can be threaded for high performance. Specify exactly how many threads you'll want. 
# If you put in -1 for the value, then the algorithm will use all available threads.     

"""classifier=svm.SVR(kernel="poly")      kernel = linear, poly, rbf (default), sigmoid     try these kernels for varied accuracies"""
                                          #  makes processing go much faster

classifier.fit(features_train,labels_train)             # train the classifier            #######

with open('regression.pickle','wb') as f:              # to be commented if once pickled  #######
    pickle.dump(classifier, f)                         #####

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

accuracy=classifier.score(features_test,labels_test)    # check accuracy

forecasted_dataframe=classifier.predict(features_lately)    # predict on list

print(forecasted_dataframe)
print(accuracy)
