#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score

train_file = pd.read_csv('./train_users_2.csv')
test_file = pd.read_csv('./test_users.csv')
labels = train_file.country_destination.values
train_file = train_file.drop(['country_destination'],axis=1)
piv_train = train_file.shape[0]
id_test = test_file['id']
train_file = pd.concat((train_file, test_file), axis=0, ignore_index=True)
train_file = train_file.fillna(-1)
########### Feature Engineering  ##################
# date_account_created
dac = np.vstack(string.split('-') for string in train_file.date_account_created.astype(str))
train_file['dac_year'] = dac[:,0]
train_file['dac_month'] = dac[:,1]
train_file['dac_date'] = dac[:,2]

# timestamp first active
tfa = np.vstack((string[0:4],string[4:6],string[6:8]) for string in train_file.timestamp_first_active.astype(str))
train_file['tfa_year'] = tfa[:,0]
train_file['tfa_month'] = tfa[:,1]
train_file['tfa_date'] = tfa[:,2]
train_file =train_file.drop(['id','date_first_booking'],axis=1)
train_file = train_file.drop(['timestamp_first_active','date_account_created'], axis=1)

av = train_file.age.values
train_file['age'] = np.where(np.logical_or(av<14, av>100), 0, av)
# One Hot Encoding#
train_file_dummy=[]
features = ['gender','age','signup_method','signup_flow','language','affiliate_channel','affiliate_provider','first_affiliate_tracked','signup_app','first_device_type','first_browser']
for feature in features:
    train_dummy = pd.get_dummies(train_file[feature],prefix=feature)
    train_file = train_file.drop(feature,axis=1)
    train_file = pd.concat((train_dummy,train_file),axis=1)

# Train and Test data split
vals = train_file.values
train_data = vals[:piv_train]
le = LabelEncoder()
train_labels = le.fit_transform(labels)   
test_data = vals[piv_train:]
# Train the Classifier.
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
xgb.fit(train_data, train_labels)
y_pred = xgb.predict_proba(test_data)
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)