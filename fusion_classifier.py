import base64
import csv
import gc
import time
try:
   import cPickle as pickle
except:
   import pickle

import numpy as np

from sklearn.decomposition import RandomizedPCA

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

import skimage.io as io
import skimage.transform as trans

from image_classifier_lib import *
from checkin_classifier import *
from tweet_classifier import *

def predict(filename):
  file = open(filename)
  reader = csv.reader(file, 'excel')
  reader.next()
  users = []
  genders = []
  ages = []
  gender_LE = LabelEncoder()
  age_LE = LabelEncoder()
  gender_LE.fit(["none", "MALE", "FEMALE"])
  age_LE.fit(["none","18-24","25-34","35-49","50-64","65-xx"])

  f = open("fusion_classifiers.bin", "rb")
  fusionClassifier = pickle.load(f)
  f.close()

  for row in reader:
    users.append(row[0])
    genders.append(row[1])
    ages.append(row[2])

  image_pred = image_predict(users)
  checkin_pred = checkin_predict(users)
  tweet_pred = tweet_predict(users)

  compiled_features = []
  compiled_targets = []

  for i in range(0, len(users)):
    dummy = []
    dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
    dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
    compiled_features.append(dummy)
    compiled_targets.append(genders[i]+" # "+ages[i])

  pred = fusionClassifier.predict(compiled_features)
  print classification_report(compiled_targets, pred)



def train(filename):
  file = open(filename)
  reader = csv.reader(file, 'excel')
  reader.next()
  users = []
  genders = []
  ages = []
  gender_LE = LabelEncoder()
  age_LE = LabelEncoder()
  gender_LE.fit(["none", "MALE", "FEMALE"])
  age_LE.fit(["none","18-24","25-34","35-49","50-64","65-xx"])

  for row in reader:
    users.append(row[0])
    genders.append(row[1])
    ages.append(row[2])

  image_pred = image_predict(users)
  checkin_pred = checkin_predict(users)
  tweet_pred = tweet_predict(users)

  compiled_features = []
  compiled_targets = []

  for i in range(0, len(users)):
    dummy = []
    dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
    dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
    compiled_features.append(dummy)
    compiled_targets.append(genders[i]+" # "+ages[i])

  compiled_features = np.array(compiled_features)

  fusionClassifier = GradientBoostingClassifier()
  fusionClassifier.fit(compiled_features, compiled_targets)

  with open('fusion_classifiers.bin', 'wb') as fp:
    pickle.dump(fusionClassifier, fp)
  fp.close()

  cv = cross_validation.KFold(len(users), n_folds=10)
  i = 1
  for traincv, testcv in cv:
    print "Starting Iteration", i
    fusionClassifier = GradientBoostingClassifier()
    fusionClassifier.fit(compiled_features[traincv[0]:traincv[-1]], compiled_targets[traincv[0]:traincv[-1]])
    pred = fusionClassifier.predict(compiled_features[testcv[0]:testcv[-1]])
    print classification_report(compiled_targets[testcv[0]:testcv[-1]], pred), "\n"
    i+=1



def perform_k_fold(filename):
  file = open(filename)
  reader = csv.reader(file, 'excel')
  reader.next()
  users = []
  genders = []
  ages = []
  gender_LE = LabelEncoder()
  age_LE = LabelEncoder()
  gender_LE.fit(["none", "MALE", "FEMALE"])
  age_LE.fit(["none","18-24","25-34","35-49","50-64","65-xx"])

  for row in reader:
    users.append(row[0])
    genders.append(row[1])
    ages.append(row[2])

  cv = cross_validation.KFold(len(users), n_folds=10)

  z = 1
  for traincv, testcv in cv:
    print "Starting Iteration ", z
    train = []
    g_train = []
    a_train = []
    target = []
    g_target = []
    a_target = []

    for z in range(0, len(users)):
      if z in traincv:
        train.append(users[z])
        g_train.append(genders[z])
        a_train.append(ages[z])
      else:
        target.append(users[z])
        g_target.append(genders[z])
        a_target.append(ages[z])



    image_pred = image_train_and_predict(train, g_train, a_train, target)
    checkin_pred = checkin_train_and_predict(train, g_train, a_train, target)
    tweet_pred = tweet_train_and_predict(train, g_train, a_train, target)

    compiled_features = []
    compiled_targets = []

    for i in range(0, len(train)):
      dummy = []
      dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
      dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
      compiled_features.append(dummy)
      compiled_targets.append(g_train[i]+" # "+a_train[i])

    fusionClassifier = GradientBoostingClassifier()
    fusionClassifier.fit(compiled_features, compiled_targets)



    image_pred = image_predict(target)
    checkin_pred = checkin_pred(target)
    tweet_pred = tweet_pred(target)

    compiled_features = []
    compiled_targets = []

    for i in range(0, len(target)):
      dummy = []
      dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
      dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
      compiled_features.append(dummy)
      compiled_targets.append(g_train[i]+" # "+a_train[i])

    pred = fusionClassifier.predict(compiled_features)
    print classification_report(compiled_targets, pred)
    z+=1


perform_k_fold("train.csv")