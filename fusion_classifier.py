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

def fusion_predict(filename):
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

  print "Starting Image Prediction"
  image_pred = image_predict(users)
  print "Starting Checkin Prediction"
  checkin_pred = checkin_predict(users)
  print "Starting Tweet Prediction"
  tweet_pred = tweet_predict(users)

  print "Fusing Results"
  compiled_features = []
  compiled_targets = []

  for i in range(0, len(users)):
    dummy = []
    dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
    dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
    compiled_features.append(dummy)
    compiled_targets.append(genders[i]+" # "+ages[i])

  print "Starting Fusion Prediction"
  pred = fusionClassifier.predict(compiled_features)
  print "Prediction Done, Report below\n"
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

  print "Starting Image Prediction"
  image_pred = image_predict(users)
  print "Starting Checkin Prediction"
  checkin_pred = checkin_predict(users)
  print "Starting Tweet Prediction"
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

  fusionClassifier = SVC()
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

    for i in range(0, len(users)):
      if i in traincv:
        train.append(users[i])
        g_train.append(genders[i])
        a_train.append(ages[i])
      else:
        target.append(users[i])
        g_target.append(genders[i])
        a_target.append(ages[i])

    print "Starting Image Prediction for training"
    image_pred = image_predict(train)
    print "Starting Checkin Prediction for training"
    checkin_pred = checkin_predict(train)
    print "Starting Tweet Prediction for training"
    tweet_pred = tweet_predict(train)

    compiled_features = []
    compiled_targets = []

    for i in range(0, len(train)):
      dummy = []
      dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
      dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
      compiled_features.append(dummy)
      compiled_targets.append(g_train[i]+" # "+a_train[i])

    print "Training Fusion Classifier"
    fusionClassifier = SVC()
    fusionClassifier.fit(compiled_features, compiled_targets)



    print "Starting Image Prediction for testing"
    image_pred = image_predict(target)
    print "Starting Checkin Prediction for testing"
    checkin_pred = checkin_predict(target)
    print "Starting Tweet Prediction for testing"
    tweet_pred = tweet_predict(target)

    compiled_features = []
    compiled_targets = []

    for i in range(0, len(target)):
      dummy = []
      dummy.extend(gender_LE.transform([image_pred[i][0], checkin_pred[i][0], tweet_pred[i][0]]))
      dummy.extend(age_LE.transform([image_pred[i][1], checkin_pred[i][1], tweet_pred[i][1]]))
      compiled_features.append(dummy)
      compiled_targets.append(g_target[i]+" # "+a_target[i])

    print "Predicting Fusion Prediction"
    pred = fusionClassifier.predict(compiled_features)
    print "Prediction Done, Report below\n"
    print classification_report(compiled_targets, pred)
    z+=1