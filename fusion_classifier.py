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

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

import skimage.io as io
import skimage.transform as trans

from image_classifier_lib import *
from checkin_classifier import *

def predict(filename):
	file = open(filename, 'rb')
	reader = csv.reader(file, 'excel')
	reader.next()
	users = []
	genders = []
	ages = []
	for row in reader:
  		users.append(row[0])
  		genders.append(row[1])
  		ages.append(row[2])

  	image_pred = image_predict(users)
  	# checkin_pred = checkin_predict(users)

  	print image_pred
  	# print checkin_pred



def perform_k_fold():
	print "dummy"




predict("test.csv")