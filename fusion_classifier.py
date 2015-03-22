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

def predict(userId):

def perform_k_fold():