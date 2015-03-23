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
from sklearn.decomposition import SparseCoder
from sklearn.decomposition import DictionaryLearning

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM

import skimage.io as io
import skimage.transform as trans


# Initialize
#########################################
START = time.time()
#########################################


# Get user info
#########################################
userMap = {}

f = open("train.csv")
reader = csv.reader(f, 'excel')

reader.next()

for row in reader:
	userMap[row[0]] = (row[1], row[2])

f.close()
#########################################


# Get image data
#########################################
start = time.time()
print "Start getting image data", time.clock()

f = open("image_features.bin", "rb")
oriData = pickle.load(f)
f.close()

f = open("image_fileMap.bin", "rb")
fileMap = pickle.load(f)
f.close()

genders = []
ages = []
both = []

train = []
test = []

for i in range (0, 4113):
	userId = fileMap[i]
	try:
		gender, age = userMap[userId]
		genders.append(gender)
		ages.append(age)
		both.append(gender + " # " + age)
		train.append(oriData[i])
	except:
		test.append(oriData[i])
	
train = np.array(train)
test = np.array(test)


size = len(train)
end = time.time()

print "End getting image data", time.clock()

print "Elapsed Time:", end-start
#########################################


# Do feature selection/extraction(PCA)
#########################################
decomp = RandomizedPCA(n_components = 25)
train  = decomp.fit_transform(train)
#########################################


# Initialize Classifier(SVC)
#########################################
genderClassifier = DecisionTreeClassifier()
ageClassifier = DecisionTreeClassifier()
bothClassifier = DecisionTreeClassifier()

end = time.time()
#########################################

# genderClassifier.fit(train, genders)
# ageClassifier.fit(train, ages)

# with open('image_classifiers.bin', 'wb') as fp:
# 	pickle.dump(genderClassifier, fp)
# 	pickle.dump(ageClassifier, fp)
# fp.close()

# Run 10-Fold Validation
#########################################
cv = cross_validation.KFold(size, n_folds=10)
resultsGender = []
resultsAge = []
resultsBoth = []
i = 1
for traincv, testcv in cv:
	print "Starting iteration ", i
	start = time.time()

	s = time.time()
	genderClassifier.fit(train[traincv[0]:traincv[-1]], genders[traincv[0]:traincv[-1]])
	ageClassifier.fit(train[traincv[0]:traincv[-1]], ages[traincv[0]:traincv[-1]])
	bothClassifier.fit(train[traincv[0]:traincv[-1]], both[traincv[0]:traincv[-1]])
	e = time.time()

	print "Classifier Building Time: ", e-s

	gP = genderClassifier.predict(train[testcv[0]:testcv[-1]])
	aP = ageClassifier.predict(train[testcv[0]:testcv[-1]])
	bP = bothClassifier.predict(train[testcv[0]:testcv[-1]])
	resultsGender.append( accuracy_score(genders[testcv[0]:testcv[-1]], gP,normalize=True ))
	resultsAge.append( accuracy_score(ages[testcv[0]:testcv[-1]], aP,normalize=True) )
	resultsBoth.append( accuracy_score(both[testcv[0]:testcv[-1]], bP,normalize=True) )

	print "Iteration ", i , "results:"
	print "Gender Accuracy: ", resultsGender[i-1]
	print classification_report(genders[testcv[0]:testcv[-1]], gP), "\n"
	print "Age Accuracy: ", resultsAge[i-1]
	print classification_report(ages[testcv[0]:testcv[-1]], aP), "\n"
	print "Combined Accuracy: ", resultsBoth[i-1]
	print classification_report(both[testcv[0]:testcv[-1]], bP), "\n"

	end = time.time()
	print "Elapsed Time for iteration ", i, " : ", end-start
	print "\n"
	i+=1
	gc.collect()

print "Gender Accuracy Results: " + str( np.array(resultsGender).mean() )
print "Age Accuracy Results: " + str( np.array(resultsAge).mean() )
print "Combined Accuracy Results: " + str( np.array(resultsBoth).mean() )

END = time.time()
print "Finished, Total Elapsed Time: ", END-START
#########################################


