from pymongo import *
import csv
import nltk.classify.util
import datetime
from nltk.classify.maxent import MaxentClassifier
from sklearn.cross_validation import KFold


# ({'h':True},'neg')
file = open('train.csv', 'rb')
reader = csv.reader(file, 'excel')
users = {}
for row in reader:
    users[row[0]] = {'gender': row[1], 'age': row[2]}

client = MongoClient('localhost', 27017)
db = client.test
checkins = db.checkins.find()


def generate_feats(l):
    return dict([(word, True) for word in l])


# split the createdAt into several categories
def determine_time_of_day(t):
    h = t.hour
    if h in range(0, 4):
        return 1
    elif h in range(4, 8):
        return 2
    elif h in range(8, 12):
        return 3
    elif h in range(12, 16):
        return 4
    elif h in range(16, 20):
        return 5
    elif h in range(20, 24):
        return 6


def convert_time(t):
    return determine_time_of_day(datetime.datetime.fromtimestamp(t))


age_feats = []
gender_feats = []

for c in checkins:
    names = []
    if 'venue' not in c:
        continue
    time = float(c['createdAt']) + 60 * float(c['timeZoneOffset'])
    converted = 'time_of_day%s' % convert_time(time)
    names.append(converted)
    venue = c['venue']
    if 'location' in venue:
        names.append(venue['location']['cc'])
    if 'categories' in venue:
        categories = venue['categories']
        for cate in categories:
            names.append(cate['name'])
    if c['userId'] not in users:
        continue
    feat = generate_feats(names)
    age_feats.append((feat, users[c['userId']]['age']))
    gender_feats.append((feat, users[c['userId']]['gender']))

kf = KFold(n=len(age_feats), n_folds=10, shuffle=False, random_state=None)
age_classifier = MaxentClassifier
gender_classifier = MaxentClassifier
age_accs = []
gender_accs = []
for train, test in kf:
    age_train_feats = age_feats[train[0]:train[-1]]
    age_test_feats = age_feats[test[0]:test[-1]]
    gender_train_feats = gender_feats[train[0]:train[-1]]
    gender_test_feats = gender_feats[test[0]:test[-1]]
    age_classifier = age_classifier.train(age_train_feats, max_iter=30)
    age_accs.append(nltk.classify.util.accuracy(age_classifier, age_test_feats))

print "age acc:", sum(age_accs) / 10.0
