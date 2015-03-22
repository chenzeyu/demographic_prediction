from pymongo import *
import csv
import nltk.classify.util
import datetime
from nltk.classify.maxent import MaxentClassifier
from sklearn.cross_validation import KFold
import pickle


def load_checkins():
    client = MongoClient('localhost', 27017)
    db = client.test
    return db.checkins.find()


# generate features for nltk classifiers
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


# convert checkin time to datetime
def convert_time(t):
    return determine_time_of_day(datetime.datetime.fromtimestamp(t))


def obtain_feats(checkins, users=None):
    age_feats = []
    gender_feats = []
    for c in checkins:
        names = []
        if 'venue' in c:
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
            feat = generate_feats(names)
            if users == None:
                age_feats.append((feat, 'dummy'))
                gender_feats.append((feat, 'dummy'))
            else:
                if c['userId'] not in users:
                    continue
                age_feats.append((feat, users[c['userId']]['age']))
                gender_feats.append((feat, users[c['userId']]['gender']))
    return age_feats, gender_feats


def save_classifier(classifier, ctype='age'):
    name = "checkin_%s_classifier.pickle" % ctype
    f = open(name, 'wb')
    pickle.dump(classifier, f)
    f.close()


def load_classifier(ctype='age'):
    name = "checkin_%s_classifier.pickle" % ctype
    f = open(name)
    classifier = pickle.load(f)
    f.close()
    return classifier


# perform ten fold and return the classifier
def checkin_perform_ten_fold(feats):
    accs = []
    kf = KFold(n=len(feats), n_folds=10, shuffle=False, random_state=None)
    classifier = MaxentClassifier
    for train, test in kf:
        train_feats = feats[train[0]:train[-1]]
        test_feats = feats[test[0]:test[-1]]
        classifier = classifier.train(train_feats, max_iter=30)
        accs.append(nltk.classify.util.accuracy(classifier, test_feats))
    print "average accuracy:", sum(accs) / 10.0
    return classifier


# feats_tuple = obtain_feats(load_checkins(), users)
# age_classifier = checkin_perform_ten_fold(feats_tuple[0])
# gender_classifier = checkin_perform_ten_fold(feats_tuple[1])
# save_classifier(age_classifier, 'age')
# save_classifier(gender_classifier, 'gender')


def get_checkins_of_user(uid):
    cs = []
    for checkin in load_checkins():
        if checkin['userId'] == uid:
            cs.append(checkin)
    return cs


def checkin_predict(user_list, classifier=False, classifiers=[]):
    if not classifier:
        age_classifier = load_classifier('age')
        gender_classifier = load_classifier('gender')
    else:
        age_classifier = classifiers[0]
        gender_classifier = classifiers[1]
    results = []
    for u in user_list:
        feats = obtain_feats(get_checkins_of_user(u))
        if not feats[0] and not feats[1]:
            results.append(('none', 'none'))
            continue
        ages = {}
        genders = {}
        for age_feat in feats[0]:
            af = age_feat[0]
            age = age_classifier.classify(af)
            if age in ages:
                ages[age] += 1
            else:
                ages[age] = 1

        for gender_feat in feats[1]:
            af = gender_feat[0]
            gender = gender_classifier.classify(af)
            if gender in genders:
                genders[gender] += 1
            else:
                genders[gender] = 1
        age = max(ages, key=ages.get)
        gender = max(genders, key=genders.get)
        results.append((gender, age))
    return results


def checkin_train_and_predict(train_l, genders, ages, test_l):
    age_classifier = MaxentClassifier
    gender_classifier = MaxentClassifier
    checkins = []
    _users = {}
    for i in range(0, len(train_l)):
        _users[train_l[i]] = {'gender': genders[i], 'age': ages[i]}
    for u in train_l:
        checkins.append(get_checkins_of_user(u))
    checkins = [item for sublist in checkins for item in sublist]
    feats = obtain_feats(checkins, _users)
    age_classifier = age_classifier.train(feats[0], max_iter=30)
    gender_classifier = gender_classifier.train(feats[1], max_iter=30)
    checkin_predict(test_l, classifier=True, classifiers=[age_classifier, gender_classifier])

