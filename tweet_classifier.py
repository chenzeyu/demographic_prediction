from pymongo import *
import csv
import re
import nltk.classify.util
from nltk.classify.maxent import MaxentClassifier
from sklearn.cross_validation import KFold
import pickle


def load_tweets():
    client = MongoClient('localhost', 27017)
    db = client.test
    return db.tweets.find()


def remove_url(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)


def remove_at_and_hash(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())


# def set_up():
# for tweet in tweets:
# text = tweet['text']
# user = tweet['userId']
# text = remove_at_and_hash(remove_url(text))
# text_file = open("tweets/%s.txt" % user, "a")
# text_file.write("%s " % text)
# text_file.close()

# generate features for nltk classifiers
file = open('train.csv', 'rb')
reader = csv.reader(file, 'excel')
users = {}
for row in reader:
    users[row[0]] = {'gender': row[1], 'age': row[2]}


def generate_feats(l):
    return dict([(word, points) for word, points in l])


def obtain_feats(u, users=None):
    file = open('liwcresult', 'rb')
    reader = csv.reader(file, delimiter='\t')
    line = reader.next()
    age_feat = None
    gender_feat = None
    for row in reader:
        names = []
        userId = row[0].replace('.txt', '')
        if u == userId:
            for i in range(4, len(row)):
                names.append((line[i], row[i]))
            feat = generate_feats(names)
            if users == None:
                age_feat = ((feat, 'dummy'))
                gender_feat = ((feat, 'dummy'))
            else:
                if userId in users:
                    age_feat = (feat, users[userId]['age'])
                    gender_feat = (feat, users[userId]['gender'])
        else:
            continue
    return age_feat, gender_feat


# kf = KFold(n=len(age_feats), n_folds=10, shuffle=False, random_state=None)
# age_classifier = MaxentClassifier
# gender_classifier = MaxentClassifier
# age_accs = []
# gender_accs = []
# for train, test in kf:
# age_train_feats = age_feats[train[0]:train[-1]]
# age_test_feats = age_feats[test[0]:test[-1]]
# gender_train_feats = gender_feats[train[0]:train[-1]]
# gender_test_feats = gender_feats[test[0]:test[-1]]
# age_classifier = age_classifier.train(age_train_feats, max_iter=20)
# age_accs.append(nltk.classify.util.accuracy(age_classifier, age_test_feats))
# gender_classifier = age_classifier.train(gender_train_feats, max_iter=20)
# gender_accs.append(nltk.classify.util.accuracy(gender_classifier, gender_test_feats))
# print "age acc:", sum(age_accs) / 10.0
# print "gender acc:", sum(gender_accs) / 10.0
# save_classifier(age_classifier, ctype='age')
# save_classifier(gender_classifier, ctype='gender')

def save_classifier(classifier, ctype='age'):
    name = "tweet_%s_classifier.pickle" % ctype
    f = open(name, 'wb')
    pickle.dump(classifier, f)
    f.close()


def load_classifier(ctype='age'):
    name = "tweet_%s_classifier.pickle" % ctype
    f = open(name)
    classifier = pickle.load(f)
    f.close()
    return classifier


def get_tweets_of_user(uid):
    cs = []
    for tweet in load_tweets():
        if tweet['userId'] == uid:
            cs.append(tweet)
    return cs


def tweet_predict(user_list, classifier=False, classifiers=[]):
    if not classifier:
        age_classifier = load_classifier('age')
        gender_classifier = load_classifier('gender')
    else:
        age_classifier = classifiers[0]
        gender_classifier = classifiers[1]
    results = []
    for u in user_list:
        temp_feats = obtain_feats(u)
        if temp_feats[0] is None and temp_feats[1] is None:
            results.append(('none', 'none'))
            continue
        age_feat = temp_feats[0][0]
        gender_feat = temp_feats[1][0]
        age = age_classifier.classify(age_feat)
        gender = gender_classifier.classify(gender_feat)
        results.append((gender, age))
    return results


def tweet_train_and_predict(train_l, genders, ages, test_l):
    age_classifier = MaxentClassifier
    gender_classifier = MaxentClassifier
    _users = {}
    age_feats = []
    gender_feats = []
    for i in range(0, len(train_l)):
        _users[train_l[i]] = {'gender': genders[i], 'age': ages[i]}
    for u in train_l:
        temp_feats = obtain_feats(u, _users)
        age_feat = temp_feats[0]
        gender_feat = temp_feats[1]
        age_feats.append(age_feat)
        gender_feats.append(gender_feat)
    age_classifier = age_classifier.train(age_feats, max_iter=10)
    gender_classifier = gender_classifier.train(gender_feats, max_iter=10)
    return tweet_predict(test_l, classifier=True, classifiers=[age_classifier, gender_classifier])

