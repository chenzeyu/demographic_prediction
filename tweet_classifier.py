from pymongo import *
import csv

file = open('train.csv', 'rb')
reader = csv.reader(file, 'excel')
users = {}
for row in reader:
    users[row[0]] = {'gender': row[1], 'age': row[2]}

client = MongoClient('localhost', 27017)
db = client.test
tweets = db.tweets.find()

cc = []
for tweet in tweets:
    cc.append(tweet)

print "hehe"