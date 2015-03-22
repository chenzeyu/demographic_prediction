from pymongo import *
import re

client = MongoClient('localhost', 27017)
db = client.test
tweets = db.tweets.find()

# Filename	WC	WPS	Qmarks	Unique	Dic	Sixltr	funct	pronoun	ppron	i	we	you	shehe	they	ipron	article	verb	auxverb	past	present	future	adverb	preps	conj	negate	quant	number	swear	social	family	friend	humans	affect	posemo	negemo	anx	anger	sad	cogmech	insight	cause	discrep	tentat	certain	inhib	incl	excl	percept	see	hear	feel	bio	body	health	sexual	ingest	relativ	motion	space	time	work	achieve	leisure	home	money	relig	death	assent	nonfl	filler


def remove_url(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=re.MULTILINE)


def remove_at_and_hash(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())


def set_up():
    for tweet in tweets:
        text = tweet['text']
        user = tweet['userId']
        text = remove_at_and_hash(remove_url(text))
        text_file = open("tweets/%s.txt" % user, "a")
        text_file.write("%s " % text)
        text_file.close()


