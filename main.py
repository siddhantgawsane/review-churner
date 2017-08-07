import json
import numpy as np
import re
import nltk 
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction import text as Text
from sklearn.cluster import KMeans
from textblob import TextBlob

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

<<<<<<< HEAD
dataset_file = open("review.json")
=======
dataset_file = open("dataset.json")
>>>>>>> 0adcead350bcbd879677d4174d87f418eb996786

dataset = json.loads(dataset_file.readline())
review_texts = []
for review in dataset:
    blob = TextBlob(review[0])
    language = blob.detect_language()

    if (language!='en'):
        translation = blob.translate(to='en')
        text = str(translation)
    else:
        text = str(blob)

    text = re.sub(r"http\S+", "", text)	#remove urls
    text = re.sub(r'[^\w\s]','',text)	#remove punctuations
    text = re.sub(r'\w*\d\w*', '', text).strip()	#remove words with numbers in them
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)	#remove unicode white spaces
    review_texts.append(text)

    if len(review_texts) > 100:
    	break
# print (review_texts)
   
    #print (review[0])
    
	#review_texts.append(review[0])
    #blob = TextBlob(review)
#print (review_texts)


# categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(newsgroups_train.data)
# print(vectors.shape)
<<<<<<< HEAD
=======

#removing stopwords
# my_stop_words = Text.ENGLISH_STOP_WORDS.union(["book"])
# vectorizer = TfidfVectorizer(stop_words=my_stop_words, analyzer=stemmed_words)
vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

>>>>>>> 0adcead350bcbd879677d4174d87f418eb996786

#removing stopwords
# my_stop_words = Text.ENGLISH_STOP_WORDS.union(["book"])
# vectorizer = TfidfVectorizer(stop_words=my_stop_words, analyzer=stemmed_words)
vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

<<<<<<< HEAD
vectors = vectorizer.fit_transform(review_texts)
terms = vectorizer.get_feature_names()
print (terms)
=======
# vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(review_texts)
# terms = vectorizer.get_feature_names()
>>>>>>> 0adcead350bcbd879677d4174d87f418eb996786
# print(vectors[0][0])
# print(vectors.shape)
"""
svd = TruncatedSVD(100) 
#For LSA, a value of 100 is recommended  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
vectors = lsa.fit_transform(vectors)
#print(vectors.shape)
"""

num_clusters = 5

km = KMeans(n_clusters=num_clusters)

km.fit(vectors)

clusters = km.labels_.tolist()

print (clusters.count(0))
print (clusters.count(1))
print (clusters.count(2))
print (clusters.count(3))
print (clusters.count(4))
<<<<<<< HEAD
=======

"""
#USING ELBOW METHOD to find optimum cluster
from sklearn.cluster import KMeans
wcss = []
#we are trying to figure out the right number clusters 
for i in range (1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    kmeans.fit(vectors)  #fit fits Kmeans algorithm to your data
    #inertia- 
    wcss.append(kmeans.inertia_)
#plot elbow method graph
plt.plot(range(1,11),wcss)#wcss is x-axis and range is y axis
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


#Clustering
true_k = 10
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
#print("Clustering sparse data with %s" % model)
#t0 = time()
model.fit(vectors)
>>>>>>> 0adcead350bcbd879677d4174d87f418eb996786


from __future__ import print_function

print ("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print ("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind])
    print