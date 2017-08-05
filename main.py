import json
import numpy as np
import re
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

dataset_file = open("dataset.json")

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
    text = re.sub(r'[^\w\s]',' ',text)	#remove punctuations
    text = re.sub(r'\w*\d\w*', '', text).strip()	#remove words with numbers in them
    text = re.sub(r"\s+", " ", text, flags=re.UNICODE)	#remove unicode white spaces
    review_texts.append(text)

    # if len(review_texts) > 50:
    # 	break
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

#removing stopwords
my_stop_words = Text.ENGLISH_STOP_WORDS.union(["book"])
vectorizer = TfidfVectorizer(stop_words=my_stop_words)



vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(review_texts)
terms = vectorizer.get_feature_names()
print(terms)
#print(vectors.shape)
"""
svd = TruncatedSVD(100) 
#For LSA, a value of 100 is recommended  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD

normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

vectors = lsa.fit_transform(vectors)
#print(vectors.shape)



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


print ("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
for i in range(true_k):
    print ("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind])
    print

"""
"""
print("done in %0.3fs" % (time() - t0))
print()

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(labels, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()


if not opts.use_hashing:
    print("Top terms per cluster:")

    if opts.n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
        
    """