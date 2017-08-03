import json
# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

dataset_file = open("dataset.json")

dataset = json.loads(dataset_file.readline())
review_texts = []
for review in dataset:
	review_texts.append(review[0])
# print review_texts[5]


# categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
# newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
# vectorizer = TfidfVectorizer()
# vectors = vectorizer.fit_transform(newsgroups_train.data)
# print(vectors.shape)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(review_texts)
print(vectors.shape)

svd = TruncatedSVD(100) 
#For LSA, a value of 100 is recommended  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD

normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

vectors = lsa.fit_transform(vectors)
print(vectors.shape)

