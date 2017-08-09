import string
import collections
import nltk
#nltk.download()
from nltk import word_tokenize

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from textblob import TextBlob
import json
import re
from gensim.summarization import keywords
from gensim.summarization import summarize
from operator import itemgetter
import csv


def word_count(words, n):
	freq_dist = {}
	for word in words:
		if word in stopwords.words('english'):
			continue
		if word not in freq_dist:
			freq_dist[word] = 1
		else:
			freq_dist[word] = freq_dist[word] + 1
	return 	sorted(freq_dist.items(), key=itemgetter(1))

def process_text(text, stem=True):
	""" Tokenize text and stem words removing punctuation """
	# text = text.translate(None, string.punctuation)
	tokens = word_tokenize(text)
	if stem:
		stemmer = PorterStemmer()
		tokens = [stemmer.stem(t) for t in tokens]
	return tokens

def cluster_texts(texts, clusters=3):
	""" Transform texts to Tf-Idf coordinates and cluster texts using K-Means """
	vectorizer = TfidfVectorizer(tokenizer=process_text,
	stop_words='english',
	max_df=0.5,
	min_df=0.1,
	lowercase=True)
	tfidf_model = vectorizer.fit_transform(texts)

	# svd = TruncatedSVD(100) 
	# #For LSA, a value of 100 is recommended  http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html#sklearn.decomposition.TruncatedSVD
	# normalizer = Normalizer(copy=False)
	# lsa = make_pipeline(svd, normalizer)
	# tfidf_model = lsa.fit_transform(tfidf_model)
	#print(tfidf_model.shape)

	
	km_model = KMeans(n_clusters=clusters)
	km_model.fit(tfidf_model)
	clustering = collections.defaultdict(list)
	for idx, label in enumerate(km_model.labels_):
		clustering[label].append(idx)
	return clustering

if __name__ == "__main__":
	dataset_size = 999
	dataset_file = open("dataset.json")

	dataset = json.loads(dataset_file.readline())
	review_texts = []
	for review in dataset:
		if int(review[1]) > 3:
			continue
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
		
		if len(review_texts) > dataset_size:
			break

	print len(review_texts)

	k = 29

	clusters = dict(cluster_texts(review_texts,k))
	problem_list = []
	for cluster_no in range(0,k):
		problems_per_cluster = ['Cluster %d'%cluster_no]
		text = ""
		# print "CLUSTER %d" % cluster_no,
		for review_number in clusters[cluster_no]:
			text = text + " " + "".join(review_texts[review_number])
			# print text
		noun_phrases = TextBlob(text).noun_phrases
		phrase_count = {}
		for phrase in set(noun_phrases):
			if phrase in stopwords.words('english') or phrase in ['came','told','dont','outside','okay','ok','oh','really', 'never','everyone','went','sat','well','definitely']:
				continue
			sentiment = TextBlob(phrase).sentiment
			if sentiment.polarity > 0.5 or sentiment.subjectivity > 0.5:
				continue
			count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(phrase), text))
			phrase_count[phrase] = count
		insights = sorted(phrase_count.items(), key=itemgetter(1), reverse=True)[:20]
		for insight in insights:
			problems_per_cluster.append(insight[0])
		problem_list.append(problems_per_cluster)

	csv_file = csv.writer(open('data_size_%d_clusters_%d.csv'%(dataset_size,k),'w'))
	for row in zip(*problem_list):
		csv_file.writerow(row)
		# for phrase in row:
			# print phrase, "\t\t\t\t\t",
		# print 
