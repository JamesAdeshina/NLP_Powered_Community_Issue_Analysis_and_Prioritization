from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np


def cluster_documents(texts, n_clusters=3):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    return kmeans.labels_


def get_top_terms_per_cluster(texts, labels, n_terms=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()

    top_terms = {}
    for cluster_id in np.unique(labels):
        cluster_mask = (labels == cluster_id)
        cluster_docs = X[cluster_mask]
        centroid = np.mean(cluster_docs, axis=0).A1
        top_indices = centroid.argsort()[::-1][:n_terms]
        top_terms[cluster_id] = [terms[i] for i in top_indices]

    return top_terms