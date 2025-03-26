from config import CLASSIFICATION_LABELS
from models.load_models import get_zero_shot_classifier


def classify_document(text):
    classifier = get_zero_shot_classifier()
    result = classifier(text, CLASSIFICATION_LABELS)
    return result["labels"][0]


def unsupervised_classification(texts, num_clusters=2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_, vectorizer, kmeans


def dynamic_label_clusters(vectorizer, kmeans):
    from models.load_models import get_zero_shot_classifier
    from config import CLASSIFICATION_LABELS

    cluster_labels = {}
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    for i in range(kmeans.n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        keyword_str = ", ".join(top_terms)
        classifier = get_zero_shot_classifier()
        result = classifier(keyword_str, CLASSIFICATION_LABELS)
        cluster_labels[i] = result["labels"][0]

    return cluster_labels