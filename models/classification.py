from config import CLASSIFICATION_LABELS
from models.load_models import get_zero_shot_classifier
import logging

# Initialize logger
logger = logging.getLogger(__name__)


def classify_document(text):
    logger.info("Starting document classification")
    logger.debug(f"Document text: {text[:100]}...")  # Log a truncated version of the input text for readability
    try:
        classifier = get_zero_shot_classifier()
        result = classifier(text, CLASSIFICATION_LABELS)
        classification = result["labels"][0]
        logger.info(f"Document classified as: {classification}")
        return classification
    except Exception as e:
        logger.error(f"Error during document classification: {e}")
        raise


def unsupervised_classification(texts, num_clusters=2):
    logger.info(f"Starting unsupervised classification with {num_clusters} clusters")
    logger.debug(f"Number of texts to classify: {len(texts)}")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans

        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        logger.debug("TF-IDF matrix created")

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)
        logger.info("Clustering completed")

        return kmeans.labels_, vectorizer, kmeans
    except Exception as e:
        logger.error(f"Error during unsupervised classification: {e}")
        raise


def dynamic_label_clusters(vectorizer, kmeans):
    logger.info("Starting dynamic label clustering")
    try:
        from models.load_models import get_zero_shot_classifier
        from config import CLASSIFICATION_LABELS

        cluster_labels = {}
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        logger.debug("Cluster centroids and terms extracted")

        for i in range(kmeans.n_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]
            keyword_str = ", ".join(top_terms)
            logger.debug(f"Cluster {i} keywords: {keyword_str}")

            classifier = get_zero_shot_classifier()
            result = classifier(keyword_str, CLASSIFICATION_LABELS)
            cluster_labels[i] = result["labels"][0]
            logger.info(f"Cluster {i} labeled as: {cluster_labels[i]}")

        return cluster_labels
    except Exception as e:
        logger.error(f"Error during dynamic label clustering: {e}")
        raise
