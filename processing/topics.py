from rake_nltk import Rake
from models.load_models import get_zero_shot_classifier
from config import CANDIDATE_LABELS_TOPIC
import streamlit as st


@st.cache_data
def dynamic_topic_label(keywords: str) -> str:
    classifier = get_zero_shot_classifier()
    result = classifier(keywords, CANDIDATE_LABELS_TOPIC)
    return result["labels"][0]


@st.cache_data
def compute_topic(text: str, top_n: int = 5) -> tuple[str, str]:
    rake_extractor = Rake()
    rake_extractor.extract_keywords_from_text(text)
    ranked_phrases = rake_extractor.get_ranked_phrases()
    top_terms = ranked_phrases[:top_n] if len(ranked_phrases) >= top_n else ranked_phrases
    keyword_str = ", ".join(top_terms)
    topic_label = dynamic_topic_label(keyword_str)
    return topic_label, keyword_str


def topic_modeling(texts, num_topics=1):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, learning_method='batch', max_iter=10)
    lda.fit(X)

    topics = []
    terms = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_terms = [terms[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(top_terms))

    return topics