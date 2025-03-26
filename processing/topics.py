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

"""
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
"""


def topic_modeling(texts, num_topics=1):
    """Enhanced topic modeling with input validation and error handling"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    import numpy as np

    # 1. Input validation and preprocessing
    if not texts or not any(isinstance(t, str) and t.strip() for t in texts):
        return ["No valid text input"] * max(num_topics, 1)

    # Filter out empty/short texts and convert to string
    processed_texts = [str(t).strip() for t in texts if t and len(str(t).split()) >= 3]

    if not processed_texts:
        return ["Insufficient text data"] * max(num_topics, 1)

    # 2. More lenient vectorizer configuration
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,  # Ignore terms that appear in <2 docs
        token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
    )

    try:
        X = vectorizer.fit_transform(processed_texts)

        # Handle case where no features survived filtering
        if X.shape[1] == 0:
            return ["No meaningful terms found"] * max(num_topics, 1)

        # 3. Topic modeling
        lda = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            learning_method='batch',
            max_iter=10
        )
        lda.fit(X)

        # 4. Extract and return topics
        terms = vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_terms = [terms[i] for i in topic.argsort()[:-6:-1]]
            topics.append(", ".join(top_terms))

        return topics

    except Exception as e:
        # Fallback return if something goes wrong
        return [f"Topic modeling error: {str(e)}"] * max(num_topics, 1)

