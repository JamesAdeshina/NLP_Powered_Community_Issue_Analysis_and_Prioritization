import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
import logging


# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)




@st.cache_resource
def get_zero_shot_classifier():
    logger.info("Loading Zero-Shot Classifier model")
    try:
        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            tokenizer="facebook/bart-large-mnli",
            framework="pt"
        )
        logger.info("Zero-Shot Classifier model loaded successfully")
        return classifier
    except Exception as e:
        logger.error(f"Error loading Zero-Shot Classifier: {e}")
        raise




@st.cache_resource
def get_abstractive_summarizer():
    logger.info("Loading Abstractive Summarizer model")
    try:
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            revision="a4f8f3e"
        )
        logger.info("Abstractive Summarizer model loaded successfully")
        return summarizer
    except Exception as e:
        logger.error(f"Error loading Abstractive Summarizer: {e}")
        raise


@st.cache_resource
def get_sentiment_pipeline():
    logger.info("Loading Sentiment Analysis model")
    try:
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None
        )
        logger.info("Sentiment Analysis model loaded successfully")
        return sentiment_analyzer
    except Exception as e:
        logger.error(f"Error loading Sentiment Analysis model: {e}")
        raise



@st.cache_resource
def get_qa_pipeline():
    logger.info("Loading Question Answering model")
    try:
        qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            tokenizer="deepset/roberta-base-squad2"
        )
        logger.info("Question Answering model loaded successfully")
        return qa_pipeline
    except Exception as e:
        logger.error(f"Error loading Question Answering model: {e}")
        raise




@st.cache_resource
def get_embeddings_model():
    logger.info("Loading Sentence Embeddings model")
    try:
        embeddings_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("Sentence Embeddings model loaded successfully")
        return embeddings_model
    except Exception as e:
        logger.error(f"Error loading Sentence Embeddings model: {e}")
        raise




@st.cache_resource
def get_ner_pipeline():
    logger.info("Loading Named Entity Recognition model")
    try:
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        logger.info("Named Entity Recognition model loaded successfully")
        return ner_pipeline
    except Exception as e:
        logger.error(f"Error loading Named Entity Recognition model: {e}")
        raise





@st.cache_resource
def load_paraphrase_model():
    logger.info("Loading Paraphrase Generation model")
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
        model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
        paraphrase_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )
        logger.info("Paraphrase Generation model loaded successfully")
        return paraphrase_pipeline
    except Exception as e:
        logger.error(f"Error loading Paraphrase Generation model: {e}")
        raise