import streamlit as st
from transformers import pipeline as hf_pipeline

@st.cache_resource
def get_zero_shot_classifier():
    return hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@st.cache_resource
def get_abstractive_summarizer():
    return hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", revision="a4f8f3e")

@st.cache_resource
def get_sentiment_pipeline():
    return hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def load_paraphrase_model():
    from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline as hf_pipeline
    tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    paraphrase_pipeline = hf_pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
    )
    return paraphrase_pipeline
