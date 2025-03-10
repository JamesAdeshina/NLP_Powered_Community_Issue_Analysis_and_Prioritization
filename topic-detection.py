from transformers import pipeline
from keybert import KeyBERT

# Load models
kw_model = KeyBERT()  # For keyword extraction
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")  # For topic classification

# Sample letter
sample_text = """Hi, I'm writing to suggest some skill workshops at the community hall on King's Road in Bristol. 
These could be small classes on everyday skills like cooking, computer use, or crafts. 
I believe this would help people learn useful things in a fun, relaxed way. 
It could be a great chance for neighbours to come together and share what they know. 
I hope the council can help make these workshops happen soon. Thanks a lot. Best, Sam"""

# Step 1: Extract candidate topics (keywords)
keywords = kw_model.extract_keywords(sample_text, keyphrase_ngram_range=(1,2), stop_words='english', top_n=5)
candidate_topics = [kw[0] for kw in keywords]  # Extract just the words

# Step 2: Use Zero-Shot Classification to find the most relevant topic
result = classifier(sample_text, candidate_topics, multi_label=False)
predicted_topic = result["labels"][0]

# Output
print(f"Predicted Topic: {predicted_topic}")
