import re
import emoji
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer







def remove_emojis(text):
    return emoji.replace_emoji(text, replace="")

def expand_contractions(text):
    return contractions.fix(text)

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def remove_mentions_hashtags(text):
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    return text

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def normalize_repeated_chars(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_lower(text):
    if not text or not isinstance(text, str):
        return []
    try:
        return [word.lower() for word in word_tokenize(text)]
    except:
        return text.lower().split()

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def comprehensive_text_preprocessing(text, use_lemmatization=True):
    text = remove_emojis(text)
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = normalize_repeated_chars(text)
    text = remove_extra_whitespace(text)
    tokens = tokenize_and_lower(text)
    tokens = remove_stopwords(tokens)
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    return ' '.join(tokens)


def extract_locations(text):
    """
    Extract potential locations from text using simple pattern matching
    """
    # UK postcode pattern (simplified)
    postcode_pattern = r'[A-Z]{1,2}[0-9][A-Z0-9]? [0-9][A-Z]{2}'
    # Common address patterns
    address_pattern = r'\d+\s+[\w\s]+(?:street|road|avenue|lane|drive|way|close|circus)\b'

    postcodes = re.findall(postcode_pattern, text, re.IGNORECASE)
    addresses = re.findall(address_pattern, text, re.IGNORECASE)

    # Return the first found location or empty string
    if postcodes:
        return postcodes[0]
    elif addresses:
        return addresses[0]
    return ""