import re
import emoji
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def remove_email_headers_and_footers(text: str) -> str:
    lines = text.split('\n')
    stripped_lines = [line.strip() for line in lines]
    if "" in stripped_lines:
        first_blank_index = stripped_lines.index("")
        content = "\n".join(lines[first_blank_index + 1:]).strip()
    else:
        content = text
    signature_markers = ('sincerely,', 'regards,', 'best regards,', 'thanks,', 'thank you,')
    final_lines = []
    for line in content.split('\n'):
        if any(line.lower().startswith(marker) for marker in signature_markers):
            break
        final_lines.append(line)
    return "\n".join(final_lines).strip()

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