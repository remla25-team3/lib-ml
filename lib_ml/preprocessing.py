import re
import string

from typing import List


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 'wordnet' not found. Downloading...")
        nltk.download('wordnet', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("NLTK 'stopwords' not found. Downloading...")
        nltk.download('stopwords', quiet=True)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' not found. Downloading...")
        nltk.download('punkt', quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

except ImportError:
    lemmatizer = None
    stop_words = set()
    word_tokenize = lambda x: x.split()

def lowercase_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.lower()

def remove_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_numbers(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r'\d+', '', text)

def remove_extra_whitespace(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())

def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            return word_tokenize(text)
        except Exception as download_error:
            print(f"Failed to download 'punkt': {download_error}")
            return text.split()
    except Exception as e:
        print(f"Error tokenizing text: {text}\nError: {e}")
        return text.split()

def remove_stopwords(tokens: List[str]) -> List[str]:
    """Removes common English stopwords."""
    if not stop_words:
        return tokens
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    if not lemmatizer:
        return tokens
    return [lemmatizer.lemmatize(word) for word in tokens]

# --- Main Preprocessing Function ---

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    try:
        text = lowercase_text(text)
        text = remove_punctuation(text)
        text = remove_numbers(text)
        text = remove_extra_whitespace(text)
        tokens = tokenize_text(text)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize_tokens(tokens)
        return " ".join(tokens)

    except Exception as e:
        print(f"Error processing text: {text}\nError: {e}")
        return ""


if __name__ == '__main__':
    sample_review = "The food at 'The Grand Place!' was AMAZINGLY good, maybe 5 stars?? Loved it! Cost 25 dollars."
    processed_review = preprocess_text(sample_review)
    print(f"Original: '{sample_review}'")
    print(f"Processed: '{processed_review}'")
