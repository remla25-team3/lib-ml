import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from typing import List


def _prepare_stopwords() -> set:
    """
    Prepare the stopwords for the preprocessing.
    """
    try:
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
        stop_words.discard('not')
        return stop_words
    except Exception as e:
        print(f"Error preparing stopwords: {e}")
        return set()


def preprocess(df: pd.DataFrame) -> List[str]:
    """
    Process reviews from a DataFrame through text preprocessing pipeline.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'Review' column
    
    Returns:
        List[str]: List of processed review texts
    
    Note:
        The function uses a default set of English stopwords (with "not" excluded) prepared by the `_prepare_stopwords` function.
    """
    if 'Review' not in df.columns:
        raise ValueError("DataFrame must contain a 'Review' column")
        
    stop_words = _prepare_stopwords()
    corpus = []
    ps = PorterStemmer()
    
    for review in df['Review']:
        # Remove non-alphabetic characters
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # Convert to lowercase
        review = review.lower()
        # Split into words
        words = review.split()
        # Remove stopwords and apply stemming
        processed_words = [ps.stem(word) for word in words if word not in stop_words]
        # Join back into a string
        processed_review = ' '.join(processed_words)
        corpus.append(processed_review)
    
    return corpus
