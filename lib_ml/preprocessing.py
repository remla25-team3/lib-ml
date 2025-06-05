import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


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


def preprocess(df: pd.DataFrame, inference: bool = False) -> Tuple[List[str], List[str]]:
    """
    Preprocess restaurant reviews from a DataFrame by applying text cleaning,
    lowercasing, stopword removal (excluding "not"), stemming, and deduplication.

    Args:
        df (pd.DataFrame): DataFrame containing at least a 'Review' column and, unless `inference=True`, a second column with labels.
        inference (bool): If True, skips label extraction and returns only the processed text.

    Returns:
        Tuple[List[str], List[str]]: A tuple where the first element is a list of unique cleaned reviews,
                                     and the second is the corresponding list of labels (empty if inference=True).

    Raises:
        ValueError: If the 'Review' column is missing.

    Notes:
        - Duplicate or non-alphabetic-only reviews are removed after preprocessing.
        - The function uses a default English stopword list with "not" retained for sentiment purposes.
    """

    # return corpus
    if 'Review' not in df.columns:
        raise ValueError("DataFrame must contain a 'Review' column")
    stop_words = _prepare_stopwords()
    ps = PorterStemmer()
    seen = set()
    pattern = re.compile(r'^[a-z ]+$')

    df = df.copy()
    df['Review'] = df['Review'].astype(str)

    corpus = []
    labels = []

    for _, row in df.iterrows():
        review = row['Review']
        label = row.iloc[1] if not inference else None

        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # Convert to lowercase
        review = review.lower()
        # Split into words
        words = review.split()
        # Remove stopwords and apply stemming
        processed_words = [ps.stem(word) for word in words if word not in stop_words]
        # Join back into a string
        processed_review = ' '.join(processed_words)

        # Deduplicate based on processed form
        if processed_review and pattern.fullmatch(processed_review) and processed_review not in seen:
            seen.add(processed_review)
            corpus.append(processed_review)
            if not inference:
                labels.append(label)

    return (corpus, labels) if not inference else (corpus, [])