# lib-ml

A lightweight and reusable Python module for preprocessing restaurant reviews in sentiment analysis tasks.  
This module was developed specifically to support the [Restaurant Sentiment Analysis](https://github.com/proksch/restaurant-sentiment) project used in the TU Delft CS4295 Release Engineering for Machine Learning Applications course.

## Features

- Preprocesses raw text by:
  - Removing non-alphabetic characters
  - Converting to lowercase
  - Removing English stopwords (with "not" retained for sentiment relevance)
  - Applying Porter stemming
  - Removing duplicate processed reviews
- Can operate in both training and inference modes
- Handles malformed or empty entries gracefully
- Automatically downloads required NLTK resources on first use

## Installation

Install the required dependencies:

```
pip install -r requirements.txt
```

## Usage

### Training mode (text + labels) example:
> **Note**: if you want to use lib-ml, write in your requirements.txt: 
> ```
> lib-ml-remla-team3 @ git+https://github.com/remla25-team3/lib-ml@v0.5.0
> ```
> (version may need to be adjusted)

```
import pandas as pd
from lib_ml.preprocessing import preprocess  

df = pd.DataFrame({
    'Review': [
        "This place is awesome!",
        "Awful food... not coming back.",
        "Great service! Not bad at all.",
        "AWFUL FOOD... NOT coming back!!!"
    ],
    'Liked': [1, 0, 1, 0]
})

corpus, labels = preprocess(df)
print(corpus)
print(labels)
```

### Inference mode (text only):

```
df = pd.DataFrame({'Review': ["Food was okay.", "Absolutely terrible experience."]})
corpus, _ = preprocess(df, inference=True)
print(corpus)
```

## Function Signature

```
def preprocess(df: pd.DataFrame, inference: bool = False) -> Tuple[List[str], List[str]]:
```

- `df`: DataFrame containing at least a 'Review' column. If `inference=False`, it should also contain a second column for labels.
- `inference`: If True, skips label extraction and only returns processed text.

Returns:
- A tuple of (`processed_reviews`, `labels`) or (`processed_reviews`, `[]`) if in inference mode.

## Project Structure Example

```
lib-ml/
|
├── preprocessing.py        # Contains the preprocess function and stopword preparation
├── __init__.py             # Optionally exposes preprocess function
├── README.md               # This file
```

## Notes

- If the 'Review' column is missing, a ValueError is raised.
- Duplicate reviews are removed after preprocessing (e.g., "Awful food" and "AWFUL FOOD" are treated as duplicates).
- Only alphabetic content (after preprocessing) is retained; reviews reduced to empty strings or punctuation are dropped.

## Frequently Asked Questions

Q: Why is "not" kept in the stopword list?  
A: "Not" carries important sentiment polarity and is essential for distinguishing negated expressions like "not bad" or "not good".

Q: Why remove duplicates?  
A: Removing post-processed duplicates avoids over-representing certain opinions and improves model generalization.

Q: Does this work for non-English text?  
A: No. It uses NLTK's English stopwords and the Porter stemmer, which are English-specific.
