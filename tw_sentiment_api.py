import os
from typing import List, Dict, Tuple

# Environment configuration (must be set before importing HF or TF internals)
# Suppress the symlink warning on Windows and the TF oneDNN info lines.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
    
# Model repository
_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# Module-level cache for model and tokenizer so they are loaded once per process
_MODEL = None
_TOKENIZER = None

LABELS = ['Negative', 'Neutral', 'Positive']


def _load_model():
    """Lazy-load the model and tokenizer into module-level variables."""
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        try:
            _MODEL = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
            _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME)
        except Exception as e:
            print("Error while loading the model/tokenizer:\n", e)
            print("If you see a message about 'hf_xet' or Xet Storage, consider installing it:")
            print("    pip install 'huggingface_hub[hf_xet]' or pip install hf_xet")
            raise


def _preprocess(text: str) -> str:
    """Simple preprocessing used in many Twitter sentiment examples: replace mentions and links."""
    words = []
    for word in text.split(' '):
        if word.startswith('@') and len(word) > 1:
            words.append('@user')
        elif word.startswith('http'):
            words.append('http')
        else:
            words.append(word)
    return ' '.join(words)


def analyze_sentiment(text: str) -> List[Dict[str, float]]:
    """Analyze the sentiment of `text` and return a list of label/score dicts.

    Returns:
        [{'label': 'Negative', 'score': 0.01}, ...]
    """
    _load_model()
    assert _MODEL is not None and _TOKENIZER is not None

    proc = _preprocess(text)
    encoded = _TOKENIZER(proc, return_tensors='pt')
    output = _MODEL(**encoded)
    scores = output[0][0].detach().numpy()
    probs = softmax(scores)

    return [{'label': LABELS[i], 'score': float(probs[i])} for i in range(len(LABELS))]


def top_label(text: str) -> Tuple[str, float]:
    """Return the top label and its score for the given text."""
    results = analyze_sentiment(text)
    best = max(results, key=lambda x: x['score'])
    return best['label'], best['score']


if __name__ == '__main__':
    demo = "Great content! subscribed 😉"
    print('Input:', demo)
    res = analyze_sentiment(demo)
    for r in res:
        print(f"{r['label']}: {r['score']:.6f}")
    top, score = top_label(demo)
    print('\nPrediction:', top, f'({score:.3f}')
