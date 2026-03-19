import os

# Silence huggingface hub symlink warning on Windows where symlinks may be unsupported
# and disable oneDNN custom ops messages from TensorFlow for cleaner output.
# These must be set before importing transformers / tensorflow-related packages.
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy

# tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
tweet = 'Great content! subscribed 😉'

# precprcess tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

try:
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
except Exception as e:
    # Common non-fatal issues on Windows: missing hf_xet (Xet storage) or symlink warnings.
    print("Error while loading the model/tokenizer:\n", e)
    print("If you see a message about 'hf_xet' or Xet Storage, consider installing it: ")
    print("    pip install huggingface_hub[hf_xet]")
    print("Or suppress the symlink warning by enabling Developer Mode or running Python as Admin.")
    raise

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)


"""for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)"""

a = scores[0]
b = scores[1]
c = scores[2]

def greatest_of_three(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c

greatest = greatest_of_three(a, b, c)
if greatest == a:
    print("Negative")
elif greatest == b:
    print("Neutral")
else:
    print("Positive")
    

