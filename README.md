🔎 Twitter Sentiment Analysis App
A Streamlit-based web application that performs sentiment analysis on tweets or short text using a pretrained Transformer model from Hugging Face.

The system classifies input text into:
Positive 😊
Neutral 😐
Negative 😠

🚀 Features
Interactive Streamlit UI with custom styling and background
Real-time sentiment prediction
Confidence score breakdown using progress bars
Emoji-based intuitive results
Lightweight preprocessing for Twitter-specific text
Cached model loading for performance optimization

🧠 Model Details
This project uses:
Model: cardiffnlp/twitter-roberta-base-sentiment
Framework: Hugging Face Transformers
Output: Probability distribution across:
Negative
Neutral
Positive

Implementation:
📂 Project Structure
.
├── main.py                  # Streamlit frontend UI
├── tw_sentiment_api.py      # Core sentiment analysis logic
├── tw-sentiment.py          # Standalone script (testing/demo)
├── requirements.txt         # Dependencies
├── sentiment.jpg            # Background image

⚙️ Installation
1. Clone the repository
git clone <your-repo-url>
cd <repo-folder>
2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
3. Install dependencies
pip install -r requirements.txt

Dependencies:
▶️ Running the App
streamlit run main.py

Main UI implementation:
🔍 How It Works
1. Preprocessin
Replaces:
@username → @user
URLs → http

2. Tokenization
Uses AutoTokenizer from Hugging Face

3. Inference
Model outputs logits
Converted to probabilities via softmax

4. Result Formatting
Returns structured output:
[
  {'label': 'Positive', 'score': 0.98},
  {'label': 'Neutral', 'score': 0.01},
  {'label': 'Negative', 'score': 0.01}
]

📊 Example
Input:
I love this product!
Output:
😊 Positive (98%)
Neutral (1%)
Negative (1%)
