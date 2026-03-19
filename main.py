import streamlit as st
from typing import List, Dict
import base64
import os

# --- Faking functions for this example ---
# (You can swap these back with your 'from tw_sentiment_api...' import)
@st.cache_data
def analyze_sentiment(text):
    """Faked API call"""
    return [{'label': 'POSITIVE', 'score': 0.98}, {'label': 'NEUTRAL', 'score': 0.01}, {'label': 'NEGATIVE', 'score': 0.01}]

@st.cache_data
def top_label(text):
    """Faked API call"""
    return "POSITIVE", 0.98
# --- End of fake functions ---


# --- Enhanced Page Config ---
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Function to encode image ---
@st.cache_data
def get_base64_of_bin_file(bin_file):
    """
    Reads a binary file and returns its Base64 encoded string.
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Set Background Image (The "Forceful" Version) ---
def set_bg_image(image_file):
    """
    Sets the background image for the Streamlit app.
    This version uses !important to override default Streamlit themes.
    """
    if not os.path.exists(image_file):
        st.error(f"Image file not found: {image_file}")
        return

    # Auto-detect file type
    file_extension = os.path.splitext(image_file)[1].lower()
    mime_type_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"
    }
    image_format = mime_type_map.get(file_extension)
    
    if image_format is None:
        st.error(f"Unsupported image format: {file_extension}")
        return

    base64_img = get_base64_of_bin_file(image_file)
    
    # --- This is the new, aggressive CSS ---
    page_bg_img = f"""
    <style>
    /* 1. Set the main background */
    [data-testid="stAppViewContainer"] > .main {{
        background-image: url("data:{image_format};base64,{base64_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* 2. Make all content containers transparent */
    [data-testid="stAppViewContainer"] > .main [data-testid="stBlockContainer"],
    [data-testid="stHeader"],
    [data-testid="stFooter"] {{
        background-color: transparent !important;
    }}

    /* 3. Ensure text is readable (force white) */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown p, .stWrite p, .stException p {{
        color: #FFFFFF !important;
    }}
    
    /* 4. Style the "Waiting for input" text */
    [data-testid="stText"] {{
         color: #DDDDDD !important; /* A light gray */
    }}

    /* 5. Style the text area */
    .stTextArea textarea {{
        background-color: rgba(30, 30, 40, 0.8) !important; /* Semi-transparent dark */
        color: #FFFFFF !important; /* White text */
    }}
    
    /* 6. Style the info/warning boxes */
    [data-testid="stInfo"], [data-testid="stWarning"] {{
        background-color: rgba(30, 30, 40, 0.8) !important;
    }}
    
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Apply the background ---
# IMPORTANT: Make sure 'sentiment.jpg' is the correct file name and path
set_bg_image("sentiment.jpg")


# --- Custom CSS Styling (Reduced) ---
# We moved most styles into the function above
st.markdown("""
    <style>
        .sentiment { 
            font-size: 22px; 
            font-weight: bold; 
            padding: 6px 10px; 
            background: #eaf5fb; /* This light bg is fine */
            border-radius: 8px; 
            color: #333; /* Dark text for a light box */
        }
        .footer { 
            font-size: 13px; 
            color: #EEE !important; /* Light footer text */
            padding: 16px 0 0 0; 
            text-align:center;
        }
        .emoji {font-size:26px; vertical-align:middle;}
    </style>
""", unsafe_allow_html=True)


# --- App Header ---
st.title("🔎 Twitter Sentiment Analysis")
st.write(
    """
    Analyze the sentiment of any tweet or short text with our ML-based model.
    Results show the prediction, score breakdown, and easy-to-understand visual cues.
    """
)

# --- Sentiment Input ---
text_input = st.text_area(
    "Enter Twitter text or any short sentence:",
    height=100,
    help="Paste or type here. Example: 'I love this!'"
)

# --- Main Analysis ---
if st.button("Analyze", use_container_width=True):
    if not text_input or text_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                results: List[Dict[str, float]] = analyze_sentiment(text_input)
            except Exception as e:
                st.error(f"Error running model: {e}")
                st.stop()

        # --- Sentiment Top Label & Icon Mapping ---
        top, score = top_label(text_input)
        sentiment_icon = {
            "POSITIVE": "😊",
            "NEGATIVE": "😠",
            "NEUTRAL": "😐"
        }
        top_emoji = sentiment_icon.get(top.upper(), "🤔")

        st.markdown(f'<div class="sentiment">{top_emoji} Prediction: <span>{top}</span> <span style="color:#2678d3;">({score:.2%})</span></div>',
                    unsafe_allow_html=True)

        # --- Score Breakdown Table ---
        st.subheader("Detailed Scores")
        col1, col2 = st.columns([2, 3])
        with col1:
            st.write("Sentiment Label")
            for item in results:
                label = item['label']
                emoji = sentiment_icon.get(label.upper(), "🤔")
                st.write(f"- {emoji} {label}")
        with col2:
            st.write("Confidence & Progress")
            for item in results:
                score = item['score']
                st.progress(int(round(score * 100)), text=f"{score:.2%}")

        # --- Insight Summary ---
        st.markdown("----")
        most_confident = max(results, key=lambda x: x['score'])
        lowest = min(results, key=lambda x: x['score'])
        st.info(
            f"👉 The model is most confident about '**{most_confident['label']}**' ({most_confident['score']:.2%}).\n"
            f"👉 The least likely sentiment is '**{lowest['label']}**' ({lowest['score']:.2%})."
        )

        # --- Footer ---
        st.markdown('<div class="footer">Thank you for using Twitter Sentiment Analysis! <br>Made with Streamlit, Python & HuggingFace 🤗</div>', unsafe_allow_html=True)

else:
    # Use st.text to apply the [data-testid="stText"] style
    st.text("👀 Waiting for your input — Try a tweet like 'So excited for this new phone!'")