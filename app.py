import streamlit as st
import pickle

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("""
<style>
.main {
    background-color: #1E1E1E;
    color: #FFFFFF;
}
.stTextArea textarea {
    background-color: #2D2D2D;
    color: white;
    font-size: 16px;
    border-radius: 10px;
    border: 1px solid #4F4F4F;
}
.stButton button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #FF6B6B;
    transform: scale(1.05);
}
.title-text {
    font-family: 'Inter', sans-serif;
    font-size: 40px;
    font-weight: 800;
    text-align: center;
    background: -webkit-linear-gradient(#FF4B4B, #FF8E53);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">Fake News Detector 📰</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0A0A0; font-size: 18px;'>Paste your news article below to check its authenticity.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        return None, None

model, vectorizer = load_model()

if model is None:
    st.error("Model not found! Please make sure model.pkl and vectorizer.pkl exist. Running train.py can generate them.")
else:
    user_input = st.text_area("News Article Text:", height=250, placeholder="Enter news text here...")
    if st.button("Detect Fake News 🚀"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text patterns..."):
                vec_input = vectorizer.transform([user_input])
                prediction = model.predict(vec_input)[0]
                
                st.markdown("---")
                if prediction.lower() == 'fake':
                    st.error("🚨 **FAKE NEWS DETECTED!** 🚨\n\nThis article is likely fabricated or highly misleading.")
                else:
                    st.success("✅ **REAL NEWS!** ✅\n\nThis article appears to be authentic.")