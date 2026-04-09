import streamlit as st
import pickle
import re

# =====================
# LOAD MODEL
# =====================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =====================
# CLEAN FUNCTION
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# =====================
# PAGE SETTINGS
# =====================
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")

# =====================
# CUSTOM CSS (DARK + ANIMATION)
# =====================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e1e2f, #2b2b4b);
    color: white;
}

h1 {
    text-align: center;
    color: #00c3ff;
    animation: fadeIn 2s ease-in-out;
}

.stTextArea textarea {
    background-color: #2b2b4b;
    color: white;
    border-radius: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #00c3ff, #0072ff);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.result {
    text-align: center;
    font-size: 22px;
    padding: 10px;
    margin-top: 15px;
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
</style>
""", unsafe_allow_html=True)

# =====================
# UI
# =====================
st.markdown("<h1>🧠 Fake News Detector</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align:center;'>AI model to detect Real vs Fake News</p>", unsafe_allow_html=True)

st.write("📊 Model Accuracy: 98.5%")

st.markdown("### ℹ️ About")
st.write("This app uses Machine Learning (TF-IDF + Logistic Regression) to classify news as Real or Fake.")

# Input
input_text = st.text_area("📝 Enter News Text")

# Prediction
if st.button("🔍 Analyze News"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = clean_text(input_text)
        vec = vectorizer.transform([cleaned])
        result = model.predict(vec)

        if result[0] == 1:
            st.markdown("<div class='result' style='color:lightgreen;'>✅ Real News</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='result' style='color:red;'>❌ Fake News</div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built with ❤️ using ML</p>", unsafe_allow_html=True)