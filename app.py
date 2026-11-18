print("Welcome to Fake News Detection System!")
print("My AI project is starting.")


import streamlit as st
import pickle

# Load trained model
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("AI Fake News Detection System")

st.write("Enter a news statement to check its prediction.")

news = st.text_area("Enter News Here:")

if st.button("Check News"):
    if news.strip():
        prediction = model.predict([news])[0]

        if prediction == "REAL":
            st.success("Prediction: REAL NEWS")
        else:
            st.error("Prediction: FAKE NEWS")
    else:
       st.warning("Please enter some news.")

import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load AI model
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

# Application title
st.title("📰 AI Fake News Detection System")

st.write(
    "An Artificial Intelligence project using "
    "Machine Learning to classify news text."
)

st.divider()

# News input
st.subheader("Check Your News")

news = st.text_area(
    "Enter a news statement:",
    placeholder="Type or paste your news here...",
    height=150
)

# Prediction button
if st.button("🔍 Detect News", use_container_width=True):

    if news.strip():

        prediction = model.predict([news])[0]

        st.subheader("Prediction Result")

        if prediction == "REAL":
            st.success("Prediction: REAL NEWS")
        else:
            st.error("Prediction: FAKE NEWS")

        st.caption(
            "This is a model prediction, not a verified fact-check."
        )

        # Show model score
        probabilities = model.predict_proba([news])[0]
        score = max(probabilities) * 100

        st.write("Model score (not factual confidence):")
        st.progress(float(score / 100))
        st.write(f"{score:.2f}%")

    else:
        st.warning("Please enter a news statement.")

st.divider()

# About section
st.subheader("About This Project")

st.write("""
This project uses Machine Learning techniques
to classify news statements as REAL or FAKE.

Technologies Used:
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression
""")

st.info(
    "Note: This is an educational prototype. "
    "Its current small sample dataset is not sufficient "
    "for reliable real-world news detection."
)
