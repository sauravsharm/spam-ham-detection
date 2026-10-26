
from pathlib import Path

import joblib
import streamlit as st


# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Spam & Ham Detector",
    page_icon="📧",
    layout="centered",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "spam_ham_model_bundle.joblib"


# --------------------------------------------------
# 2. Load the trained model
# --------------------------------------------------
@st.cache_resource
def load_model_bundle():
    """Load the saved classifier and TF-IDF vectorizer."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 3. User interface
# --------------------------------------------------
st.title("📧 Intelligent Spam & Ham Detector")
st.write(
    "Enter an email subject and message to classify "
    "it as spam or legitimate email (ham)."
)

subject = st.text_input(
    "Email subject",
    placeholder="Example: Congratulations! You won a prize",
)

message = st.text_area(
    "Email message",
    height=200,
    placeholder="Paste the email content here...",
)

analyze = st.button(
    "Analyze Email",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# 4. Predict the email class
# --------------------------------------------------
if analyze:
    combined_text = f"{subject.strip()} {message.strip()}".strip()

    if not combined_text:
        st.warning("Please enter a subject or email message.")
    else:
        try:
            bundle = load_model_bundle()
            classifier = bundle["model"]
            vectorizer = bundle["vectorizer"]

            features = vectorizer.transform([combined_text])
            prediction = int(classifier.predict(features)[0])
            decision_score = float(
                classifier.decision_function(features)[0]
            )

            if prediction == 1:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: HAM (Legitimate Email)")

            st.metric(
                "SVM decision score",
                f"{decision_score:.4f}",
            )

            with st.expander("How to interpret this result"):
                st.write(
                    "Positive decision scores indicate the spam class; "
                    "negative scores indicate the ham class."
                )
                st.write(
                    "The decision score is not a probability or a "
                    "guaranteed measure of prediction confidence."
                )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "Machine-learning demonstration. Review important emails manually; "
    "the classifier can make mistakes."
)
