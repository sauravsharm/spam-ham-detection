
# Intelligent Spam & Ham Detection

An NLP machine-learning project that classifies email text as spam or legitimate (ham) using TF-IDF features and a Linear Support Vector Machine.

## Features

- Email subject and message classification
- TF-IDF text representation
- Comparison of Multinomial Naive Bayes, Logistic Regression, and Linear SVM
- Validation and held-out test evaluation
- Basic robustness experiments
- Interactive Streamlit application

## Model Performance

The selected word-based TF-IDF + Linear SVM achieved the following results on the Enron-Spam dataset:

| Metric | Validation | Test |
|---|---:|---:|
| Accuracy | 99.34% | 99.18% |
| F1-score | 99.36% | 99.19% |

These results are dataset-specific and may not reflect performance on modern real-world email.

## Run Locally

1. Create and activate a Python virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

4. Open the local URL printed by Streamlit.

## Model Artifacts

The `models/` directory contains the serialized model bundle and its metadata.

## Limitations

- The classifier can produce false positives and false negatives.
- The decision score is not a probability.
- Robustness checks so far use a small set of manually written examples.
- Model performance should be independently validated on representative, newer email data.

## Tech Stack

Python, pandas, scikit-learn, TF-IDF, Linear SVM, joblib, Streamlit.
