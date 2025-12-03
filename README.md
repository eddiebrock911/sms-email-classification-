# Spam Detector Dashboard

**A Streamlit-based web app for SMS/Message spam detection with UI, EDA, model training, and evaluation pipelines.**

---

## Live Demo [Go Live](https://antispamkit.onrender.com/)

## Overview

This project is an all-in-one Spam Detector Dashboard. It supports single-message prediction, CSV-based bulk prediction, prediction history, model performance analysis (ROC/PR, threshold optimization), word clouds, and training with TF-IDF and multiple classifiers along with model export.

## Key Features

* Single message prediction with spam probability gauge.
* Bulk prediction from CSV with downloadable report.
* Prediction history with export option.
* Model evaluation: confusion matrix, ROC/PR curves, F1 vs threshold, and best threshold selection.
* WordCloud visualization for spam vs ham.
* Train & Compare pipelines using Logistic Regression, Multinomial Naive Bayes, Linear SVC, and Random Forest.
* Automatic saving of the best model as `best_pipeline.pkl`, `vectorizer.pkl`, and `model.pkl`.
* Automated EDA option using Sweetviz with fallback.

## Requirements

* Python 3.8+ (Python 3.12 is also supported)
* Core dependencies:

  * streamlit
  * scikit-learn
  * pandas
  * numpy
  * joblib
  * matplotlib
  * seaborn
  * plotly
  * wordcloud
  * sweetviz (optional, for EDA)

> CMD run `pip install -r smsrequirements.txt`.

## How to Run

1. Clone the repository or place the project files locally.
2. Create and activate a virtual environment (optional).
3. Install dependencies:

```bash
pip install -r smsrequirements.txt
# or
pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn plotly wordcloud sweetviz
```

4. Run the app:

```bash
streamlit run sms_app2.py
```

5. Open in your browser: `http://localhost:8501`

## Important Files and Outputs

* `sms_app2.py` — Main Streamlit application.
* `model.pkl` — Trained classifier (optional).
* `vectorizer.pkl` — TF-IDF vectorizer (optional).
* `best_pipeline.pkl` — Complete pipeline with vectorizer + classifier (generated from Train & Compare).

> If `model.pkl` and `vectorizer.pkl` are not present, single and bulk prediction features that depend on a trained model will not work. In that case, train a model using the "Train & Compare Models" tab.

## CSV Input Format

* For bulk prediction, the CSV file must contain a `message` column.
* For model performance, WordCloud, and training, the CSV must contain both `message` and `label` columns (`ham` / `spam`).

## Training Notes

* The Train & Compare section uses TF-IDF with multiple classifiers.
* The best model is saved as a full pipeline and also extracted into `vectorizer.pkl` and `model.pkl`.
* Cross-validation and test-set evaluation are performed with interactive plots.

## Troubleshooting

* If `joblib.load("model.pkl")` or `vectorizer.pkl` is missing, the app will show a warning.
* If `get_feature_names_out` is unavailable for the TF-IDF vectorizer, feature importance display may fail.
* LinearSVC does not provide probabilities. The app scales the `decision_function` output to show a probability-like score.

## Future Improvements

* Add user authentication.
* Add database support (SQLite/PostgreSQL) for persistent history.
* Integrate advanced NLP embeddings from Hugging Face transformers.
* Improve mobile responsiveness and performance.

## License

You may apply any license as needed. MIT License is recommended.

---



