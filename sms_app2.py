# =========================
# Spam Detector Dashboard (All-in-One)
# streamlit run sms_app2.py
# =========================

import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from wordcloud import WordCloud

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Spam Detector Dashboard", page_icon="📩", layout="wide")

# -------------------------
# Try to Load Pretrained Model/Vectorizer (optional)
# -------------------------
def safe_load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None

#MODEL_PATH = "model.pkl"
#VECT_PATH  = "vectorizer.pkl"

#model = safe_load(MODEL_PATH)
#vectorizer = safe_load(VECT_PATH)

import joblib
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------------
# Session State
# -------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------
# Helpers
# -------------------------
def get_proba(model, X):
    """
    Return probability for positive class if available.
    If only decision_function exists, scale to [0,1] via min-max on scores.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Min-max scale (avoid zero division)
        smin, smax = scores.min(), scores.max()
        if smax == smin:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - smin) / (smax - smin)
    else:
        # Fallback: use predictions as coarse probs
        return model.predict(X).astype(float)

def basic_text_features(series: pd.Series) -> pd.DataFrame:
    """Create light EDA features for correlation heatmap."""
    s = series.fillna("")
    df = pd.DataFrame({
        "char_len": s.str.len(),
        "word_count": s.str.split().map(len),
        "digit_count": s.str.count(r"\d"),
        "upper_ratio": s.apply(lambda x: sum(c.isupper() for c in x) / (len(x) if len(x) else 1)),
        "exclaim_count": s.str.count("!"),
        "link_count": s.str.count(r"http|www")
    })
    return df

def top_feature_importance(model, vectorizer, top_k=20):
    """
    Return DataFrame of top positive/negative features for linear models with coef_,
    or top features for tree models with feature_importances_.
    """
    if vectorizer is None or not hasattr(vectorizer, "get_feature_names_out"):
        return None

    feature_names = np.array(vectorizer.get_feature_names_out())

    if hasattr(model, "coef_"):  # linear models (LogReg, LinearSVC with calibrated or not)
        coefs = model.coef_.ravel()
        order_pos = np.argsort(coefs)[-top_k:][::-1]  # top positive
        order_neg = np.argsort(coefs)[:top_k]         # top negative
        df_pos = pd.DataFrame({"feature": feature_names[order_pos],
                               "weight": coefs[order_pos],
                               "direction": "spam ↗"})
        df_neg = pd.DataFrame({"feature": feature_names[order_neg],
                               "weight": coefs[order_neg],
                               "direction": "spam ↘ (ham ↗)"})
        return pd.concat([df_pos, df_neg], ignore_index=True)

    if hasattr(model, "feature_importances_"):  # trees
        importances = model.feature_importances_
        order = np.argsort(importances)[-top_k:][::-1]
        return pd.DataFrame({"feature": feature_names[order],
                             "weight": importances[order],
                             "direction": "importance"})

    return None

def plot_f1_vs_threshold(y_true, y_proba):
    thresholds = np.linspace(0, 1, 201)
    f1s = []
    for t in thresholds:
        preds_t = (y_proba >= t).astype(int)
        f1s.append(f1_score(y_true, preds_t))
    f1s = np.array(f1s)
    best_idx = int(f1s.argmax())
    best_t = float(thresholds[best_idx])
    best_f1 = float(f1s[best_idx])

    fig, ax = plt.subplots()
    ax.plot(thresholds, f1s, lw=2)
    ax.axvline(best_t, linestyle="--")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score")
    ax.set_title("F1-Score vs Threshold (best marked)")
    return fig, best_t, best_f1

def confusion_at_threshold(y_true, y_proba, t):
    preds_t = (y_proba >= t).astype(int)
    return confusion_matrix(y_true, preds_t), classification_report(y_true, preds_t, digits=3)

def df_download_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, filename, "text/csv")

def bytes_download_button(b: bytes, filename: str, label: str, mime: str = "application/octet-stream"):
    st.download_button(label, b, filename, mime)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("📩 Spam Detector Dashboard")
option = st.sidebar.radio(
    "Navigate",
    [
        "🔹 Single Message",
        "📂 Bulk Upload",
        "📜 History & Export",
        "📊 Model Performance",
        "☁ WordCloud",
        "🧪 Train & Compare Models",
        "📑 EDA Report"
    ]
)

# =====================
# 1) Single Message
# =====================
if option == "🔹 Single Message":
    st.title("🔹 Single Message Spam Detector")

    if model is None or vectorizer is None:
        st.warning("Pretrained model/vectorizer नहीं मिले (model.pkl / vectorizer.pkl). नीचे वाले tabs में Train & Compare से model बना सकते हो।")

    message = st.text_area("Enter a message:", height=140, placeholder="Type SMS/Email text here...")
    if st.button("Predict", type="primary"):
        if not message.strip():
            st.warning("⚠️ Please enter a message!")
        elif (model is None) or (vectorizer is None):
            st.error("❌ Prediction unavailable (missing model/vectorizer).")
        else:
            Xv = vectorizer.transform([message])
            pred = model.predict(Xv)[0]
            proba = get_proba(model, Xv)[0] * 100
            proba = round(float(proba), 2)

            label = "🚨 Spam" if pred == 1 else "✅ Not Spam"
            st.success(f"**Prediction:** {label}  •  **Spam Probability:** {proba}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba,
                title={'text': "Spam Probability (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': proba}
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.session_state["history"].append([message, label, proba])

# =====================
# 2) Bulk Upload
# =====================
elif option == "📂 Bulk Upload":
    st.title("📂 Bulk Upload Spam Detector")

    if model is None or vectorizer is None:
        st.warning("Pretrained model/vectorizer नहीं मिले (model.pkl / vectorizer.pkl).")

    up = st.file_uploader("Upload CSV with a 'message' column", type="csv")
    if up:
        df = pd.read_csv(up)
        if "message" not in df.columns:
            st.error("❌ CSV must contain 'message' column.")
        elif (model is None) or (vectorizer is None):
            st.error("❌ Prediction unavailable (missing model/vectorizer).")
        else:
            Xv = vectorizer.transform(df["message"].fillna(""))
            preds = model.predict(Xv)
            probs = get_proba(model, Xv) * 100

            out = df.copy()
            out["Prediction"] = np.where(preds == 1, "🚨 Spam", "✅ Not Spam")
            out["Spam Probability (%)"] = np.round(probs, 2)

            st.subheader("📊 Predictions")
            st.dataframe(out, use_container_width=True)
            df_download_button(out, "predictions.csv", "⬇ Download Predictions")

            for msg, lab, pr in zip(df["message"], out["Prediction"], out["Spam Probability (%)"]):
                st.session_state["history"].append([msg, lab, pr])

# =====================
# 3) History & Export
# =====================
elif option == "📜 History & Export":
    st.title("📜 Prediction History")
    if len(st.session_state["history"]) == 0:
        st.info("No history yet. Run predictions first.")
    else:
        hist = pd.DataFrame(st.session_state["history"], columns=["Message", "Prediction", "Spam Probability (%)"])
        st.dataframe(hist, use_container_width=True, height=400)
        df_download_button(hist, "history.csv", "⬇ Download History")

# =====================
# 4) Model Performance
# =====================
elif option == "📊 Model Performance":
    st.title("📊 Model Performance (with Threshold Optimization, ROC/PR, Feature Importance, Correlation)")

    if model is None or vectorizer is None:
        st.warning("Pretrained model/vectorizer नहीं मिले (model.pkl / vectorizer.pkl).")

    up = st.file_uploader("Upload CSV with 'label' (ham/spam) and 'message' columns", type="csv")
    if up:
        df = pd.read_csv(up)
        if not {"label", "message"}.issubset(df.columns):
            st.error("❌ CSV must contain 'label' and 'message'.")
        elif (model is None) or (vectorizer is None):
            st.error("❌ Evaluation unavailable (missing model/vectorizer).")
        else:
            df = df.copy()
            df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
            Xv = vectorizer.transform(df["message"].fillna(""))
            y = df["label_num"].values

            preds = model.predict(Xv)
            y_proba = get_proba(model, Xv)

            # Accuracy & Confusion Matrix
            acc = accuracy_score(y, preds)
            st.metric("Accuracy (threshold=0.5)", f"{acc:.3f}")

            cm = confusion_matrix(y, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix (0.5)")
            st.pyplot(fig)

            st.subheader("Classification Report (threshold=0.5)")
            st.text(classification_report(y, preds, digits=3))

            # ROC Curve
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0,1], [0,1], linestyle="--")
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve"); ax.legend(loc="lower right")
            st.pyplot(fig)

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, y_proba)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, lw=2)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            st.pyplot(fig)

            # F1 vs Threshold + Best Threshold
            st.subheader("F1-Score vs Threshold (Best threshold highlighted)")
            fig, best_t, best_f1 = plot_f1_vs_threshold(y, y_proba)
            st.pyplot(fig)
            st.success(f"Best Threshold: **{best_t:.3f}**  •  Best F1: **{best_f1:.3f}**")

            cm_best, report_best = confusion_at_threshold(y, y_proba, best_t)
            fig, ax = plt.subplots()
            sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix (@ threshold={best_t:.3f})")
            st.pyplot(fig)
            st.text("Classification Report (@ best threshold)")
            st.text(report_best)

            # Feature Importance
            st.subheader("Top Features (by model weights/importances)")
            imp_df = top_feature_importance(model, vectorizer, top_k=20)
            if imp_df is None or imp_df.empty:
                st.info("Feature importance unavailable (model/vectorizer may not expose weights).")
            else:
                st.dataframe(imp_df, use_container_width=True, height=420)
                # Simple bar plot (absolute weight)
                plot_df = imp_df.copy()
                plot_df["abs_w"] = plot_df["weight"].abs()
                plot_df = plot_df.sort_values("abs_w", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.barh(plot_df["feature"], plot_df["weight"])
                ax.invert_yaxis()
                ax.set_xlabel("Weight / Importance"); ax.set_title("Top Features")
                st.pyplot(fig)

            # Light EDA Correlation Heatmap on basic text features
            st.subheader("Correlation Heatmap (basic text features vs label)")
            feat_df = basic_text_features(df["message"])
            feat_df["label_num"] = y
            corr = feat_df.corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

# =====================
# 5) WordCloud
# =====================
elif option == "☁ WordCloud":
    st.title("☁ WordCloud Visualization")
    up = st.file_uploader("Upload CSV with 'label' (ham/spam) and 'message' columns", type="csv")
    if up:
        df = pd.read_csv(up)
        if not {"label", "message"}.issubset(df.columns):
            st.error("❌ CSV must have 'label' and 'message'.")
        else:
            spam_text = " ".join(df[df["label"] == "spam"]["message"].fillna(""))
            ham_text  = " ".join(df[df["label"] == "ham"]["message"].fillna(""))

            st.subheader("🚨 Spam Messages WordCloud")
            fig, ax = plt.subplots()
            ax.imshow(WordCloud(width=900, height=500, background_color="black", colormap="Reds").generate(spam_text), interpolation="bilinear")
            ax.axis("off"); st.pyplot(fig)

            st.subheader("✅ Ham Messages WordCloud")
            fig, ax = plt.subplots()
            ax.imshow(WordCloud(width=900, height=500, background_color="white", colormap="Greens").generate(ham_text), interpolation="bilinear")
            ax.axis("off"); st.pyplot(fig)

# =====================
# 6) Train & Compare Models
# =====================
elif option == "🧪 Train & Compare Models":
    st.title("🧪 Train & Compare Models (TF-IDF Pipelines)")

    up = st.file_uploader("Upload CSV with 'label' (ham/spam) and 'message' columns", type="csv")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_splits = st.slider("CV folds (StratifiedKFold)", 3, 10, 5, 1)
    max_features = st.select_slider("TF-IDF max_features", options=[500, 1000, 2000, 5000, 10000, 20000], value=5000)

    if up:
        df = pd.read_csv(up)
        if not {"label", "message"}.issubset(df.columns):
            st.error("❌ CSV must have 'label' and 'message'.")
        else:
            df = df.dropna(subset=["message"]).copy()
            df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
            X_train, X_test, y_train, y_test = train_test_split(
                df["message"], df["label_num"], test_size=test_size, random_state=42, stratify=df["label_num"]
            )

            models = {
                "LogisticRegression": LogisticRegression(max_iter=200),
                "MultinomialNB": MultinomialNB(),
                "LinearSVC": LinearSVC(),  # will use decision_function
                "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42)
            }

            results = []
            roc_curves = {}
            pr_curves = {}

            for name, clf in models.items():
                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1,2))),
                    ("clf", clf)
                ])
                # CV Accuracy
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_acc = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv).mean()

                # Fit and evaluate on test
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                # Probabilities / scores
                if hasattr(pipe.named_steps["clf"], "predict_proba"):
                    y_proba = pipe.predict_proba(X_test)[:, 1]
                elif hasattr(pipe.named_steps["clf"], "decision_function"):
                    scores = pipe.decision_function(X_test)
                    smin, smax = scores.min(), scores.max()
                    y_proba = (scores - smin) / (smax - smin) if smax != smin else np.full_like(scores, 0.5, dtype=float)
                else:
                    y_proba = y_pred.astype(float)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                try:
                    auc_roc = roc_auc_score(y_test, y_proba)
                except Exception:
                    auc_roc = np.nan

                results.append([name, cv_acc, acc, f1, auc_roc])

                # Curves
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                prec, rec, _ = precision_recall_curve(y_test, y_proba)
                roc_curves[name] = (fpr, tpr)
                pr_curves[name] = (rec, prec)

                # Keep best (by F1) for export
                pipe._val_f1 = f1
                models[name] = pipe

            res_df = pd.DataFrame(results, columns=["Model", "CV_Acc", "Test_Acc", "Test_F1", "Test_ROC_AUC"]).sort_values("Test_F1", ascending=False)
            st.subheader("📈 Model Comparison")
            st.dataframe(res_df, use_container_width=True)

            # ROC Overlay
            st.subheader("ROC Curves (Test Set)")
            fig, ax = plt.subplots()
            for name, (fpr, tpr) in roc_curves.items():
                ax.plot(fpr, tpr, lw=1.8, label=name)
            ax.plot([0,1],[0,1], linestyle="--")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC")
            ax.legend()
            st.pyplot(fig)

            # PR Overlay
            st.subheader("Precision-Recall Curves (Test Set)")
            fig, ax = plt.subplots()
            for name, (rec, prec) in pr_curves.items():
                ax.plot(rec, prec, lw=1.8, label=name)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall")
            ax.legend()
            st.pyplot(fig)

            # Save & Download best model
            best_name = res_df.iloc[0]["Model"]
            best_pipe = models[best_name]
            st.success(f"🏆 Best Model: **{best_name}**  (Test F1 = {res_df.iloc[0]['Test_F1']:.3f})")

            # Persist best pipeline (includes vectorizer), plus separate model/vectorizer for compatibility
            joblib.dump(best_pipe, "best_pipeline.pkl")
            # Extract components if TF-IDF + clf
            best_vect = best_pipe.named_steps["tfidf"]
            best_clf  = best_pipe.named_steps["clf"]
            joblib.dump(best_vect, "vectorizer.pkl")
            joblib.dump(best_clf, "model.pkl")

            with open("best_pipeline.pkl", "rb") as f: pipe_bytes = f.read()
            with open("vectorizer.pkl", "rb") as f: vect_bytes = f.read()
            with open("model.pkl", "rb") as f: model_bytes = f.read()

            st.caption("Download trained artifacts")
            bytes_download_button(pipe_bytes, "best_pipeline.pkl", "⬇ Download best_pipeline.pkl", "application/octet-stream")
            bytes_download_button(vect_bytes, "vectorizer.pkl", "⬇ Download vectorizer.pkl", "application/octet-stream")
            bytes_download_button(model_bytes, "model.pkl", "⬇ Download model.pkl", "application/octet-stream")

# =====================
# 7) EDA Report
# =====================
elif option == "📑 EDA Report":
    st.title("📑 Automated EDA Report (Sweetviz / ydata-profiling)")
    up = st.file_uploader("Upload CSV (any schema). If 'label' & 'message' present, extra NLP info will show.", type="csv")
    tool = st.selectbox("Pick EDA engine", ["Auto (try both)", "Sweetviz", "Quick Fallback"])

    if up:
        df = pd.read_csv(up)
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        if tool in ("Auto (try both)", "Sweetviz"):
            try:
                import sweetviz as sv
                st.info("Generating Sweetviz report...")
                report = sv.analyze(df)
                report_path = "sweetviz_report.html"
                report.show_html(report_path, open_browser=False)
                with open(report_path, "rb") as f:
                    st.download_button("⬇ Download Sweetviz Report", f.read(), "sweetviz_report.html", "text/html")
                st.components.v1.html(open(report_path, "r", encoding="utf-8").read(), height=700, scrolling=True)
            except Exception as e:
                if tool == "Sweetviz":
                    st.error(f"Sweetviz failed: {e}")

        if tool == "Quick Fallback":
            st.info("Quick fallback EDA (no external libs).")
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
            st.write("**Nulls per column:**")
            st.write(df.isna().sum())
            st.write("**Basic Describe (numeric):**")
            st.write(df.describe())

            if {"label", "message"}.issubset(df.columns):
                st.subheader("Extra NLP Quick Stats")
                lens = df["message"].fillna("").str.len()
                st.write("Avg length:", lens.mean())
                st.write("Max length:", lens.max())
                st.write("Min length:", lens.min())

                fig, ax = plt.subplots()
                ax.hist(lens, bins=50, color="skyblue", edgecolor="black")
                ax.set_title("Message Length Distribution")
                ax.set_xlabel("Message Length")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
               






##############
#### pip install -r smsrequirements.txt

