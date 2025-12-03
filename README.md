# Spam Detector Dashboard

**एक स्ट्रीमलिट (Streamlit) ऐप जो SMS/Message spam detection के लिए UI, EDA और training pipelines देता है।**

---

## क्या है

यह प्रोजेक्ट एक All-in-one Spam Detector Dashboard है। इसमें single message prediction, CSV-based bulk prediction, prediction history, model performance analysis (ROC/PR, threshold optimization), wordcloud, और TF-IDF + विभिन्न क्लासिफायर्स से training और model export की सुविधा है।

## मुख्य फ़ीचर्स

* एकल संदेश की prediction और स्पैम probability gauge।
* CSV से bulk prediction और रिपोर्ट डाउनलोड।
* Prediction history और export।
* Model evaluation: confusion matrix, ROC/PR curves, F1-vs-threshold और best threshold।
* WordCloud visualization (spam vs ham)।
* Train & Compare pipelines: LogisticRegression, MultinomialNB, LinearSVC, RandomForest।
* Best model को `best_pipeline.pkl`, `vectorizer.pkl`, और `model.pkl` के रूप में बचाना।
* Automated EDA विकल्प (Sweetviz या quick fallback)।

## आवश्यकताएँ

* Python 3.8+ (आप Python 3.12 भी उपयोग कर रहे हैं तो ठीक है)
* आवश्यक पैकेज (कम से कम):

  * streamlit
  * scikit-learn
  * pandas
  * numpy
  * joblib
  * matplotlib
  * seaborn
  * plotly
  * wordcloud
  * sweetviz (वैकल्पिक, EDA के लिए)

>  `pip install -r smsrequirements.txt` चला सकते हैं।

## कैसे चलाएँ (Run)

1. रिपोजिटरी क्लोन या फ़ाइलें अपनी मशीन पर रखो।
2. वर्चुअल एनवायरनमेंट बनाओ और एक्टिवेट करो (optional)।
3. dependencies इंस्टॉल करो:

```bash
pip install -r smsrequirements.txt
# या
pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn plotly wordcloud sweetviz
```

4. ऐप रन करो:

```bash
streamlit run sms_app2.py
```

5. ब्राउज़र में `http://localhost:8501` खोलो।

## जरूरी फ़ाइलें / आउटपुट

* `sms_app2.py` — मुख्य Streamlit ऐप।
* `model.pkl` — प्रशिक्षित क्लासिफायर (optional)
* `vectorizer.pkl` — TF-IDF वेक्टराइज़र (optional)
* `best_pipeline.pkl` — pipeline जिसमें vectorizer + classifier होता है (जब Train & Compare से सेव किया गया हो)

> अगर `model.pkl` और `vectorizer.pkl` मौजूद नहीं हैं तो Single Message और Bulk Prediction जितने विकल्प मॉडल-आधारित prediction हैं वे उपलब्ध नहीं होंगे। उस स्थिति में आप "Train & Compare Models" tab से dataset अपलोड करके model train कर सकते हैं।

## CSV इनपुट फॉर्मैट

* Bulk prediction के लिए CSV में `message` कॉलम होना चाहिए।
* Model Performance / WordCloud / Train के लिए CSV में `message` और `label` (values: `ham`/`spam`) कॉलम चाहिए।

## Training नोट्स

* Train & Compare section TF-IDF + multiple classifiers चलाता है।
* Best model को pipeline के रूप में सेव कर देता है और `vectorizer.pkl` तथा `model.pkl` भी अलग से निकाल देता है।
* Cross-validation और test-set पर evaluation compute होता है और graphs बनते हैं।

## Troubleshooting

* `joblib.load("model.pkl")` या `vectorizer.pkl` न मिलने पर ऐप warning देगा।
* TF-IDF vectorizer के लिए `get_feature_names_out` method न मिलने पर feature importance दिखने में problem आ सकती है।
* LinearSVC के साथ probability नहीं आता। ऐप decision_function को min-max करके probability जैसा स्कोर दर्शाता है।

## सुझाव / आगे सुधार

* UI में user authentication जोड़ना।
* Backend storage के लिए डेटाबेस (SQLite/Postgres) जोड़कर history persistent बनाना।
* HuggingFace transformers से बेहतर NLP embeddings जोड़ना।
* मोबाइल responsiveness और performance optimization।

## लाइसेंस

इस प्रोजेक्ट पर आप अपनी आवश्यकता के अनुसार लाइसेंस लगा सकते हैं। (MIT सुझावनीय)

---

अगर चाहो तो मैं यह README अंग्रेज़ी में भी दे दूँ या GitHub repo के लिए `README.md` के साथ LICENSE और `.gitignore` भी बना दूँ।

