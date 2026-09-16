<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=34&duration=3000&pause=800&color=7C3AED&center=true&vCenter=true&width=700&lines=%F0%9F%98%8A+NLP+Emotion+Classifier;Text+In.+Emotion+Out.;Powered+by+TF-IDF+%2B+Logistic+Regression" alt="NLP Emotion Classifier" />

### A Machine Learning powered NLP application that understands emotions from text.

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

**[📦 Repository](#-installation)** &nbsp;•&nbsp; **[🚀 Live Demo](#-interactive-web-app)** &nbsp;•&nbsp; **[📖 Documentation](#-nlp--machine-learning-pipeline)**

</div>

---

## 🧠 What Does This Project Do?

This application reads a sentence written in natural language and predicts **which of six emotions** the text expresses — along with how confident the model is in that prediction.

Type *"I achieved my goal today!"* and the classifier returns **Joy**, with a confidence percentage attached.

```mermaid
flowchart LR
    A["📝 User Text"] --> B["🧹 Text Preprocessing"]
    B --> C["🔢 TF-IDF Vectorization"]
    C --> D["📈 Logistic Regression"]
    D --> E["🎯 Emotion Prediction"]
    E --> F["📊 Confidence Score"]

    style A fill:#1e1b4b,stroke:#7C3AED,color:#fff
    style B fill:#1e1b4b,stroke:#7C3AED,color:#fff
    style C fill:#1e1b4b,stroke:#7C3AED,color:#fff
    style D fill:#312e81,stroke:#a78bfa,color:#fff
    style E fill:#4c1d95,stroke:#c4b5fd,color:#fff
    style F fill:#4c1d95,stroke:#c4b5fd,color:#fff
```

The project ships in two halves: a **Jupyter notebook** holding the full experimentation and training pipeline, and a **Streamlit web app** that serves the trained model for real-time prediction.

---

## 🎭 Emotion Classes

The model predicts exactly one of six classes. Labels were encoded alphabetically with scikit-learn's `LabelEncoder`, producing this mapping:

<div align="center">

| ID | Emotion | | ID | Emotion |
|:--:|:--------|:-:|:--:|:--------|
| **0** | 😡 &nbsp; Anger | | **3** | ❤️ &nbsp; Love |
| **1** | 😨 &nbsp; Fear | | **4** | 😢 &nbsp; Sadness |
| **2** | 😄 &nbsp; Joy | | **5** | 😲 &nbsp; Surprise |

</div>

> These six classes are what the application implements — any input is assigned to one of them.

---

## ⚙️ NLP & Machine Learning Pipeline

Every stage below is implemented in `main.ipynb`.

```text
Raw Text
   ↓
Punctuation Removal          ·  string.punctuation
   ↓
Number Removal               ·  digit stripping
   ↓
Tokenization                 ·  NLTK word_tokenize
   ↓
Stopword Removal             ·  NLTK English stopwords
   ↓
Feature Extraction           ·  CountVectorizer  /  TfidfVectorizer
   ↓
Model Training               ·  MultinomialNB  /  LogisticRegression
   ↓
Evaluation                   ·  accuracy_score
   ↓
Model Serialization          ·  Joblib
```

**Tools used**

<div align="center">

| Stage | Library |
|:------|:--------|
| Data loading | `Pandas` |
| Text processing | `NLTK` |
| Feature extraction | `CountVectorizer`, `TfidfVectorizer` |
| Classification | `MultinomialNB`, `LogisticRegression` |
| Evaluation | `scikit-learn` |
| Serialization | `Joblib` |
| Interface | `Streamlit` |

</div>

---

## 📊 Model Experiments

Four combinations of feature extraction and classifier were trained and compared on an 80/20 split (`random_state=42`).

<div align="center">

| Feature Extraction | Algorithm | Accuracy |
|:-------------------|:----------|---------:|
| Bag of Words | Logistic Regression | **88.88%** |
| TF-IDF | Logistic Regression | 85.03% |
| Bag of Words | Multinomial Naive Bayes | 77.16% |
| TF-IDF | Multinomial Naive Bayes | 65.56% |

</div>

```text
Accuracy comparison

BoW  + LogisticRegression   ████████████████████████████████████░░░░   88.88%
TFIDF + LogisticRegression  ██████████████████████████████████░░░░░░   85.03%
BoW  + MultinomialNB        ███████████████████████████████░░░░░░░░░   77.16%
TFIDF + MultinomialNB       ██████████████████████████░░░░░░░░░░░░░░   65.56%
```

> **Recorded Experiment Results** — these values are the accuracies recorded in the project's notebook. Accuracy is the only metric measured; no precision, recall or F1 scores were computed.

The deployed application uses the **TF-IDF + Logistic Regression** model, which is the combination serialized to `LogisticRegression.pkl` and `tfidf_vectorizer.pkl`.

---

## 🔍 Why TF-IDF?

Machine-learning classifiers operate on numbers, not words. TF-IDF (Term Frequency–Inverse Document Frequency) converts each sentence into a numerical vector, weighting terms by how often they appear in a document against how common they are across the whole corpus — so words appearing everywhere carry less weight than distinctive ones.

This is a representation step, not a magic one. It gives the classifier a usable numerical view of the text; the accuracy table above shows what that produced on this dataset.

---

## 🚀 Interactive Web App

`app.py` serves the model through Streamlit:

```text
┌─────────────────────────────────────────┐
│  😊 Emotion Classifier                  │
│                                         │
│  Enter a sentence and the trained NLP   │
│  model will predict its emotion.        │
│                                         │
│  Enter your text                        │
│  ┌───────────────────────────────────┐  │
│  │ I achieved my goal today!         │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│        [  Predict Emotion  ]            │
│                                         │
│  ✅ Predicted Emotion: Joy              │
│  ℹ️  Confidence: XX.XX%                  │
└─────────────────────────────────────────┘
```

**Implemented behaviour**

- Text area for free-form input
- Warns the user if the input is empty
- Transforms input with the saved TF-IDF vectorizer
- Predicts the class with the trained Logistic Regression model
- Maps the predicted ID to its emotion name
- Displays a confidence percentage via `predict_proba()` where the model supports it
- Models are cached with `@st.cache_resource`, so they load once per session

---

## 🏗️ System Architecture

```text
                ┌───────────────────┐
                │    User Text      │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Text Preprocessing│
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ TF-IDF Vectorizer │
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ LogisticRegression│
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Emotion Prediction│
                └─────────┬─────────┘
                          ↓
                ┌───────────────────┐
                │ Confidence Score  │
                └───────────────────┘
```

---

## 📁 Project Structure

```text
NLP-Emotion-Classifier/
│
├── app.py                    # Streamlit web application
├── main.ipynb                # Training & experimentation notebook
├── train.txt                 # Dataset — "text;emotion" per line
├── LogisticRegression.pkl    # Serialized trained classifier
├── tfidf_vectorizer.pkl      # Serialized fitted TF-IDF vectorizer
├── requirements.txt          # Python dependencies
└── README.md
```

| File | Purpose |
|:-----|:--------|
| `app.py` | Loads the saved model and vectorizer, renders the UI, runs prediction |
| `main.ipynb` | Full pipeline: cleaning → vectorizing → training → comparison → export |
| `train.txt` | Source data, each line formatted as `text;emotion` |
| `*.pkl` | Joblib artifacts consumed by the app at runtime |

---

## 🔧 Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd NLP-Emotion-Classifier
pip install -r requirements.txt
```

> If you intend to run `main.ipynb`, note that it also requires **NLTK** along with its `stopwords` and `punkt` data:
> ```python
> import nltk
> nltk.download('stopwords')
> nltk.download('punkt')
> ```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

`app.py` resolves the model paths relative to its own location:

```python
BASE_DIR = Path(__file__).resolve().parent
```

So **`LogisticRegression.pkl` and `tfidf_vectorizer.pkl` must sit in the same directory as `app.py`**, or loading will fail on startup.

---

## 📓 Notebook Workflow

`main.ipynb` contains the complete experimentation pipeline, in order:

1. Load `train.txt` with Pandas (`sep=';'`, columns `text` / `emotions`)
2. Encode emotion labels with `LabelEncoder`
3. Remove punctuation
4. Remove numeric characters
5. Tokenize with NLTK `word_tokenize`
6. Remove English stopwords
7. Build feature matrix `X` and target `y`
8. Generate Bag-of-Words features with `CountVectorizer`
9. Generate TF-IDF features with `TfidfVectorizer`
10. Split 80/20 with `random_state=42`
11. Train Multinomial Naive Bayes
12. Train Logistic Regression
13. Compare all four feature/model combinations by accuracy
14. Export the final model and vectorizer with Joblib

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-2C5BB4?style=flat-square&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

</div>

---

## ⚠️ Limitations

- The classifier predicts among **six predefined emotion classes only** — any input, however unrelated, is forced into one of them.
- Prediction quality depends on the training data and the preprocessing choices made in the notebook.
- Text carrying **ambiguous, mixed or multiple emotions** is difficult for a single-label classifier to handle.
- The confidence value is the model's **probability estimate**, not a guarantee that the prediction is correct. A high confidence score can still accompany a wrong answer.
- Evaluation used **accuracy alone**. Accuracy on its own does not reveal how the model performs on individual emotion classes.
- Preprocessing removes stopwords and punctuation, so negations and emphasis carried by those tokens are discarded before the model ever sees the text.

---

## 🔮 Future Improvements

> Ideas for future work — **none of the following are implemented in the current version.**

- [ ] Confusion matrix and per-class precision / recall / F1
- [ ] Hyperparameter tuning (grid or randomized search)
- [ ] Richer preprocessing: lemmatization, n-grams, negation handling
- [ ] Compare further classifiers (SVM, Random Forest, gradient boosting)
- [ ] Transformer-based emotion classification (e.g. BERT fine-tuning)
- [ ] Larger and more diverse training data
- [ ] Probability visualization across all six emotions, not just the top one
- [ ] Improved Streamlit UI with example inputs and history
- [ ] Deployment as a hosted app or REST API
- [ ] Dockerization for reproducible environments
- [ ] Automated testing for the preprocessing and prediction path

---

<div align="center">

### Built with Python, NLP & Machine Learning.

⭐ **If you found this project useful, consider starring the repository.**

</div>