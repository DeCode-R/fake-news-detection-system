# 📰 Fake News Detection System

## 📌 Project Overview

The Fake News Detection System is a Machine Learning-based project designed to classify news articles as either **Fake News** or **Real News** using Natural Language Processing (NLP) and Machine Learning techniques.

This project demonstrates skills in:

* Machine Learning
* Natural Language Processing (NLP)
* Data Preprocessing
* Text Classification
* Predictive Modeling

---

# 🚀 Features

✅ Detects Fake and Real News Articles
✅ Uses NLP for text preprocessing
✅ TF-IDF Vectorization for feature extraction
✅ Logistic Regression Machine Learning model
✅ High prediction accuracy
✅ User input prediction system

---

# 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* NLP Techniques

---

# 📂 Dataset

This project uses the **ISOT Fake News Dataset** from Kaggle.

Dataset Source:

* Fake News Detection Datasets by Emine Yetm

Dataset includes:

* `Fake.csv`
* `True.csv`

Labels used in this project:

| Label | Meaning   |
| ----- | --------- |
| 0     | Fake News |
| 1     | Real News |

The dataset contains thousands of real and fake news articles collected from multiple online news sources. ([Kaggle][1])

---

# ⚙ Installation

Install required libraries:

```bash id="o6u4l9"
pip install -r requirements.txt
```

---

# ▶ Run the Project

Run the application:

```bash id="c2r7zn"
python app.py
```

---

# 🧠 Machine Learning Workflow

```text id="m7q5rx"
Dataset → Text Cleaning → TF-IDF Vectorization → Train Model → Predict News
```

---

# 📊 Model Performance

The model is trained using:

* TF-IDF Vectorization
* Logistic Regression

The system achieves high prediction accuracy for detecting fake and real news articles.

---

# 📁 Project Structure

```text id="m4s2pk"
fake-news-detection-system/
│
├── dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── screenshots/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔥 Future Improvements

* Streamlit Web Application
* Deep Learning Integration
* Better NLP preprocessing
* Online deployment
* Real-time news detection

---

# 👨‍💻 Author

**Md. Raihan Rafe**

🎓 Computer Science & Engineering Graduate
💡 Interested in Artificial Intelligence, Machine Learning, and Data Science

---

[1]: https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets?utm_source=chatgpt.com "Fake News Detection Datasets"
