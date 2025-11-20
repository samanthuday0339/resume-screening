# 📄 Resume Screening System with NLP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit_Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📖 Project Overview
HR departments often receive thousands of resumes for various job descriptions. Manually categorizing them is time-consuming and prone to error. 

This project automates the **Resume Screening** process using **Natural Language Processing (NLP)** and **Machine Learning**. The system takes a raw resume text as input, cleans/processes the text, and predicts the specific job category (e.g., Data Science, Java Developer, HR) it belongs to.

## 📊 Dataset Details
The project uses the `ResumeDataSet.csv` file.
* **Total Records:** 962 Resumes
* **Total Categories:** 25 unique job profiles
* **Key Categories:** Data Science, Java Developer, HR, DevOps Engineer, Blockchain, Python Developer, etc.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **NLP Preprocessing:** NLTK (Stopwords, WordNetLemmatizer), Regex (`re`)
* **Machine Learning:** Scikit-Learn (TfidfVectorizer, KNeighborsClassifier, OneVsRestClassifier)

## ⚙️ Methodology

### 1. Data Preprocessing & Cleaning
Raw text data contains noise which is removed using Regex and NLTK:
* Removed URLs (http/https).
* Removed RT and cc.
* Removed Hashtags (#) and Mentions (@).
* Removed Special Characters and Punctuation.
* **Lemmatization:** Converted words to their base root form.

### 2. Feature Extraction
* **TF-IDF (Term Frequency-Inverse Document Frequency):** Converted the cleaned text data into numerical vectors to reflect the importance of keywords in the resumes.

### 3. Model Building
* **Algorithm:** K-Nearest Neighbors (KNN).
* **Strategy:** OneVsRestClassifier (to handle multi-class classification).
* **Training Split:** 80% Training, 20% Testing.

## 📈 Performance
The model achieved high accuracy on the test dataset:

| Metric | Score |
| :--- | :---: |
| **Accuracy** | **96.37%** |

## 🚀 Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/resume-screening.git](https://github.com/your-username/resume-screening.git)
    cd resume-screening
    ```

2.  **Install Dependencies**
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn
    ```

3.  **Run the Project**
    * Run the training script (or notebook) to generate the models.
    * The trained models are saved as `clf.pkl` (Classifier) and `tfidf.pkl` (Vectorizer).

## 🔮 Prediction System
You can use the saved models to predict the category of a new resume:

```python
import pickle
import re

# Load Models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(txt):
    # ... (Include cleaning function from code) ...
    return cleaned_text

# Example Input
my_resume = "Skills: Python, Data Analysis, Machine Learning..."
cleaned_text = clean_resume(my_resume)
input_features = tfidf.transform([cleaned_text])

# Predict
prediction_id = clf.predict(input_features)[0]
# Map ID to Category Name (Mapping provided in script)
print(prediction_id)
