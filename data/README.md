# Natural Language Processing with Disaster Tweets

## Project Overview

This project focuses on classifying tweets into two categories:

- `0` — Not a real disaster
- `1` — Real disaster

The project is based on Kaggle's **Natural Language Processing with Disaster Tweets** competition.

The goal is to develop and compare NLP and machine learning models for disaster-related tweet classification and progressively improve the Kaggle leaderboard score.

---

## Kaggle Competition

[Disaster Tweets - Kaggle](https://www.kaggle.com/competitions/nlp-getting-started)

---

## Team Members

- Venkata Suraj K
- Reshmanth CH

---

## Dataset

The dataset contains tweets with the following major features:

- `id` — Tweet identifier
- `keyword` — Keyword associated with the tweet
- `location` — Location associated with the tweet
- `text` — Tweet content
- `target` — Binary target available in the training data

### Dataset Split

- `train.csv` — Training data with target labels
- `test.csv` — Test data without target labels
- `sample_submission.csv` — Kaggle submission format

> Dataset files are not stored in this repository.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- TensorFlow / Keras
- Hugging Face Transformers
- Matplotlib
- Seaborn
- Google Colab
- Kaggle

---

## Project Workflow

```text
Raw Tweets
    ↓
Data Exploration
    ↓
Text Preprocessing
    ↓
Train / Validation Split
    ↓
TF-IDF Feature Extraction
    ↓
Machine Learning Model
    ↓
Validation
    ↓
Kaggle Submission
    ↓
Model Improvement
