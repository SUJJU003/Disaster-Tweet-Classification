# Disaster Tweet Classification Using Natural Language Processing

## 1. Project Overview

This project focuses on classifying tweets into two categories: disaster-related and non-disaster-related tweets using Natural Language Processing (NLP).

Two independent approaches are implemented and evaluated:

1. **Traditional Machine Learning:** TF-IDF Vectorization with Logistic Regression.
2. **Transformer-Based Deep Learning:** Fine-tuning a pretrained DistilBERT model.

The objective is to compare traditional text classification techniques with contextual language representation models.

## 2. Problem Statement

Social media platforms contain large amounts of information during emergencies. However, not every tweet containing disaster-related words refers to an actual disaster.

This project aims to develop text classification models that distinguish genuine disaster-related tweets from ordinary or figurative usage.

## 3. Dataset Description

The project uses the Kaggle Natural Language Processing with Disaster Tweets dataset.

| Dataset | Records |
|---|---:|
| Training dataset | 7,613 |
| Testing dataset | 3,263 |

**Target classes:**
- `0` – Not Disaster
- `1` – Disaster

The training data contains labeled tweets, while the test data is used to generate Kaggle predictions.

## 4. Approach 1: Traditional Machine Learning

**TF-IDF + Logistic Regression**

This approach uses traditional text representation and machine learning classification.

### Workflow

1. Load the training and testing datasets.
2. Clean the tweet text by removing URLs, mentions, and unnecessary whitespace.
3. Split the training data into 80% training and 20% validation sets using stratification.
4. Convert text into numerical features using TF-IDF.
5. Train a Logistic Regression classifier.
6. Evaluate the model using classification metrics and a confusion matrix.
7. Generate predictions for the Kaggle test dataset.

### Techniques Used

- Text preprocessing
- TF-IDF Vectorization
- Unigrams and Bigrams
- Logistic Regression
- Classification Report
- Confusion Matrix

### Notebook

`notebooks/disaster_tweet_baseline.ipynb`

## 5. Approach 2: Transformer-Based Deep Learning

**DistilBERT**

This approach uses a pretrained transformer model to capture contextual information in tweets.

### Workflow

1. Load the training and testing datasets.
2. Use the original tweet text as model input.
3. Split the training data into 80% training and 20% validation sets using stratification.
4. Load the pretrained DistilBERT tokenizer.
5. Tokenize tweets with a maximum sequence length of 128.
6. Fine-tune the pretrained DistilBERT model for binary classification.
7. Evaluate the model using accuracy, precision, recall, and F1 score.
8. Generate predictions for the Kaggle test dataset.

### Model Configuration

| Parameter | Value |
|---|---|
| Pretrained model | distilbert-base-uncased |
| Epochs | 3 |
| Learning rate | 0.00002 |
| Training batch size | 16 |
| Evaluation batch size | 32 |
| Maximum token length | 128 |
| Weight decay | 0.01 |
| Optimizer | AdamW (Trainer default) |
| Hardware | GPU (Google Colab T4) |

### Notebook

`notebooks/bert_implementation.ipynb`

## 6. Experimental Results

Both approaches were evaluated independently using a stratified 80:20 validation split.

| Model | Validation F1 | Kaggle Public Score |
|---|---:|---:|
| TF-IDF + Logistic Regression (baseline) | 0.7701 | 0.79803 |
| Tuned TF-IDF + Logistic Regression | 0.7793 | 0.80447 |
| DistilBERT | 0.8129 | 0.83266 |

**Best-performing approach:** DistilBERT.

DistilBERT achieved a Kaggle public F1 score of 0.83266, outperforming the tuned TF-IDF approach in these experiments.

*Note: Validation F1 and Kaggle public leaderboard scores are measured on different evaluation sets and are reported separately.*

## 7. Model Comparison

| Feature | TF-IDF + Logistic Regression | DistilBERT |
|---|---|---|
| Approach | Traditional ML | Transformer-based deep learning |
| Text representation | Statistical word weights | Contextual representations |
| Feature extraction | TF-IDF | Pretrained tokenizer and transformer |
| Context understanding | Limited | Context-sensitive |
| Computational requirements | Low | Higher |
| Interpretability | Relatively high | More limited |
| Kaggle public score | 0.80447 | 0.83266 |

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Google Colab
- Kaggle

## 9. Repository Structure

```text
Disaster-Tweet-Classification/
│
├── data/
│
├── images/
│   ├── model_comparison.png
│   └── confusion_matrix.png
│
├── notebooks/
│   ├── disaster_tweet_baseline.ipynb
│   └── bert_implementation.ipynb
│
├── models/
│
├── outputs/
│   ├── classification_report.txt
│   └── classification_report.csv
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 10. How to Run the Project

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the competition datasets from Kaggle.
4. Place the CSV files in the appropriate data directory or update the notebook file paths.
5. Open the required notebook in Google Colab or Jupyter Notebook.
6. Execute the cells sequentially.
7. Evaluate the model and generate Kaggle submission files.

For DistilBERT, enable GPU acceleration in Google Colab for faster training.

## 11. Conclusion

This project demonstrates two independent approaches to disaster tweet classification.

The TF-IDF and Logistic Regression approach provides an efficient traditional machine learning baseline, while DistilBERT uses contextual language representations through transformer-based deep learning.

Based on the experimental results, DistilBERT achieved the highest F1 score among the implemented approaches.

## 12. Future Enhancements

- Experiment with additional transformer architectures.
- Perform systematic hyperparameter optimization.
- Analyze false positives and false negatives.
- Improve model generalization.
- Explore ensemble classification techniques.

## Dataset and Competition

[Kaggle – Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
