# Twitter Sentiment Analysis — Logistic Regression with TF-IDF

> A reproducible pipeline for sentiment classification of tweets using the Sentiment140 dataset, applying preprocessing, TF-IDF vectorization, and Logistic Regression to distinguish between positive and negative sentiments.

---

## 🎯 Objective

This repository demonstrates a practical approach to sentiment analysis using real-world Twitter data. The goal is to:

✔ Load and preprocess a large corpus of tweets

✔ Apply TF-IDF vectorization with unigrams and bigrams

✔ Train and evaluate a Logistic Regression baseline model

✔ Visualize classification results with confusion matrices

✔ Explore most frequent words in each sentiment class via Word Clouds

---

## 🧰 Technologies

- **Language**: Python  
- **Core Packages**:  
  - pandas, numpy – Data manipulation
  - re – Text preprocessing with regular expressions  
  - scikit-learn – Model training, evaluation, and TF-IDF
  - matplotlib, seaborn – Visualization 
  - wordcloud – Word cloud generation
    
---

## 📊 Analysis Workflow

1. **Data Loading**
   - Load the Sentiment140 dataset (1.6M tweets)
   - Map labels from {0, 4} to {Negative, Positive}

2. **Text Preprocessing**
   - Convert to lowercase
   - Remove URLs, mentions, punctuation, numbers
   - Remove extra spaces

3. **Vectorization**
   - Apply TF-IDF with:
     - max_features = 10,000
     - ngram_range = (1, 2) (unigrams + bigrams)

4. **Model Training**
   - Train a Logistic Regression classifier (max_iter = 100)
   - Split into training (80%) and testing (20%)

5. **Evaluation**
   - Compute classification report (Precision, Recall, F1-score)
   - Plot confusion matrix heatmap

6. **Visualization**
   - Generate word clouds for positive and negative tweets
          
---

## 📝 Example Output

### Confusion Matrix

![Posterior Plot](analise_missing_mice_mf_amelia.png)

> The matrix shows balanced performance between classes, with the model achieving ~83% accuracy on unseen tweets.

---

## 📝 Word Cloud Examples

### Positive Tweets

![Posterior Plot](analise_missing_mice_mf_amelia.png)

### Negative Tweets

![Posterior Plot](analise_missing_mice_mf_amelia.png)

---

## 🔄 Possible Extensions

- 🧠 Add deep learning models (e.g., LSTM, BERT) for improved accuracy
- 📊 Include a "Neutral" sentiment class for finer granularity
- 📁 Export results and predictions to interactive dashboards
- 🌐 Integrate with the Twitter API for real-time sentiment analysis
- 📉 Perform hyperparameter tuning and cross-validation

---

## 📚 Use Case

This project is ideal for:
- Data scientists learning NLP preprocessing and modeling
- Students practicing text classification with real-world data 
- Researchers building baseline sentiment analysis models 
- Portfolio demonstration of machine learning pipelines in Python

---

> 📎 _This project is for educational and demonstration purposes only. It simulates hypothetical data and is not based on real clinical trial results._
