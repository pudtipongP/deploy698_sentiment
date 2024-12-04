https://deploy698sentiment-bdmrtujuxbtrfdc67kqdrt.streamlit.app/

# Sentiment Analysis Workshop

## Overview
This workshop provides a comprehensive exploration of sentiment analysis techniques using various approaches in Python, covering rule-based, machine learning, and pre-trained models.

## Key Components

### 1. Rule-Based Approaches
- **Methodology**: Uses lexicon-based methods and pattern matching
- **Techniques**:
  - Tokenization with NLTK
  - Stopwords removal
  - Simple sentiment scoring based on predefined word lists

#### Example Rule-Based Sentiment Analyzer
- Uses positive and negative word dictionaries
- Calculates sentiment score by comparing tokens
- Classifies text as Positive, Negative, or Neutral

### 2. Machine Learning Approaches

#### TF-IDF Vectorization
- Converts text data into numerical features
- Captures word importance in documents
- Uses scikit-learn's TfidfVectorizer

#### Naive Bayes Classifier
- Probabilistic machine learning model
- Works well with text classification
- Uses MultinomialNB from scikit-learn

#### Workflow
1. Data Preparation
2. Text Preprocessing
3. Feature Extraction (TF-IDF)
4. Model Training
5. Prediction and Evaluation

### 3. Pre-trained Models with Hugging Face

#### English Sentiment Analysis
- Model: DistilBERT fine-tuned on SST-2 dataset
- Quick and easy sentiment prediction
- Provides confidence scores

#### Thai Language Sentiment Analysis
- Model: WangchanBERTa fine-tuned for sentiment
- Supports Thai language text analysis
- Classifies text into positive, negative sentiment

## Key Libraries
- NLTK: Natural Language Processing
- Scikit-learn: Machine Learning
- Transformers: Pre-trained models
- Streamlit: Web application development

## Limitations
- Context understanding challenges
- Sarcasm and irony detection
- Varying performance across domains
- Language-specific constraints

## Future Work
- Incorporate deep learning models
- Develop transfer learning techniques
- Create more robust multilingual models
- Improve contextual understanding
