# RNN-Based Text Classification Model for Steam Comments

## Overview
This project is a **text classification system** designed to predict recommendations based on Steam user comments. It uses pre-trained Word2Vec embeddings with various deep learning architectures (LSTM, SimpleRNN, GRU) to analyze the text and classify it into binary categories (recommended or not recommended).

## Features
- **Data Preprocessing:** Text cleaning, lemmatization, stopword removal, and tokenization.
- **Word Embedding:** Utilizes a pre-trained Word2Vec model (CBOW with 300 dimensions).
- **Deep Learning Models:** Supports LSTM, SimpleRNN, and GRU architectures.
- **K-Fold Cross-Validation:** Implements 10-fold cross-validation to evaluate model performance.
- **Performance Metrics:** Calculates accuracy, precision, recall, and F1-score.

## Technologies Used
- **Python 3**
- **Keras (TensorFlow Backend):** Deep learning model development
- **Gensim:** Loading Word2Vec embeddings
- **NLTK:** Text preprocessing (tokenization, stopwords, lemmatization)
- **Scikit-learn:** Evaluation metrics and cross-validation
- **NumPy & Pandas:** Data manipulation

## Installation & Setup
1. **Prerequisites:**
   - Python 3.x
   - Install required libraries:
     ```bash
     pip install numpy pandas nltk gensim tensorflow scikit-learn
     ```

2. **Download NLTK Resources:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   nltk.download('stopwords')
   ```

3. **File Structure:**
   - `steam_train_comments.csv`: Dataset containing Steam comments with `comments` and `recommend` columns.
   - `Steam_Word2Vec_model_CBOW_300.bin`: Pre-trained Word2Vec binary file.

## Data Preprocessing
- **Cleaning:** Removes punctuation, digits, and converts text to lowercase.
- **Stopword Removal:** Eliminates common English stopwords.
- **Lemmatization:** Reduces words to their base form using POS tagging.
- **Tokenization:** Converts text into sequences of integers for model input.

## Model Architecture
- **Embedding Layer:** Uses pre-trained Word2Vec vectors (non-trainable).
- **Recurrent Layer:** Can be LSTM, SimpleRNN, or GRU with 128 units.
- **Dense Layers:** One hidden layer with ReLU activation and dropout for regularization.
- **Output Layer:** Sigmoid activation for binary classification.

## Running the Model
1. **Adjust Model Type:** Change `create_model('LSTM')` to `'SimpleRNN'` or `'GRU'` if needed.
2. **Train and Evaluate:** The code automatically performs 10-fold cross-validation.
3. **Output Metrics:** Displays average accuracy, precision, recall, and F1-score after training.

## Example Metrics Output
```
Average Accuracy: 0.85
Average Precision: 0.84
Average Recall: 0.83
Average F1-Score: 0.84
```

## License
This project is open-source and available for educational and research purposes.

## Acknowledgments
- **Steam Dataset** for providing real-world data.
- **Gensim** for efficient Word2Vec model integration.
- **Keras and TensorFlow** for robust deep learning capabilities.

For any issues, suggestions, or improvements, feel free to contribute!

