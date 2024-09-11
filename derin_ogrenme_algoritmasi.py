import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout, LSTM, SimpleRNN, GRU
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
import nltk
import json


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Lemmatizer ve stop word'ler belirleme
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# POS etiketini (sözlük türünü) almak için fonksiyon
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('N'):
        return NOUN
    elif treebank_tag.startswith('R'):
        return ADV
    else:
        return NOUN

# Metni lemmatize etmek için fonksiyon
def lemmatize_text(text):
    try:
        if isinstance(text, str):
            words = word_tokenize(text)
            pos_tags = nltk.pos_tag(words)
            lemmatized_words = [lemma.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tags]
            return " ".join(lemmatized_words)

    except TypeError:
        pass

# Verinin yüklenmesi
file_path_data = '/content/drive/MyDrive/Metin Madenciliği/steam_train_comments.csv'
df = pd.read_csv(file_path_data)

df['comments'] = df['comments'].astype(str) # verilerin string formatina dönüştürülmesi
df['comments'] = df['comments'].str.replace('[^\w\s]', '', regex=True) #noktalama işaretlerinin kaldırılması 
df['comments'] = df['comments'].str.replace('\d', '', regex=True) # sayıların kaldırılması 
df['comments'] = df['comments'].apply(lambda x: " ".join(word.lower() for word in x.split() if word.lower() not in stop_words)) # stopwordlerın kaldırılması 
df = df[df['comments'].apply(lambda x: len(x.split()) > 1)] # 1 kelımeden az cumlelerın kullanılması 
df['comments'] = df['comments'].apply(lemmatize_text) # kok bulma ıslemı 
df['comments'] = df['comments'].apply(lambda x: ' '.join(word for word in x.split() if len(word) > 2)) # 1 harften az olan kelımelerın sılınmesı
df = df.replace(r'^\s*$', np.nan, regex=True) 
df.dropna(subset=['comments'], inplace=True) # bos satırların sılınmesı 

# Tokenize ıslemı
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['comments'])
sequences = tokenizer.texts_to_sequences(df['comments'])
word_index = tokenizer.word_index
max_len = max(len(seq) for seq in sequences)
padded_data = pad_sequences(sequences, maxlen=max_len)

# Etiketlerin hazırlanması
labels = df['recommend'].values
labels = to_categorical(labels)

# Verilerin boyutlarında sıkıntı yasadıgım ıcın verı boyutu kontolu yapıldı satır uyusmazlıgı yasadım 
assert len(padded_data) == len(labels), "Veri boyutları eşleşmiyor!"

# Word2Vec modelini yüklenmesı 
file_path_w2v = '/content/drive/MyDrive/Metin Madenciliği/Steam_Word2Vec_model_CBOW_300.bin'
word_vectors = KeyedVectors.load_word2vec_format(file_path_w2v)
embedding_dim = 300
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in word_vectors:
        embedding_matrix[i] = word_vectors[word]

# Model oluşturma fonksiyonu
def create_model(model_type='LSTM'):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False))
    if model_type == 'LSTM':
        model.add(LSTM(128))
    elif model_type == 'SimpleRNN':
        model.add(SimpleRNN(128))
    elif model_type == 'GRU':
        model.add(GRU(128))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# K-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train, test in kfold.split(padded_data, labels.argmax(axis=1)):
    model = create_model('LSTM')
    model.fit(padded_data[train], labels[train], epochs=5, batch_size=128, verbose=1)
    predictions = model.predict(padded_data[test])
    predicted_labels = predictions.argmax(axis=1)
    true_labels = labels[test].argmax(axis=1)
    accuracy_scores.append(accuracy_score(true_labels, predicted_labels))
    precision_scores.append(precision_score(true_labels, predicted_labels, average='macro'))
    recall_scores.append(recall_score(true_labels, predicted_labels, average='macro'))
    f1_scores.append(f1_score(true_labels, predicted_labels, average='macro'))

print(f'Average Accuracy: {np.mean(accuracy_scores)}')
print(f'Average Precision: {np.mean(precision_scores)}')
print(f'Average Recall: {np.mean(recall_scores)}')
print(f'Average F1-Score: {np.mean(f1_scores)}')
