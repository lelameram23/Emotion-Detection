
import string
import speech_recognition as sr
import nltk
import re
import pandas as pd
from textblob import Word
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
import textblob as textblb
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

#class emoDatasetClass:
 #   def __init__(self):

nb = MultinomialNB()
from textblob.classifiers import NaiveBayesClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import voice

dataset = pd.read_csv("emo_dataset.csv", engine="python")
dataset = dataset.drop('tweet_id', axis=1)
dataset = dataset.drop('author', axis=1)
dataset = dataset.drop(dataset[dataset.sentiment == 'boredom'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'enthusiasm'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'empty'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'fun'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'relief'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'surprise'].index)
dataset = dataset.drop(dataset[dataset.sentiment == 'love'].index)
dataset.sentiment.unique()
# print(dataset.groupby('sentiment').size())
# print(dataset.columns)
# print(dataset["content"].describe())
# PREPROCESSİNG
dataset['content'] = dataset['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# dataset['arroba'] = dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))

# dataset['hastags'] = dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
# print(dataset.head())
def remove_pattern(text, pattern):
    r = re.findall(pattern, text)
    for i in r:
        text = re.sub(i, '', text)
    return text


dataset['content'] = np.vectorize(remove_pattern)(dataset['content'], "@[\w]*")
dataset['content'] = dataset['content'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)", '', str(x)))
# print(dataset.head())
dataset['content'] = dataset['content'].str.replace('[^\w\s]', ' ')
stop = stopwords.words('english')
dataset['content'] = dataset['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dataset['content'] = dataset['content'].apply(
    lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# print(dataset.head())
def de_repeat(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
#symspellpy

dataset['content'] = dataset['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
freq = pd.Series(' '.join(dataset['content']).split()).value_counts()[-10000:]
freq = list(freq.index)
dataset['content'] = dataset['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
# print(freq)
# dataset['content'] = dataset['content'][:5].apply(lambda x: str(TextBlob(x).correct()))
# print(dataset.head(10))
# print(dataset.describe()
# dataset = pd.read_csv('C:/Users/hp/PycharmProjects/EmotionsReg/train_data.csv')

# FEATURE EXTRACTİON-ÖZELLİK ÇIKARIMI-VEKTÖRİZE ETME İŞLEMİ

pre_data = preprocessing.LabelEncoder()
y = pre_data.fit_transform(dataset.sentiment.values)
# print(y)

X_train, X_val, y_train, y_val = train_test_split(dataset.content.values, y, stratify=y, shuffle=True,
                                                  random_state=42, test_size=0.1)
# print(X_train, X_val)
# print(X_train.shape)
# print(X_val.shape)

tfidf = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1, 3))
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)
# print(X_train_tfidf, X_val_tfidf)
count_vect = CountVectorizer(analyzer='word')
count_vect.fit(dataset['content'])
X_train_count = count_vect.transform(X_train)
X_val_count = count_vect.transform(X_val)
# print(X_train_count, X_val_count)
# print(dataset.head(5))

#### Linear SVM
# text=voice.sesal()
lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
lsvm.fit(X_train_count, y_train)
y_pred = lsvm.predict(X_val_count)


# print('lsvm using count vectors accuracy: %s' % accuracy_score(y_pred, y_val))
def pre_pro(text):
    print("Preprocessing fonksiyoyuna girdi")
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = " ".join(text.split())
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filter_text = [word for word in word_tokens if word not in stop_words]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    x_count = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    x_count = count_vect.transform([text])
   # x_count_lsvm = lsvm.predict(x_count)
   # return x_count_lsvm

    duygu_tahmini = lsvm.predict(x_count)
    duygu_tahmini = duygu_tahmini[0]
    print(duygu_tahmini)

    if duygu_tahmini == 0:
        duygu = "Kızgın"

    elif duygu_tahmini == 1:
        duygu = "Mutlu"

    elif duygu_tahmini == 2:
        duygu = "Nötr"

    elif duygu_tahmini == 3:
        duygu = "Üzgün"

    elif duygu_tahmini == 4:
        duygu = "Endişeli"


    print("Sonuç döndürülüyor")
    return [duygu, text]


# def lsvm(text):
# def sesidinle():
'''
while True:

    print()
    text = voice.sesal()
    if text is not None:
        if text == "quit" or text == "exit":
            break
        else:
            state = pre_pro(text)
            # result =lsvm(state)
            speak_text = ''.format(state)
            print("Current Mood ", speak_text)

        #  return speak_text
# print(result())
'''