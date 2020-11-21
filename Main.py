from plistlib import Data

import nltk
import re
import pandas as pd
from textblob import Word
from textblob import TextBlob
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import textblob as textblb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
from textblob.classifiers import NaiveBayesClassifier
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import voice
import string
import speech_recognition as sr
from textblob import TextBlob
import pyaudio
import _portaudio
from textblob.classifiers import  NaiveBayesClassifier
import pyttsx3

def sesal():

    #ses tanıma işlemi gerçekleşiyor
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening..")
        r.adjust_for_ambient_noise(source)
        r.pause_threshold=0.6
        audio=r.listen(source)
    try:
        print("Recording..")
        #speechrecognition kütüphanesi kullanılarak text işlemi gerçekleşiyor
        sonuc=r.recognize_google(audio,language='en-in')
        print(sonuc)

    except:
        return None
    return sonuc

def pre_pro(text):
    text=text.lower()
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text=text.translate(translator)
    text=" ".join(text.split())
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filter_text = [word for word in word_tokens if word not in stop_words]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return text

def feature_ext(lemmas):
        pre_data = preprocessing.LabelEncoder()
        y = pre_data.fit_transform(lemmas.values)
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
        text_tfidf = tfidf.fit_transform(lemmas)
        count_vect = CountVectorizer(analyzer='word')
        count_vect.fit(lemmas)
        text_count = count_vect.transform(lemmas)
        return text_count

while True:

    print()
    text = sesal()
    if text is not None:
        if text == "quit" or text == "exit":
            break
        else:
            state = pre_pro(text)
            result =feature_ext(state)
            print(result)
            speak_text = 'I think You Are speaking {}'.format(result)
            print("Computer:", speak_text)


'''
#def proprocessing():
#Yeni dataset eklendi.Bu yazar sütununa ihtiyacımız olmadığı için kaldırdık.
#13 duygu durumundan işimize yarayan 5(mutlu-üzüntü-kızgın-endişeli-nötr)duygu harici duytguları kaldırdık

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
#print(dataset.groupby('sentiment').size())         #sentimentte bulunan uniq değerleri gruplayıp boyutlarını yazdırıyporuz
#print(dataset.columns) #SÜtunları ve tip gösterir
#print(dataset["content"].describe())  #content sütunundaki istatiksel özellikler.boş olmayan satır, tek bulunan,ençok bulanan,kaç kere bulunduğu
#PREPROCESSİNG
#büyük harfleri küçük harfe döndürür
    dataset['content'] = dataset['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#Datasetteki # ve @ ile başlayan kelimeleri kontrol ediyoruz
    #dataset['arroba'] = dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
#'#' ile başlayan kelime yoktur
    #dataset['hastags'] = dataset['content'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
    #print(dataset.head())
#Sadece @ olan kelimeleri kaldırıp datasetimizi iyileştirmeye çalışıyoruz.
    def remove_pattern(text, pattern):
        r = re.findall(pattern, text)
        for i in r:
            text = re.sub(i, '', text)
        return text
    dataset['content'] = np.vectorize(remove_pattern)(dataset['content'], "@[\w]*")
#http bağlantılarını kaldırmak için
    dataset['content'] = dataset['content'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
    #print(dataset.head())
#gereksiz noktalama işlemlerini yapar
    dataset['content'] = dataset['content'].str.replace('[^\w\s]',' ')
    stop = stopwords.words('english')
#Durdurma kelimeleri olan a an with vb bize lazım olmayan kelimeleri kaldırır
    dataset['content'] = dataset['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Stemming yerine lemmatization işlemini yapıyoruz çünkü kelimeynin köküne inerek morfolojik analiz yapar ve gereken kök kalıbını verir.
    dataset['content'] = dataset['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#kelimelerde ard arda 2 harften fazla gelmeyeceği için harf tekrarlarını düzeltme işlemi
    #print(dataset.head())
    def de_repeat(text):
        pattern = re.compile(r"(.)\1{2,}")
        return pattern.sub(r"\1\1", text)
    dataset['content'] = dataset['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))
#Nadir kelimeleri kaldırmak için, çok nadir olduğu için bize sadece yük olacaktır
    freq = pd.Series(' '.join(dataset['content']).split()).value_counts()[-10000:]
    freq = list(freq.index)
    dataset['content'] = dataset['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
    #print(freq)
#Yazım yanlışlarını düzeltmek için, özellik çıkarımında işimize yarayacak.
    #dataset['content'] = dataset['content'][:5].apply(lambda x: str(TextBlob(x).correct()))
    #print(dataset.head(10))
    #print(dataset.describe()
    #dataset = pd.read_csv('C:/Users/hp/PycharmProjects/EmotionsReg/train_data.csv')

#def feature_ext():
    #dataset=preprocessing
    #FEATURE EXTRACTİON-ÖZELLİK ÇIKARIMI-VEKTÖRİZE ETME İŞLEMİ
#ileri seviye nlp işlemleri yapılır-N-gram-TFİDF-BoW
#metin verilerini sayısal verilere dönüştürüp etiketlemek için labelencoder kullanılır
    pre_data = preprocessing.LabelEncoder()
#sentiment sütununun verilerini etiketlemek için y değişkeninin içine verileri sıkıştırıp, yeniden kodlanmış metinler elde ediyoruz.
    y = pre_data.fit_transform(dataset.sentiment.values)
#Duygu durumlarını sayısal veri olarak 1-
    #print(y)

#Dataseti train-test kısımlarına ayırıyoruz.X-CONTENT, Y-SENTİMENT
#random state-bölmeyi uygulamadan önce karıştırma işlemini kontrol eder.Shuffle-karıştırılıp karıştırılmayacağını kontrol eder
#stratify-karıştırılmazsa y yi olduğu gibi al
    X_train, X_val, y_train, y_val = train_test_split(dataset.content.values, y, stratify=y, shuffle=True, random_state=42, test_size=0.1)
    #print(X_train, X_val)
#x trainin kaç satır ve sütundan oluştuğunu söyler
    #print(X_train.shape)
    #print(X_val.shape)

#n-gram, TFIDF
#analyzer:önişleme ve ngram adımlarını korur.Word şeklinde alır
#ngram aralığı word biçiminde bigram-trigram arasındadır
#max_features alınan kelimelerin max kaç tane oluşacağını söyler.1000
    tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
#eğitim özelliklerine uyun ve vektöre dönüştürün
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.fit_transform(X_val)
#tf-idf bize terimin datasetteki değerini verir
    #print(X_train_tfidf, X_val_tfidf)
#BoW
#word tipinde arama yapılıyor
    count_vect = CountVectorizer(analyzer='word')
#verileri bir kelime torbasına dönüştürür
    count_vect.fit(dataset['content'])
    X_train_count =  count_vect.transform(X_train)
    X_val_count =  count_vect.transform(X_val)
    #print(X_train_count, X_val_count)
    #print(dataset.head(5))


#def classfier():
 #   X_train_count=feature_ext()
  #  y_train=feature_ext()
   # X_val_count=feature_ext()
    #y_val=feature_ext()
#MULTİNOMİAL CLASSFİER

###Multinomal NB

    nb.fit(X_train_count, y_train)
    y_pred = nb.predict(X_val_count)
    print(y_pred)
    print('Multinomial Naive Bayes count vectors accuracy: %s' % accuracy_score(y_pred, y_val))

###Random Forest

    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train_count, y_train)
    y_pred = rf.predict(X_val_count)
    print('Random Forest with count vectors accuracy: %s' % accuracy_score(y_pred, y_val))

#### Linear SVM

    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(X_train_count, y_train)
    y_pred = lsvm.predict(X_val_count)
    print('lsvm using count vectors accuracy: %s' % accuracy_score(y_pred, y_val))

### Logistic Regression

    logreg = LogisticRegression(C=1, dual=False, solver='lbfgs', max_iter=4000)
    logreg.fit(X_train_count, y_train)
    y_pred = logreg.predict(X_val_count)
    print('Logistic Regression count vectors accuracy: %s' % accuracy_score(y_pred, y_val))

#text=voice.sesal()
#text=preprocessing

'''
def pre_pro(text):
    text=text.lower()
    text = re.sub(r'\d+', '', text)
    translator = str.maketrans('', '', string.punctuation)
    text=text.translate(translator)
    text=" ".join(text.split())
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filter_text = [word for word in word_tokens if word not in stop_words]
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return text
    #FEATURE EXTRACTİON-ÖZELLİK ÇIKARIMI-VEKTÖRİZE ETME İŞLEMİ
#ileri seviye nlp işlemleri yapılır-N-gram-TFİDF-BoW
#metin verilerini sayısal verilere dönüştürüp etiketlemek için labelencoder kullanılır
pre_data = preprocessing.LabelEncoder()
#sentiment sütununun verilerini etiketlemek için y değişkeninin içine verileri sıkıştırıp, yeniden kodlanmış metinler elde ediyoruz.
y = pre_data.fit_transform(dataset.sentiment.values)
#Duygu durumlarını sayısal veri olarak 1-
    #print(y)

#Dataseti train-test kısımlarına ayırıyoruz.X-CONTENT, Y-SENTİMENT
#random state-bölmeyi uygulamadan önce karıştırma işlemini kontrol eder.Shuffle-karıştırılıp karıştırılmayacağını kontrol eder
#stratify-karıştırılmazsa y yi olduğu gibi al
X_train, X_val, y_train, y_val = train_test_split(dataset.content.values, y, stratify=y, shuffle=True, random_state=42, test_size=0.1)
    #print(X_train, X_val)
#x trainin kaç satır ve sütundan oluştuğunu söyler
    #print(X_train.shape)
    #print(X_val.shape)

#n-gram, TFIDF
#analyzer:önişleme ve ngram adımlarını korur.Word şeklinde alır
#ngram aralığı word biçiminde bigram-trigram arasındadır
#max_features alınan kelimelerin max kaç tane oluşacağını söyler.1000
tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
#eğitim özelliklerine uyun ve vektöre dönüştürün
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.fit_transform(X_val)
#tf-idf bize terimin datasetteki değerini verir
    #print(X_train_tfidf, X_val_tfidf)
#BoW
#word tipinde arama yapılıyor
count_vect = CountVectorizer(analyzer='word')
#verileri bir kelime torbasına dönüştürür
count_vect.fit(dataset['content'])
X_train_count =  count_vect.transform(X_train)
X_val_count =  count_vect.transform(X_val)
    #print(X_train_count, X_val_count)
    #print(dataset.head(5))
def feature_ext(lemmas):
    pre_data = preprocessing.LabelEncoder()
    y = pre_data.fit_transform(dataset.sentiment.values)
    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
    text_tfidf = tfidf.fit_transform(X_train)
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(dataset['content'])
    text_count = count_vect.transform(X_train)

    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(text_count)
    y_pred = lsvm.predict(text_count)
    print(y_pred)
    print('lsvm using count vectors accuracy: %s' % accuracy_score(y_pred))
    return text_count


#MULTİNOMİAL CLASSFİER

#### Linear SVM


def sesal():

    #ses tanıma işlemi gerçekleşiyor
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening..")
        r.adjust_for_ambient_noise(source)
        r.pause_threshold=0.6
        audio=r.listen(source)
    try:
        print("Recording..")
        #speechrecognition kütüphanesi kullanılarak text işlemi gerçekleşiyor
        sonuc=r.recognize_google(audio,language='en-in')
        print(sonuc)

    except:
        return None
    return sonuc

while True:

    print()
    text = sesal()
    if text is not None:
        if text == "quit" or text == "exit":
            break
        else:
            state = pre_pro(text)
            result =feature_ext(state)
            print(result)
            speak_text = 'I think You Are speaking {}'.format(result)
            print("Computer:", speak_text)




    tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
    x_tfidf = tfidf.fit_transform(text)
    count_vect = CountVectorizer(analyzer='word')
    count_vect.fit(dataset['content'])
    x_count =  count_vect.transform(text)
    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(text)
    Y_pred = lsvm.predict(text)
    print(y)
    print('lsvm using accuracy: %s' % accuracy_score(Y_pred, text))


    def lsvm(text):
        from sklearn.preprocessing import MultiLabelBinarizer

        onehot_enc = MultiLabelBinarizer()
        onehot_enc.fit(text)
        X_train, X_test, y_train, y_test = train_test_split(text, test_size=0.25, random_state=None)
        from sklearn.svm import LinearSVC

        lsvm = LinearSVC()
        lsvm.fit(onehot_enc.transform(X_train), y_train)

        score = lsvm.score(onehot_enc.transform(X_test), y_test)
        print(score)
        return y_test