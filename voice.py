

import speech_recognition as sr
from textblob import TextBlob
#import pyaudio
#import _portaudio
from textblob.classifiers import  NaiveBayesClassifier
import pyttsx3

def sesal():
    print("Ses dinleniyor")
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

    print("Ses dinlendi sonuc döndürülüyor")
    return sonuc
'''
def sesal():

    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)
    try:
        print("Google Speech Recognition thinks you said :" + r.recognize_google(audio))
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
    return r.recognize_google(audio)
'''