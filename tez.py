from turtledemo.round_dance import stop

import content
import nltk.data
import pypyodbc
import re
import os
import nltk
import string
import numpy as np
import itertools
import math
import datetime
import spacy
import docx
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from string import punctuation
import matplotlib.pyplot as plt #kmeans
import pandas as pd #kmeans
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#Read Word
file_path="C:\\Users\\ozent\\OneDrive\\Masaüstü\\orhanPamukYeniHayat.docx"
def read_docx(file_path):
    document = docx.Document(file_path)
    text = []
    for paragraph in document.paragraphs:
        text.append(paragraph.text)
    return 'n'.join(text)
document=read_docx(file_path)

#stringi array/series/df çevir
document_split=document.split(" ")
vector=pd.Series(document_split)
document_vector=vector[1:len(vector)] #0 dan başladığı için 1 den başlasın
document_df=pd.DataFrame(document_vector,columns=["stories"])
#küçük harf dönüşümü
copy_document_df=document_df.copy() #orjinali dursun
upperLowerCase=copy_document_df["stories"].apply(lambda x:" ".join(x.lower() for x in x.split()))
#sayıların silinmesi
removeNumbers=[x for x in upperLowerCase if not x.isnumeric()]
#noktalama işaretleri
noktalama=str.maketrans('','',string.punctuation)
removeNoktalama=[x.translate(noktalama) for x in removeNumbers]
#stopwords silinmesi
sw=stopwords.words("turkish")
stopWords=[x for x in removeNoktalama if x not in sw]
#frekansı az olan kelimelerin silinmesi
"""delete=pd.Series(" ".join(stopWords).split()).value_counts()<5 #sayısı 5 ten küçük olanlar
deleteFewWords=stopWords(lambda x:" ".join(x for x in x.split() if x not in delete))"""
#tokenization
nltk.download("punkt")
import textblob
from textblob import TextBlob
TextBlob(stopWords)
print(deleteFewWords)

