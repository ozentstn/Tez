import  bs4
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
sentence="Veri Bilimi Okuluna hoş geldiniz. Bugünkü blog yazısının konusu Natural Language Toolkit"
print(sent_tokenize(sentence))
etkisiz_kelimeler = list(stopwords.words('turkish'))