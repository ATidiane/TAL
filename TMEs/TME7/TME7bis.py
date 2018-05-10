# -*- coding: utf-8 -*-

import codecs
import re
import string
from collections import *

import nltk
import numpy as np
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

path2train = "corpus.tache1.learn.utf8"
path2test = "corpus.tache1.test.utf8"

# Pour changer le path du nltk_data, très très important 
nltk.data.path.append("/Infos/nltk/nltk_data")

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.snowball_stemmer = SnowballStemmer('french')
        #self.snowball_stemmer.stem(t)
        
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class SnowballTokenizer(object):
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer('french')

    def __call__(self, doc):
        return [self.snowball_stemmer.stem(t) for t in word_tokenize(doc)]


def readfile(path):
    """
    """
    with open(path, "r") as f:
        one = f.read()
        databrut = one.split("\n")
        data = np.array([d.split(' ', 1) for d in databrut])
        datay = np.array([d[0] for d in data])
        datax = np.array([d[-1] for d in data])

    return one, datax[:-1], datay[:-1]


def processing_datay(datay):
    """
    """
    datay = [re.sub('.*<[0-9]*:[0-9]*:C>', '0', dy) for dy in datay]
    datay = [re.sub('.*<[0-9]*:[0-9]*:M>', '1', dy) for dy in datay]
    # -1 cause the last one is equal to ''
    return np.array(datay, int)


def countCM(path):
    """
    """
    with open(path, 'r') as f:
        d = np.array(f.read().split("\n"))
        chirac = np.where(d == 'C')[0]
        mitterand = np.where(d == 'M')[0]

    return chirac.shape, mitterand.shape


punc = string.punctuation

languages = ['french', 'english', 'german', 'spanish']
stop_words = []
for l in languages:
    for w in stopwords.words(l):
        stop_words.append(w)

text_clf = Pipeline([('vect', txt.CountVectorizer(encoding=u'utf-8',
                                                  strip_accents=u'ascii',
                                                  lowercase=True, analyzer=u'word',
                                                  binary=True, vocabulary=None,
                                                  preprocessor=None)),
                     ('tfidf', txt.TfidfTransformer()),
                     ('clf', lin.LogisticRegression(n_jobs=-1)),
])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'vect__stop_words': (stop_words, None),
              'vect__tokenizer': (LemmaTokenizer(), None),
              'vect__max_features': (10000, 20000, 30000, 40000, 50000),
}

databrut, datax, datay = readfile(path2train)
datay = processing_datay(datay)
all_words = databrut.split()

databrut_test, datax_t, datay_t = readfile(path2test)

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(datax, datay)

prediction = gs_clf.predict(datax_t)  # usage sur une nouvelle donnée

gs_clf.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

txt_clf = Pipeline([('vect', txt.CountVectorizer(encoding=u'utf-8',
                                                  preprocessor=None,
                                                  ngram_range=(1, 2),
                                                  stop_words=None,
                                                  tokenizer=SnowballTokenizer(),
                                                  max_features=None)),
                    ('tfidf', txt.TfidfTransformer(use_idf=False)),
                    ('clf', lin.LogisticRegression(n_jobs=-1)),
])


# parameters = {'tfidf__use_idf': (True, False),
#               'vect__tokenizer': (LemmaTokenizer(), SnowballTokenizer(), None),
# }


# parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
#               'tfidf__use_idf': (True, False),
#               'vect__stop_words': (stop_words, None),
#               'vect__tokenizer': (LemmaTokenizer(), None),
#               'vect__max_features': (10000, 20000, 30000, 40000, 50000),
# }

datay=processing_datay(datay)
all_words=databrut.split()
databrut, datax, datay=readfile(path2train)

databrut_test, datax_t, datay_t=readfile(path2test)

# gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf=text_clf.fit(datax, datay)

prediction=text_clf.predict(datax_t)  # usage sur une nouvelle donnée

# gs_clf.best_score_

# for param_name in sorted(parameters.keys()):
#     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))


cleaned_prediction=[]
for label in prediction:
    cleaned_prediction.append("C" if label == 0 else "M")

with open("SORTIE2.txt", "w") as file:
    file.write("\n".join(cleaned_prediction))
    file.write("\nC\n")

print("CountCM : ", countCM("SORTIE.txt"))
print("countCM : ", countCM("SORTIE2.txt"))
