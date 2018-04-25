# -*- coding: utf-8 -*-

import re
import string
import numpy as np
from collections import *
import sklearn.naive_bayes as nb
from sklearn import svm
from sklearn import linear_model as lin
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer

<<<<<<< HEAD

def readfile(path):
    """ Reads the file in the given path, then returns a string which contains
        all the text, a numpy array text data for each sentence and finally the 
        corresponding labels "?:?:C" or "?:?:M".
    """
    with open(path, "r") as f:
        punc = string.punctuation
        one = f.read()
        databrut = one.split("\n")[:-1]
        data = np.array([d.split(' ', 1) for d in databrut])
        
        datax = np.array([d[-1] for d in data])
        
        datay = np.array([d[0] for d in data])
        datay = [re.sub('.*<[0-9]*:[0-9]*:C>', '0', dy) for dy in datay]
        datay = [re.sub('.*<[0-9]*:[0-9]*:M>', '1', dy) for dy in datay]
        
    return one, datax, np.array(datay, int)

||||||| merged common ancestors
path2train = "corpus.tache1.learn.utf8"
path2test = "corpus.tache1.test.utf8"
=======
<<<<<<< HEAD

=======
>>>>>>> 464cffd19d8abf01ab42503fa84ae8e9d6ec25cd
path2train = "corpus.tache1.learn.utf8"
path2test = "corpus.tache1.test.utf8"
>>>>>>> 58e23cdfeb37ebf467a0e34ae394ee6f6f1f198e

class LemmaTokenizer(object):
    """ Lematize the data in WordNet's way
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class SnowballTokenizer(object):
    """ Lematize the data in Snowball's way, the best way actually, cause we
        have the option to specify the language
    """
    def __init__(self):
        self.snowball_stemmer = SnowballStemmer('french')

    def __call__(self, doc):
        return [self.snowball_stemmer.stem(t) for t in word_tokenize(doc)]


def removeCharacters(doc, charac):
    """ Fonction eliminant les caractères présents dans charac
        au document doc
        :param doc: liste de characters
        :return: the new doc
    """
    return doc.translate(str.maketrans(charac, ' ' * len(charac)))
    

def removeNum(doc):
    """ Elimination des chiffres
    """
    return re.sub('[0-9]+', '', doc)


def countCM(path):
    """
    """
    with open(path, 'r') as f:
        d = np.array(f.read().split("\n"))
        chirac = np.where(d == 'C')[0]
        mitterand = np.where(d == 'M')[0]

    return chirac.shape, mitterand.shape


def mycountVectorizer(corpuspp):

    # bags of words: parametrage
    analyzer = u'word'                                            # {‘word’, ‘char’, ‘char_wb’}
    ngram_range = (1,2)                                          # unigrammes
    languages = ['french']
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
            stop_words.append(w)
    lowercase = True
    token = u"[\\w']+\\w\\b"
    max_df=1.0                                                    # default
    # mots apparaissant plus de 5 fois
    
    max_features = 25000
    binary=True                                                   # presence coding
    strip_accents = u'ascii'                                      # {‘ascii’, ‘unicode’, None}

    
    corpus = []

    min_df=5. * 1./len(corpuspp) # on enleve les mots qui apparaissent moins de 5 fois
    vec = txt.CountVectorizer(encoding=u'utf-8', strip_accents=strip_accents, lowercase=lowercase, preprocessor=None, stop_words=stop_words, token_pattern=token, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=None, binary=binary, tokenizer=LemmaTokenizer())
    
    return vec

def myTfidfVectorizer(corpuspp):
    ngram_range = (1, 2)
    min_df = 5. * 1./len(corpuspp)
    token = u"[\\w']+\\w\\b"
    languages = ['french', 'english', 'german', 'spanish']
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
            stop_words.append(w)

    vec = txt.TfidfVectorizer(encoding=u'utf-8', strip_accents=u'ascii', lowercase=True, preprocessor=None, stop_words=stop_words, token_pattern=token, ngram_range=ngram_range, analyzer=u'word', max_df=1.0, min_df=min_df, max_features=25000, vocabulary=None, binary=True, tokenizer=LemmaTokenizer())
    
    return vec


def getCM(datax, datay):
    """ Creates a numpy text data for mitterand sentences and a separated one 
        for chirac's
    """
    indexes_c = np.where(datay == 0)[0]
    indexes_m = np.where(datay == 1)[0]
    data_c, data_m = datax[indexes_c], datax[indexes_m]
    
    return data_c, data_m


def storePredictions(predictions, filename):
    """ Stores the predicted values in a given name file
    """
    cleaned_prediction = []
    for label in prediction:
        cleaned_prediction.append("C" if label == 0 else "M")

    with open(filename, "w") as file:
        file.write("\n".join(cleaned_prediction))
        file.write("\nC\n")


def main():
    """ Fonctions that stores everything we want to call in main.
    """

    path2train = "corpus.tache1.learn.utf8"
    path2test = "corpus.tache1.test.utf8"

    #=========================================== Train

    databrut, datax, datay = readfile(path2train)
    datay = processing_datay(datay)
    all_words = databrut.split()
    vec = mycountVectorizer(all_words)
    nature = txt.CountVectorizer()
    corpus = nature.fit_transform(datax)
    #tfidf = myTfidfVectorizer(all_words)
    #corpus = tfidf.fit_transform(datax)
    X = corpus.toarray()
    y = np.array(datay)
    
    print(X.shape, y.shape)
    
    vocabulary = nature.vocabulary_
    
    #=========================================== Test
    
    databrut_test, datax_t, datay_t = readfile(path2test)
    X_test = nature.transform(datax_t).toarray()
    
    print(X_test.shape)
    
    print(vocabulary)
    #=========================================== Learning

    # SVM
    #clf = svm.LinearSVC()
    
    # Naive Bayes
    clf = nb.MultinomialNB()
    
    # regression logistique
    #clf = lin.LogisticRegression(n_jobs=-1)

    # apprentissage
    clf.fit(X, y) 
    prediction = clf.predict(X_test) # usage sur une nouvelle donnée


    print("verif : ", countCM("SORTIE.txt"))
    print("verif : ", countCM("SORTIE2.txt"))


if __name__=="__main__":
    main()
