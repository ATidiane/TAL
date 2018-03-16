# -*- coding: utf-8 -*-

import string
import unicodedata
import re
import nltk.corpus.reader as pt
from collections import *
import sklearn.feature_extraction.text as txt
from nltk.corpus import stopwords


path2datatrain = u'/users/nfs/Etu4/3502264/Documents/TAL/TMEs/TME6/20news-bydate-train/'
path2datatest = u'/users/nfs/Etu4/3502264/Documents/TAL/TMEs/TME6/20news-bydate-test/'


def decorator_unicode(fonc):
    def unicode_vec(doc):
        return unicode(doc, 'utf-8')
    return unicode_vec

def characters(doc, charac):
    """ Fonction eliminant les caractères présents dans charac
        au document doc
        :param doc: liste de characters
        :return: the new doc
    """
    return doc.translate(str.maketrans(charac, ' ' * len(charac)))

@decorator_unicode
def normalize(doc):
    """ Elimination des accents, normalisation du texte 
    """
    return  unicodedata.normalize('NFD', doc).encode('ascii', 'ignore').decode("utf-8")

def miniscule(doc):
    """ Transforme la chaine de caractères doc en miniscule
    """
    return doc.lower()

def removeNum(doc):
    """ Elimination des chiffres
    """
    return re.sub('[0-9]+', '', doc) # remplacer une séquence de chiffres par rien

def removeUrl(doc):
    """ Elimination des URLs
    """
    return re.sub('www[a-z0-9\.\/:%_+.#?!@&=-]+', 'URL', doc)

def removeSpaces(doc):
    """ Suppression d'espaces multiples
    """
    return re.sub(' +', ' ', doc)

def removeHeader(doc):
    """ Renvoie l'indice après un double saut de ligne,
        Supression d'entête
    """
    return [doc.find('\n\n')]

def categorize(path):
    """
    """
    rdr = pt.CategorizedPlaintextCorpusReader(path, '.*/[0-9]+', encoding='latin1', cat_pattern='([\w\.]+)/*')
    docs = [[rdr.raw(fileids=[f]) for f in rdr.fileids(c) ] for c in rdr.categories()]
    
    return docs

def nbUnique(corpuspp):
    """ Renvoie le nombre de mot unique d'un corpus
        :param corpuspp:attention! doit être déjà splité(corpuss[0][0].split())
    """
    return len(set(corpuspp))

def countWords(corpuspp):
    """ Compte le nombre d'occurences de chaque mot dans le document
        :param corpuspp: le corpus
        :return: retourne le dico ayant comme clé le mot et valeur, son nombre
        d'occurences
    """
    dico = Counter()
    for topic in corpuspp:
        for d in topic:
            for mot in d.split():
                # création ou incrément
                dico[mot] += 1 

    return dico

def countDocs(corpuspp):
    """Compte le nombre de documents dans lequel est présent chaque mot
        :param corpuspp: le corpus
        :return: retourne le dico ayant comme clé le mot et valeur, le nombre
        de documents dans lequel il est présent, ainsi que le nb de documents 
        au total
    """
    dico, nbDocs = Counter(), 0
    for topic in corpuspp:
        for d in topic:
            nbDocs += 1
            for mot in set(d.split()):
                dico[mot] += 1

    return dico, nbDocs

def unigrammes(corpuspp):
    """ Transforme le corpus en unigrammes
    """
    return [[d.split() for d in topic] for topic in corpuspp]

def bigrammes(corpuspp):
    """ Transforme le corpus en bigrammes
    """
    bigrams = []
    for i, topic in enumerate(corpuspp):
        bigrams.append([])
        for d in topic:
            d = d.split()
            bigrams[i].append([m1+'-'+m2 for m1,m2 in zip(d[:-1], d[1:])])

    return bigrams

def ngrammes(corpuspp, n):
    """  Transforme le corpus en ngrams
    """
    ngrams = []
    for t, topic in enumerate(corpuspp):
        ngrams.append([])
        for d in topic:
            d = d.split()
            ngrams[t].append([[''.join(d[i:j]) for j in range(i+1,min(i+n+1,len(d)+1))] for i in range(len(d))])
            
    return ngrams

def filtrage(dico, nb, frequence):
    """ Supprime les mots qui apparaissent dans moins de nb documents
        ou dans plus de 60% des documents
    """
    ndocs = sum([len(t) for t in corpuspp])
    dicoFilt = [k for k,v in dico.items() if v>nb and v<frequence*ndocs]
    return dicoFilt

def countVectorizer(corpuspp):

    # bags of words: parametrage
    analyzer = u'word'                                            # {‘word’, ‘char’, ‘char_wb’}
    ngram_range = (1, 1)                                          # unigrammes
    languages = ['french', 'english', 'german', 'spanish']
    stop_words = []
    for l in languages:
        for w in stopwords.words(l):
            stop_words.append(w)
    lowercase = True
    token = u"[\\w']+\\w\\b" 
    max_df=1.0                                                    # default
    # mots apparaissant plus de 5 fois
    
    max_features = 10000
    binary=True                                                   # presence coding
    strip_accents = u'ascii'                                      # {‘ascii’, ‘unicode’, None}

    
    corpus = []
    for t, topic in enumerate(corpuspp):
        corpus.append([])
        for d in topic:
            min_df=5. * 1./len(d.split()) # on enleve les mots qui apparaissent moins de 5 fois
            vec = txt.CountVectorizer(encoding=u'utf-8', charset=None,  strip_accents=strip_accents, lowercase=lowercase, preprocessor=None, stop_words=stop_words, token_pattern=token, ngram_range=ngram_range, analyzer=analyzer, max_df=max_df, min_df=min_df, max_features=max_features, vocabulary=None, binary=binary)
    
            corpus[t].append(vec.fit_transform(d.split()).tocsr())

    return corpus




corpuspp = categorize(path2datatrain)
print(corpus[0:50])

#countingWord, word = countWords(corpuspp), 'the'
#print("Nombre d'occurences du mot {}: ".format(word), countingWord[word])

#countingDocs, nbDocs = countDocs(corpuspp)
#print("Nombre de corpus dans lequel apparait le mot {}: ".format(word),
#      countingDocs['the'])

#print("Nombre de documents total: ", nbDocs)

#dicoFilt = filtrage(countingDocs, 3, 0.6)

#print('the' in dicoFilt) # False
#print('human' in dicoFilt) # True


#print(unigrammes(corpuspp))

#print(bigrammes(corpuspp)[0][0])

#print(ngrammes(corpuspp, 4)[0][0])

#====================== Affichage d'un beau texte
#print(corpuspp[0][0])


#========================= Utilisation des outils de sklearn
#count_vect = txt.CountVectorizer()
#X_train_counts = count_vect.fit_transform(corpuspp[0][0].split())

print(countVectorizer(corpuspp)[0][0])

