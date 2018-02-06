# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:58:14 2018

@author: 3502264
"""

import io
import nltk
import numpy as np

#tokenizer = nltk.data.load('/Infos/nltk/nltk_data/tokenizers/punkt/english.pickle')

###############################################################################
#---------------------Différentes fonctions testées en nltk-------------------#
###############################################################################

# Pour changer le path du nltk_data, très très important 

nltk.data.path.append("/Infos/nltk/nltk_data")

def read_file_txt(fichier):
    """Lit un fichier texte et renvoie une liste des phrases qui y sont"""
    f = io.open(fichier, 'rU', encoding='utf-8')

    sentences = []    
    for line in f.readlines():
        sentences.append(line)
        
    return sentences
    
def read_all_files():
    """Lit toutes les lignes de tous les fichiers et renvoie une liste contenant
    ces différentes lignes"""
    files = []
    for i in range(1,6):
        file0X = read_file_txt("tbbt/s3/txt/tbbts03e0{}.txt".format(i))
        files.extend(file0X)
        
    return files
    
def tokenize_in_word(sentence):
    """Découper une chaine de caractère en mots"""
    return nltk.word_tokenize(sentence)
    

def tokenize_sentences_in_word(liste_sentences):
    """Tokenize une liste de chaine de caractères et renvoie une liste de liste
    tokénisés"""
    list_sentences = []
    for sentence in liste_sentences:
        list_sentences.append(tokenize_in_word(sentence))
    
    return list_sentences
    
def tokenize_sentences(liste_sentences):
    """Détecte la fin d'une phrase"""
    list_sentences = []
    for sentence in liste_sentences:
        list_sentences.append(nltk.sent_tokenize(sentence))
    
    return np.array(list_sentences)
    
def analyse_morpho_syntaxique(tokens):
    """Analyse morpho syntaxique d'une phrase, détecte les adjectifs, les pronoms
    , etc... et les attribut une étiquette"""
    return nltk.pos_tag(tokens)
    
def analyse_morpho_syntaxique_list_tokens(list_tokens):
    """Analyse morpho syntaxique d'une liste de phrases"""
    analyse = []
    for token in list_tokens:
        analyse.extend(analyse_morpho_syntaxique(token))
        
    return analyse
    

def recognize_named_entity(tokens):
    tagged = ()
    if (tokens[0] is list):
        tagged = analyse_morpho_syntaxique_list_tokens(tokens)
    else:
        tagged = analyse_morpho_syntaxique(tokens)
        
    return nltk.ne_chunk(tagged)


###############################################################################
#--------------Objectifs les qualificatifs de Shelbon Cooper------------------#
###############################################################################

def qualificatifs_Shelbon_Cooper(files):
   tokens = tokenize_sentences_in_word(files)
   return recognize_named_entity(tokens)
   
###############################################################################
#-------------------------------------Main------------------------------------#
###############################################################################

if __name__=="__main__":

    sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""
    tokeniser = tokenize_in_word(sentence)
    #print(tokeniser,"\n")
    
    file01 = read_file_txt("tbbt/s3/txt/tbbts03e01.txt")
    list_tokeniser = tokenize_sentences_in_word(file01)
    #print(list_tokeniser,"\n")
    
    tokenize_phrases = tokenize_sentences(file01)
    #print(tokenize_phrases,"\n")
    
    analyse = analyse_morpho_syntaxique(tokenize_in_word(sentence))    
    #print(analyse)
    
    files = read_all_files()
    #print(files)
    
    tokenize_files = tokenize_sentences_in_word(files)
    print(tokenize_files)
    
    recognize_named_entity(tokenize_in_word(sentence))
    
    print(qualificatifs_Shelbon_Cooper(files))
    
#tagged = nltk.pos_tag(tokens)
