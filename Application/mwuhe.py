# -*- coding: utf-8 -*-

import nltk
import numpy as np
from nltk.corpus import brown

# Pour changer le path du nltk_data, très très important
nltk.data.path.append("/Infos/nltk/nltk_data")


def sents_categories(*categories):
    if not len(categories):
        return brown.sents(), brown.tagged_sents()
    return brown.sents(
        categories=categories), brown.tagged_sents(
        tagset=categories)


def words_categories(*categories):
    if not len(categories):
        return brown.words(), brown.tagged_words()
    return brown.words(
        categories=categories), brown.tagged_words(
        tagset=categories)


def ngrammes(words_text, n):
    """  Transforme le corpus en ngrams
    """
    ngrams, d = [], words_text
    ngrams.append([[''.join(d[i:j]) for j in range(
        i + 1, min(i + n + 1, len(d) + 1))] for i in range(len(d))])

    return ngrams


def words_ngrams():

    pass


if __name__ == "__main__":
    words_text, tags_text = words_categories()
    all_ngrams = ngrammes(words_text, 2)
    print(all_ngrams[0])
