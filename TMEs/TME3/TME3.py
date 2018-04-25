# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def load(filename):
    listeDoc = list()
    with open(filename, "r") as f:
        doc = list()
        for ligne in f:
            # print "l : ",len(ligne)," ",ligne
            if len(ligne) < 2:  # fin de doc
                listeDoc.append(doc)
                doc = list()
                continue
            mots = ligne.split(" ")
            doc.append((mots[0], mots[1]))
    return listeDoc


def constructDictionnary(tousDocs):
    """ Méthode de référence à base de dictionnaire """

    dico = {}
    for doc in tousDocs:
        for elem in doc:
            if elem[0] not in dico.keys():
                dico[elem[0]] = [elem[1]]
            elif elem[1] not in dico[elem[0]]:
                dico[elem[0]].append(elem[1])

    return dico


def manyTags(dico):
    """ Retourne les mots ayant plusieurs tags """

    mistakes = {}
    for k, v in zip(dico.keys(), dico.values()):
        # Si possède plus de deux tags
        if len(v) > 1:
            mistakes[k] = v

    return mistakes


def evaluateDictionnaryMethod(learningDocs, testDocs):
    """ Evalue la performance sur la base de test
        :param learningDocs: dictionnaire d'apprentissage conçue,
        :param testDocs : base de test à évaluer,
        :return: le pourcentage d'erreurs ainsi que la liste des éléments sur
        lesquels elles se sont produites.
    """

    dicoLearning = constructDictionnary(learningDocs)
    erreur = 0
    for doc in testDocs:
        for elem in doc:
            if elem[0] not in dicoLearning.keys():
                erreur += 1
            elif elem[1] not in dicoLearning[elem[0]]:
                erreur += 1

    nbElem = sum([len(doc) for doc in testDocs])
    print("found :", nbElem - erreur)
    print("nbElem:", nbElem)
    return erreur / nbElem


def learnHMM(allx, allq, N, K, initTo1=True):
    """ Apprend les paramètres d'un modèle HMM par comptage d'une série de
        séquences étiquetée

        :param allx: observations
        [[obs1, ... , obsT], [obs1, ..., obsT], ...]
             Seq 1                 Seq 2        ...
        :param allq: étiquetage
        [[s1, ... , sT], [s1, ..., sT], ...]
             Seq 1            Seq 2        ...
        :param N: nombre d'états
        :param K: nombre d'observations
        :param initTo1: initialisation à 1 (ou epsilon) pour éviter les proba 0
        :return: Pi, A, B
        Les matrices de paramétrage des HMM
    """

    if initTo1:
        eps = 1e-5
        A = np.ones((N, N)) * eps
        B = np.ones((N, K)) * eps
        Pi = np.ones(N) * eps
    else:
        A = np.zeros((N, N))
        B = np.zeros((N, K))
        Pi = np.zeros(N)
    for x, q in zip(allx, allq):
        Pi[int(q[0])] += 1
        for i in range(len(q) - 1):
            A[int(q[i]), int(q[i + 1])] += 1
            B[int(q[i]), int(x[i])] += 1
        B[int(q[-1]), int(x[-1])] += 1  # derniere transition
    A = A / np.maximum(A.sum(1).reshape(N, 1), 1)  # normalisation
    B = B / np.maximum(B.sum(1).reshape(N, 1), 1)  # normalisation
    Pi = Pi / Pi.sum()
    return Pi, A, B


def viterbi(x, Pi, A, B):
    """ Algorithme de Viterbi (en log) pour le décodage des séquences d'états:
        argmax_s p(x, s | lambda)
        :param x: [obs1, ... , obsT] (UNE séquence)
        :param Pi: param HMM
        :param A: param HMM
        :param B: param HMM
        :return: s (la séquence d'état la plus probable), estimation de
        p(x|lambda)
    """

    T = len(x)
    N = len(Pi)
    logA = np.log(A)
    logB = np.log(B)
    logdelta = np.zeros((N, T))
    psi = np.zeros((N, T), dtype=int)
    S = np.zeros(T, int)
    logdelta[:, 0] = np.log(Pi) + logB[:, x[0]]
    # forward
    for t in range(1, T):
        logdelta[:, t] = (logdelta[:, t - 1].reshape(N, 1) +
                          logA).max(0) + logB[:, x[t]]
        psi[:, t] = (logdelta[:, t - 1].reshape(N, 1) + logA).argmax(0)
    # backward
    logp = logdelta[:, -1].max()
    S[T - 1] = logdelta[:, -1].argmax()
    for i in range(2, T + 1):
        S[T - i] = psi[S[T - i + 1], T - i + 1]
    return S, logp  # , delta, psi


###############################################################################
# =============== Proposition de code pour la mise en forme ================= #
###############################################################################

buf = [[m for m, c in d] for d in alldocs]
mots = []
[mots.extend(b) for b in buf]
mots = np.unique(np.array(mots))
nMots = len(mots) + 1  # mot inconnu

mots2ind = dict(zip(mots, range(len(mots))))
mots2ind["UUUUUUUU"] = len(mots)

buf2 = [[c for m, c in d] for d in alldocs]
cles = []
[cles.extend(b) for b in buf2]
cles = np.unique(np.array(cles))
cles2ind = dict(zip(cles, range(len(cles))))

nCles = len(cles)

print("\nThere's ", nMots, " distint words and, ", nCles, " distinct",
      " tags in the dictionary\n")

# mise en forme des données
allx = [[mots2ind[m] for m, c in d] for d in alldocs]
allxT = [[mots2ind.setdefault(m, len(mots)) for m, c in d] for d in alldocsT]

allq = [[cles2ind[c] for m, c in d] for d in alldocs]
allqT = [[cles2ind.setdefault(c, len(cles)) for m, c in d] for d in alldocsT]

###############################################################################
# ==================== Fin de code pour la mise en forme ==================== #
###############################################################################


def evaluateViterbi(X, T):
    """ Evalue la méthode par approche des HMM, avec :
        :param X: allxT, ensemble des observations de la base de test
        :param T: allqT, ensemble des seq d'états ou de tags de la base de test
        :return: le pourcentage de réussite par la méthode de viterbi.
    """

    Pi, A, B = learnHMM(allx, allq, nCles, nMots)
    found = 0
    for x, t in zip(X, T):
        tag = viterbi(x, Pi, A, B)[0]
        found += sum([1 if sx == st else 0 for sx, st, in zip(tag, t)])

    nbElem = sum([len(x) for x in X])
    print("found :", found)
    print("nbElem:", nbElem)
    return found / nbElem


def visualizeTransitionMatrix(
        matrix,
        filename=None,
        localLabsX=cles,
        localLabsY=cles):
    """ Visualisation des matrices de transitions afin de mieux comprendre le
        modèle de language, avec :
        :param matrix: la matrice,
        :param filename: le nom du fichier dans lequel sauvegarder l'image,
        :param localLabsX: liste des POS-TAG,
        :param localLabsY: liste des POS-TAG ou des mots.
    """

    plt.figure()
    plt.imshow(matrix, interpolation='nearest')
    plt.xticks(range(len(localLabsX)), localLabsX, rotation=55)
    plt.yticks(range(len(localLabsY)), localLabsY)  # affichage sur l'image
    # plt.colorbar()
    if filename is not None:
        plt.savefig(filename)

        ###################
        # ------Main----- #
        ###################


# ======== chargement
filename = "data/wapiti/chtrain.txt"  # a modifier
filenameT = "data/wapiti/chtest.txt"  # a modifier
alldocs, alldocsT = load(filename), load(filenameT)


# ======== Evaluate Dictionnary Method
print("Approach by dictionnary :")
percentFailD = evaluateDictionnaryMethod(alldocs, alldocsT)
print("Pourcentage de réussite : ", (1 - percentFailD) * 100, "\n")

# ======== Evaluate HMM Approach
print("Approach by HMM :")
percentFoundV = evaluateViterbi(allxT, allqT)
print("Pourcentage de réussite : ", percentFoundV * 100, "\n")

# ======== Visualisation des marices de transitions
Pi, A, B = learnHMM(allx, allq, nCles, nMots)
#visualizeTransitionMatrix(A, "A")
#visualizeTransitionMatrix(B, "B")
