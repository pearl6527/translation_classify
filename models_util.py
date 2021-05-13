from collections import defaultdict
import numpy as np
import pandas as pd
import nltk, re, pprint, string
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import en_core_web_sm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

stop_words = set(stopwords.words('english'))

def build_info_lists(categs, texts, avg_sent_len, avg_word_len, stops, folder, f, n, entities=None, start=False):
    """
    parameters:
        categs: list for the categories of the texts
        texts: list for the textual content of the texts
        avg_sent_len: list for average sentence length (in words) of the texts
        avg_word_len: list for average word length (in characters) of the texts
        stops: list for percentage of words that are stopwords
        folder: name of or file path to the directory containing the texts
        f: list for the identifiers of the texts
        n: number of texts in the dataset/directory
        entities: (optional) if not None, list for percentage of words that are entities (using spaCy)
        start: (optional) if True, does not skip the first 2 lines of each text file

    return:
        none

    Append information to the lists that are then used to construct the dataframe.
    """
    # load en_core_web_sm of English for entities
    if entities:
        nlp = en_core_web_sm.load()
    
    for i in range(1, n+1):
        data = open(folder+"/"+str(i), encoding="utf8").read()
        f.append(folder+"_"+str(i))

        if not start:
            # remove first 2 lines
            data = data.split('\n')
            data = '\n'.join(data[2:])

        # remove quotation marks
        data = re.sub(r'\s\'|\'\s|’\s', ' ', data)
        data = re.sub(r'"|“|”|\'$|‘|^\'|’$', '', data)

        # count number of entities
        if entities:
            d = nlp(data)
            ec = len(d.ents)

        # avg sent len, word len, stopwords percentage
        sents = sent_tokenize(data)
        num_sents = len(sents)
        wc = 0
        cc = 0
        sc = 0
        for sent in sents:
            tokens = word_tokenize(sent)
            stopwords_x = [w for w in tokens if w in stop_words]
            sc += len(stopwords_x)
            filtered = ''.join(filter(lambda x: x not in '".,;!-–’\'', sent))
            words = [word for word in filtered.split() if word]
            cc += sum(map(len, words))
            wc += len(words)
        cc = cc / wc
        sc = sc / wc
        if entities:
            ec = ec / wc
        wc = wc / num_sents

        texts.append(data)
        categs.append(folder)
        avg_sent_len.append(wc)
        avg_word_len.append(cc)
        stops.append(sc)
        if entities:
            entities.append(ec)

def plot_confusion_matrix(true_label, pred_label, plot_title, save_as):
    """
    parameters:
        true_label: list of correct labels
        pred_label: list of predicted labels
        plot_title: title of plot
        save_as: name of file to save figure to
    
    return:
        none

    Generate a plot of the confusion matrix.
    """
    cm = metrics.confusion_matrix(true_label, pred_label)
    plt.figure(figsize=(6, 6))
    matplotlib.rcParams.update({'font.size': 18})
    sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False, xticklabels=['english', 'translation'], yticklabels=['english', 'translation'])
    plt.ylabel('true label')
    plt.xlabel('predicted label')
    plt.title(plot_title)
    plt.savefig(save_as)

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

def nn_predict(classifier, x_test):
    """
    parameters:
        classifier: a classifier
        x_test: feature matrix of items to apply classifier on
    
    return:
        predictions: list of predictions

    Return predictions as binary array.
    """
    # predict the labels on validation dataset
    predictions = classifier.predict(np.asarray(x_test).astype(np.float32))
    predictions = predictions > 0.5
    
    return predictions

def create_model_architecture(input_size):
    """
    parameters:
        input_size: input size

    return:
        classifier: a classifier
    """

    # input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # hidden layer
    hidden_layer = layers.Dense(180, activation="relu")(input_layer)
    
    # output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return classifier

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

punctuations = string.punctuation

# create list of stopwords
nlp = spacy.load('en_core_web_sm')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# load English tokenizer, tagger, parser, NER and word vectors
parser = English()

def lemma_tokenizer(sentence):
    """
    Tokenize text using lemmas.
    Modified from: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/.
    """
    toks = parser(sentence)

    # lemmatize tokens
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in toks ]

    # return preprocessed list of tokens
    return mytokens

def pos_tokenizer(sentence):
    """
    Tokenize text using POS tags.
    """
    toks = parser(sentence)
    # POS tokens
    POStokens = [ word.pos_.strip() for word in toks ] # + [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in toks ]

    # return preprocessed list of tokens
    return POStokens

