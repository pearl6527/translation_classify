import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from models_util import build_info_lists, create_model_architecture, nn_predict, lemma_tokenizer, pos_tokenizer

categs = []             # 'translation' or 'english'
texts = []              # actual text content
avg_sent_len = []       # average sentence length
avg_word_len = []       # average word length
stops = []              # percentage of stopwords
f = []                  # identifier
# entities = []

trainDF = pd.DataFrame()

# translated texts
build_info_lists(categs, texts, avg_sent_len, avg_word_len, stops, "translation", f, 100) #, entities)

# english texts
build_info_lists(categs, texts, avg_sent_len, avg_word_len, stops, "english", f, 83) #, entities)

# store in dataframe
trainDF['file'] = f
trainDF['text'] = texts
trainDF['categ'] = categs
trainDF['avg_sent'] = avg_sent_len
trainDF['avg_word'] = avg_word_len
trainDF['stopwords'] = stops
# trainDF['entities'] = entities

trainDF['label'] = trainDF['categ'].apply(lambda x: 0 if x == "english" else 1)

# make the relevant columns of the dataframe into a feature matrix
feat_v = trainDF.drop(['categ', 'label'], axis=1).to_numpy()
trainDF.drop(['categ', 'label'], axis=1)

# create a count matrix (bag-of-words / ngram features)
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1, 1))
cv = count_vect.fit(trainDF['text'])
xtrain_count = count_vect.transform(trainDF['text'])
word_v = xtrain_count.toarray()

# tf-idf vectors as features (top 5000 2-/3-grams)
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000, ngram_range=(2, 3))
tv = tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(trainDF['text'])
word_tv = xtrain_tfidf.toarray()

# concatenate these ngram features with the other features we made in the dataframe
full_v = np.concatenate((feat_v, word_v, word_tv), axis=1)
count_ind = feat_v.shape[1]
tfidf_ind = count_ind + word_v.shape[1]

# split the examples into train & test (validation) sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(full_v, trainDF['label'], random_state=1)

# NB on custom + BOW features
nb = naive_bayes.MultinomialNB()
nb.fit(X_train[:, 2:tfidf_ind], y_train)

# NB on TF-IDF features
nb_tfidf = naive_bayes.MultinomialNB()
nb_tfidf.fit(X_train[:, tfidf_ind:], y_train)

# logistic regression on custom + BOW features
lm = linear_model.LogisticRegression()
lm.fit(X_train[:, 2:tfidf_ind], y_train)

# SVM on custom + BOW features
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(X_train[:, 2:tfidf_ind], y_train)

# simple neural network on TF-IDF features
classifier = create_model_architecture(X_train[:, tfidf_ind:].shape[1])
classifier.fit(np.asarray(X_train[:, tfidf_ind:]).astype(np.float32), y_train, epochs=150)

def prepare_new(folder, n, test_label, start=False):
    """
    parameters:
        folder: name of or file path to the directory containing the texts
        n: number of texts in the dataset/directory
        test_label: list of correct labels for the texts
        start: (optional) if True, does not skip the first 2 lines of each text file
    
    return:
        df: dataframe containing information about each item in new dataset
        test_v: feature matrix for new dataset
    
    Prepare new texts for classification.
    """

    categs = []             # 'translation' or 'english'
    texts = []              # actual text content
    avg_sent_len = []       # average sentence length
    avg_word_len = []       # average word length
    stops = []              # percentage of stopwords
    f = []                  # idenfitier
    # entities = []

    df = pd.DataFrame()
    build_info_lists(categs, texts, avg_sent_len, avg_word_len, stops, folder, f, n, start) #, entities)

    df['file'] = f
    df['text'] = texts
    df['avg_sent'] = avg_sent_len
    df['avg_word'] = avg_word_len
    df['stopwords'] = stops
    df['label'] = test_label
    # df['entities'] = entities
    feat_test_v = df.drop(['label'], axis=1).to_numpy()

    xtrain_count = count_vect.transform(df['text'])
    word_test_v = xtrain_count.toarray()
    xtrain_tfidf =  tfidf_vect.transform(df['text'])
    word_test_tv = xtrain_tfidf.toarray()

    test_v = np.concatenate((feat_test_v, word_test_v, word_test_tv), axis=1)

    return df, test_v

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

# tokenize with spaCy
bow_vector = CountVectorizer(tokenizer = pos_tokenizer, ngram_range=(1, 1))
tfidf_vector = TfidfVectorizer(tokenizer = lemma_tokenizer, ngram_range=(1, 1))

class predictors(TransformerMixin):
    """
    Custom transformer using spaCy
    Taken from: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/.
    """
    def transform(self, X, **transform_params):
        # clean Text
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

def clean_text(text):
    """
    Basic function to clean text
    Taken from: https://www.dataquest.io/blog/tutorial-text-classification-in-python-using-spacy/.
    """
    return text.strip().lower()

# BOW features with spaCy tokenization (using POS tags)
bow_spacy = linear_model.LogisticRegression()
bow_pipe = Pipeline([("cleaner", predictors()), ('vectorizer', bow_vector), ('classifier', bow_spacy)])
bow_pipe.fit(X_train[:, 1], y_train)

# TF-IDF features with spaCy tokenization (using lemmatized words)
tfidf_spacy = linear_model.LogisticRegression()
tfidf_pipe = Pipeline([("cleaner", predictors()), ('vectorizer', tfidf_vector), ('classifier', tfidf_spacy)])
tfidf_pipe.fit(X_train[:, 1], y_train)

def classify_new(test_v, test_y):
    """
    parameters:
        test_v: feature matrix to apply models on
        test_y: correct labels to check against

    return:
        models_df: dataframe containing accuracy, precision, recall statistics for each model
        texts_df: dataframe containing each model's prediction on each item in test dataset

    Apply each model to dataset.
    """
    nb_pred = nb.predict(test_v[:, 2:tfidf_ind])
    nb_tfidf_pred = nb_tfidf.predict(test_v[:, tfidf_ind:])
    lm_pred = lm.predict(test_v[:, 2:tfidf_ind])
    svm_pred = SVM.predict(test_v[:, 2:tfidf_ind])
    nn_pred = nn_predict(classifier, np.asarray(test_v[:, tfidf_ind:]).astype(np.float32))
    spacy_bow_pred = bow_pipe.predict(test_v[:, 1])
    spacy_tfidf_pred = tfidf_pipe.predict(test_v[:, 1])

    models_df = pd.DataFrame()
    ms = {'NB, custom + BOW': nb_pred, 'NB, TF-IDF': nb_tfidf_pred, 'LR, custom + BOW': lm_pred, 'SVM, custom + BOW': svm_pred, \
        'Neural Net, TF-IDF': nn_pred, 'spaCy, BOW (LR)': spacy_bow_pred, 'spaCy, TF-IDF (LR)': spacy_tfidf_pred}
    
    models_df['model'] = ms.keys()
    accs = []
    pres = []
    recs = []
    for m, pred in ms.items():
        print(m)
        accs.append(metrics.accuracy_score(test_y, pred))
        pres.append(metrics.precision_score(test_y, pred))
        recs.append(metrics.recall_score(test_y, pred))
    models_df['accuracy'] = accs
    models_df['precision'] = pres
    models_df['recall'] = recs

    true_labels = []
    for i in test_y:
        if i == 1:
            true_labels.append('translation')
        else:
            true_labels.append('english')
    tests_df = pd.DataFrame({'categ': true_labels})
    for m, pred in ms.items():
        tests_df[m] = pred
    tests_df['file'] = test_v[:, 0]
    tests_df['text'] = test_v[:, 1]

    return models_df, tests_df

def get_validation_set():
    """
    parameters:
        none

    return:
        X_test: feature matrix of validation set
        y_test: correct labels of validation set
    
    Return feature matrix and list of correct labels of validation set used to develop models.
    """
    return X_test, y_test
