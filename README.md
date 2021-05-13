# An Exploration of Models That Differentiate Translationese from Original English Texts

In this project, we seek to build machine learning models that can differentiate between translated texts and non-translated texts in English using previously hypothesized characteristics of translated texts, which were coined “translationese.” We succeeded in building seven models that are combinations of different classifiers and features that seem promising in their ability to distinguish between these two types of texts. Our best model is a simple neural network trained on TF-IDF features and reaches 86.7% accuracy, 82.4% precision, and 93.3% recall performance on our test set. Our success in building these models affirms the hypothesis that distinct features exist within translated texts, and also supports the fact that these features are apparent enough for simple machine learning models to distinguish between translated and non-translated texts.

## Dependencies

* ```scikit-learn 0.24.0```
* ```pandas 1.2.1```
* ```nltk 3.5```
* ```matplotlib 3.3.3```
* ```numpy 1.19.2```
* ```spacy 2.3.5```
    * ```en-core-web-sm 2.3.1```
* ```keras 2.4.3```
* ```tensorflow 2.4.1```
* ```seaborn 0.11.1``` (for displaying confusion matrices if desired)

## Usage

You can interact with our implementation through a Jupyter Notebook, as in ```classify_texts.ipynb```.

### Import the relevant modules
```python
import pandas as pd
import numpy as np
from classify_trans import prepare_new, classify_new, get_validation_set
from models_util import plot_confusion_matrix
```

```get_validation_set()``` returns the feature matrix (```X_valid```) and the labels (```y_valid```) of the validation set we used to develop our models.

```python
X_valid, y_valid = get_validation_set()
```

```classify_new()``` applies all models to the given feature matrix, compares them against the correct labels, and returns two dataframes:
* ```models``` shows the accuracy, precision, and recall of each model when applied to the dataset
* ```texts``` shows the output of each model on each item in the dataset

```python
models, texts = classify_new(X_valid, y_valid)
```

For example, ```models``` might look like the following:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>accuracy</th>
      <th>precision</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NB, custom + BOW</td>
      <td>0.869565</td>
      <td>0.806452</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NB, TF-IDF</td>
      <td>0.782609</td>
      <td>0.714286</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LR, custom + BOW</td>
      <td>0.804348</td>
      <td>0.833333</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SVM, custom + BOW</td>
      <td>0.804348</td>
      <td>0.807692</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Neural Net, TF-IDF</td>
      <td>0.913043</td>
      <td>0.888889</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>spaCy, BOW (LR)</td>
      <td>0.652174</td>
      <td>0.645161</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>spaCy, TF-IDF (LR)</td>
      <td>0.782609</td>
      <td>0.727273</td>
      <td>0.96</td>
    </tr>
  </tbody>
</table>
</div>

### Classify new texts

Please save all text files in a single directory. Note that texts must be named ```1```, ```2```, ..., ```99```, ```100```, ```101```, ... That is, the file names of the texts must be numbered from 1 up to however many texts there are. The numbering must be consecutive (e.g., no jumping from file ```5``` to ```7```).

```prepare_new()``` takes in the path to the directory in which we store our texts, an integer indicating the number of files in the folder, and a list of correct labels (```1``` if a translation, ```0``` otherwise) in corresponding order. By default, ```prepare_new()``` **skips the first two lines of each file**, because we formatted our texts such that each file starts with a line containing the source language (if a translation) and the URL of the article, followed by a blank line, followed by the actual text. Set the optional parameter ```start=True``` to not skip the first two lines.

```python
test_y = [0]
df, test_v = prepare_new("test", 1, test_y)  # prepare_new("test", 1, test_y, start=True)
```

```df``` is a dataframe that contains the following information for each text:
* text content (```df['text']```)
* average sentence length (in words) (```df['avg_sent']```)
* average word length (in characters) (```df['avg_word']```)
* percentage of stopwords in text (```df['stopwords']```)
* correct label (```0``` or ```1```) (```df['label']```)

```test_v``` is a feature matrix in a format compatible for our models.

Classify the texts with ```classify_new()```.

```python
test_models, test_texts = classify_new(test_v, test_y)
```

Below is ```test_texts``` on a dataset containing a single article, with file name ```1```, in the directory ```test```.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual_label</th>
      <th>NB, custom + BOW</th>
      <th>NB, TF-IDF</th>
      <th>LR, custom + BOW</th>
      <th>SVM, custom + BOW</th>
      <th>Neural Net, TF-IDF</th>
      <th>spaCy, BOW (LR)</th>
      <th>spaCy, TF-IDF (LR)</th>
      <th>file</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>english</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>0</td>
      <td>test_01</td>
      <td>This article fell in our laps completely by su...</td>
    </tr>
  </tbody>
</table>
</div>


\
\
Pearl Hwang, Cynthia Lin
Yale University
LING 227: Language & Computation
May 2021
