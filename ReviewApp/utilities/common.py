import math
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import pickle
import re
import scipy.misc


def heading(text, character='='):
    if type(text) is not str:
        heading('<INVALID_HEADING>')
        return
    print('\n')
    print(text)
    print(character * len(text))


def show_error(message):
    return 'ERROR: ' + str(message)


def ensemble_error(n_classifers, error_probability):
    k_start = math.ceil(n_classifers / 2)

    # Probabilites generator
    probabilities = (
        scipy.misc.comb(n_classifers, k)
        * error_probability**k
        * (1.0 - error_probability)**(n_classifers - k)
        for k in range(k_start, n_classifers + 1)
    )

    return sum(probabilities)


def load_stopwords():
    memory_dirname = 'memory'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, memory_dirname)
    stopwords_filepath = os.path.join(dir_path, 'stopwords.pkl')
    if not os.path.exists(stopwords_filepath):
        nltk.download('stopwords')
        stop = stopwords.words('english')
        save_object(
            object_=stop,
            filepath=stopwords_filepath
        )
    else:
        # print('Loading stopwords from: ', stopwords_filepath)
        stop = load_object(stopwords_filepath)

    return stop


def clean_text(text):
    pattern = '<[^>]*>'
    text = re.sub(pattern, '', text)

    emoticons = re.findall(
        '(?::|;|=)(?:-)?(?:\)|\(|D|P)',
        text
    )
    text = (
        re.sub('[\W]+', ' ', text.lower())
        + ''.join(emoticons).replace('-', ''))
    return text


def tokenize(text):
    text_clean = clean_text(text)
    return remove_stopwords(text_clean.split())


def tokenize_porter(text):
    text_clean = clean_text(text)
    porter = PorterStemmer()
    return remove_stopwords([
        porter.stem(word) for word in text_clean.split()
    ])


def remove_stopwords(tokens_list):
    stop = load_stopwords()
    return [
        word for word in tokens_list if word not in stop
    ]


def save_object(object_, filepath):
    filepath = os.path.realpath(filepath)
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    file_ = open(filepath, 'wb')
    pickle.dump(
        obj=object_,
        file=file_,
        protocol=pickle.HIGHEST_PROTOCOL
    )


def load_object(filepath):
    filepath = os.path.realpath(filepath)

    file_ = open(filepath, 'rb')
    object_ = pickle.load(
        file=file_
    )
    return object_
