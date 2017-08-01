import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import os
import pandas as pd
import pickle
import pprint as pp
import pyprind
import random
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from utilities.common import heading


class MovieDataOutOfCore(object):

    def __init__(self, n_reviews=50000):
        self.n_reviews = n_reviews
        self.dataset_categories = ['test', 'train']
        self.root_dirname = 'aclImdb'
        self.labels_dict = {
            'pos': 1,
            'neg': 0
        }
        self.csv_filename = 'reviews.csv'
        # nltk.download('stopwords')
        self.stop = MovieDataOutOfCore.load_stopwords()

    @staticmethod
    def load_stopwords():
        memory_dirname = 'memory'
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.path.join(dir_path, memory_dirname)
        stopwords_filepath = os.path.join(dir_path, 'stopwords.pkl')
        if not os.path.exists(stopwords_filepath):
            nltk.download('stopwords')
            stop = stopwords.words('english')
            MovieDataOutOfCore.save_object(
                object_=stop,
                filepath=stopwords_filepath
            )
        else:
            # print('Loading stopwords from: ', stopwords_filepath)
            stop = MovieDataOutOfCore.load_object(stopwords_filepath)

        return stop

    @staticmethod
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

    @staticmethod
    def tokenize(text):
        text_clean = MovieDataOutOfCore.clean_text(text)
        return MovieDataOutOfCore.remove_stopwords(text_clean.split())

    @staticmethod
    def tokenize_porter(text):
        text_clean = MovieDataOutOfCore.clean_text(text)
        porter = PorterStemmer()
        return MovieDataOutOfCore.remove_stopwords([
            porter.stem(word) for word in text_clean.split()
        ])

    @staticmethod
    def remove_stopwords(tokens_list):
        stop = MovieDataOutOfCore.load_stopwords()
        return [
            word for word in tokens_list if word not in stop
        ]

    def get_stream(self, csv_filepath=None):
        if not csv_filepath:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            csv_filepath = os.path.join(current_dir, self.csv_filename)

        with open(csv_filepath, 'r') as csv:
            next(csv)  # skip header
            for line in csv:
                # Extract review text and class labels
                line = line.strip()
                review_text = line[:-2]
                class_label = (line[-1])
                yield (review_text, class_label)

    def get_batch(self, stream, size=10):

        reviews = []
        labels = []

        for _ in range(size):
            try:
                (review, label) = next(stream)
            except StopIteration:
                return (reviews, labels)

            reviews.append(review)
            labels.append(label)

        return (reviews, labels)

    @staticmethod
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

    @staticmethod
    def load_object(filepath):
        filepath = os.path.realpath(filepath)
        dirpath = os.path.dirname(filepath)

        file_ = open(filepath, 'rb')
        object_ = pickle.load(
            file=file_
        )
        return object_


def run_training():
    n_reviews = 50000
    movies = MovieDataOutOfCore(n_reviews=n_reviews)

    randomized_csv_filename = 'reviews_randomized.csv'
    current_dir = os.path.dirname(os.path.realpath(__file__))
    csv_filepath = os.path.join(current_dir, movies.csv_filename)
    randomized_csv_filepath = os.path.join(
        current_dir, randomized_csv_filename)

    # Randomize .csv file lines
    with open(csv_filepath, 'r') as source:
        data = [(random.random(), line) for line in source]
    data.sort()
    with open(randomized_csv_filepath, 'w') as target:
        for _, line in data:
            target.write(line)

    print('\nGetting stream.. ', end='')
    review_stream = movies.get_stream(randomized_csv_filepath)
    print('Done')

    hashing_vectorizer = HashingVectorizer(
        decode_error='ignore',
        n_features=2**21,
        preprocessor=None,
        tokenizer=MovieDataOutOfCore.tokenize
    )

    memory_dirname = 'memory'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, memory_dirname)
    classifier_filepath = os.path.join(dir_path, 'classifier.pkl')

    batch_size = 1000
    train_size = 1.0
    train_count = int(train_size * n_reviews)

    if not os.path.exists(classifier_filepath):
        # Train a loggistic regression classifier and pickle it

        classifier = SGDClassifier(
            loss='log',
            random_state=1,
            n_iter=1
        )

        n_batches = int(np.ceil(train_count / batch_size))
        classes = np.array(['0', '1'])

        progress_bar = pyprind.ProgBar(n_batches)

        heading('Partial-fitting the classifer batchwise')

        count = -1

        for _ in range(n_batches):
            (reviews, labels) = movies.get_batch(
                stream=review_stream,
                size=batch_size
            )

            # Vectorize X_train
            X_train = hashing_vectorizer.transform(reviews)
            y_train = labels
            # print(y_train)

            # Partial fit the classifier
            classifier.partial_fit(X_train, y_train, classes=classes)
            count += 1
            progress_bar.update()

        heading('Saving')

        print('Saving classifier: ', classifier_filepath, '.. ', end='')
        MovieDataOutOfCore.save_object(classifier, classifier_filepath)
        print('Done')
    else:
        # Load the classifier from the pickle file
        print('\nLoading classifier from: ', classifier_filepath)
        classifier = MovieDataOutOfCore.load_object(classifier_filepath)

    return classifier


if __name__ == '__main__':
    run_training()
