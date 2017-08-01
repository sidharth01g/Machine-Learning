from flask import Flask
from flask import render_template
from flask import request
from sklearn.feature_extraction.text import HashingVectorizer
import numpy as np
import os
import pprint as pp
import pickle
import sqlite3
from utilities import common
from utilities import initial
from wtforms import Form
from wtforms import TextAreaField
from wtforms import validators


def train(review_text, class_label, classifier):
    hashing_vectorizer = HashingVectorizer(
        decode_error='ignore',
        n_features=2**21,
        preprocessor=None,
        tokenizer=common.tokenize
    )
    X = hashing_vectorizer.transform([review_text])
    classifier.partial_fit(X, [class_label])

    return classifier


def classify(review_text, classifier):

    hashing_vectorizer = HashingVectorizer(
        decode_error='ignore',
        n_features=2**21,
        preprocessor=None,
        tokenizer=common.tokenize
    )

    X = hashing_vectorizer.transform([review_text])
    pred = classifier.predict(X)
    y = pred[0]  # probability that the class is 'positive' ( class = 1)

    probability = np.max(classifier.predict_proba(X))

    label = {'1': 'positive', '0': 'negative'}
    return (label[y], probability)


def database_entry(database_file_path, review_text, class_label):
    connection = sqlite3.connect(database_file_path)
    cur = connection.cursor()

    if not os.path.exists(db_file_path):
        connection = sqlite3.connect(database_file_path)
        cur = connection.cursor()
        cur.execute(
            "CREATE TABLE %s (review TEXT, sentiment INTEGER, time_stamp TEXT)"
            % table_name
        )

    cur.execute(
        (
            "INSERT INTO reviews_table (review, sentiment, time_stamp) "
            + "VALUES (?, ?, DATETIME('now'))"
        ),
        (review_text, class_label)
    )
    connection.commit()
    connection.close()


if __name__ == '__main__':
    app = Flask(__name__)
    db_filename = 'reviews.sqlite'
    table_name = 'reviews_table'
    classifier = initial.run_training()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    db_file_path = os.path.join(dir_path, db_filename)

    # app.run(debug=True)
    print(classify('It was very good', classifier))
    print(classify('It was very bad', classifier))

    classifier = train('lame', '0', classifier)

    database_entry(
        database_file_path=db_file_path,
        review_text='Wow', class_label='1')
