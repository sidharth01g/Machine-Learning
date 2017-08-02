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

classifier = initial.run_training()
app = Flask(__name__)
db_filename = 'reviews.sqlite'
table_name = 'reviews_table'
dir_path = os.path.dirname(os.path.realpath(__file__))
db_file_path = os.path.join(dir_path, db_filename)


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
    global classifier

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


class ReviewForm(Form):
    movie_review = TextAreaField(
        '',
        [
            validators.DataRequired(),
            validators.length(min=5)
        ]
    )

app = Flask(__name__)


@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    global classifier
    form = ReviewForm(request.form)
    if request.method == 'POST' and form.validate():
        review = request.form['movie_review']
    (y, probability) = classify(review_text=review, classifier=classifier)

    return render_template(
        'results.html',
        content=review,
        prediction=y,
        probability=round(probability * 100, 2)
    )


@app.route('/thanks', methods=['POST'])
def feedback():
    global classifier, db_file_path
    feedback = request.form['feedback_button']
    review = request.form['review']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = str(not(y))

    train(review_text=review, class_label=y, classifier=classifier)

    database_entry(
        database_file_path=db_file_path,
        review_text=review,
        class_label=y
    )
    return render_template('thanks.html')


if __name__ == '__main__':

    app.run(debug=True)
    """
    print(classify('It was very good', classifier))
    print(classify('It was very bad', classifier))

    classifier = train('lame', '0', classifier)

    database_entry(
        database_file_path=db_file_path,
        review_text='Wow', class_label='1')
    """
