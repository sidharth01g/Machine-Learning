import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import os
import pandas as pd
import pprint as pp
import pyprind
import re
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

from utils.common import heading


class MovieData(object):

    def __init__(self, n_reviews=50000):
        self.n_reviews = n_reviews
        self.dataset_categories = ['test', 'train']
        self.root_dirname = 'aclImdb'
        self.labels_dict = {
            'pos': 1,
            'neg': 0
        }
        self.csv_filename = 'reviews.csv'
        nltk.download('stopwords')

    def read_data(self):

        progress_bar = pyprind.ProgBar(self.n_reviews)

        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_filepath = os.path.join(current_dir, self.csv_filename)
        if os.path.exists(csv_filepath):
            print('Reading dataset from ', csv_filepath, '..')
            df = pd.read_csv(csv_filepath)
        else:

            df = pd.DataFrame()

            for dataset in self.dataset_categories:
                for label in self.labels_dict.keys():
                    path = os.path.abspath(
                        current_dir + '/' + 'movie_review_dataset' + '/'
                        + self.root_dirname + '/' + dataset + '/' + label)
                    if not os.path.exists(path):
                        raise Exception(path + ' does not exist')

                    for file_ in os.listdir(path):
                        file_path = os.path.join(path, file_)
                        with open(file_path, 'r') as infile:
                            text = infile.read()
                        df = df.append(
                            [
                                [
                                    text,
                                    self.labels_dict[label]
                                ]

                            ],
                            ignore_index=True
                        )
                        progress_bar.update()
            print('Writing to file ', csv_filepath, '..')
            df.to_csv(csv_filepath, index=False)

        df.columns = ['review', 'sentiment']
        np.random.seed(0)
        df = df.reindex(np.random.permutation(df.index))

        self.df = df
        return df

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
        return text.split()

    @staticmethod
    def tokenize_porter(text):
        porter = PorterStemmer()
        return [
            porter.stem(word) for word in text.split()
        ]

    @staticmethod
    def remove_stopwords(tokens_list):
        stop = stopwords.words('english')
        return [
            word for word in tokens_list if word not in stop
        ]


def main():
    movies = MovieData(n_reviews=50000)
    print('Reading data..')
    movies.read_data()
    review_text_array = np.asarray(movies.df.values[:, 0])
    heading('Data:')
    pp.pprint(review_text_array)
    pp.pprint(review_text_array.shape)

    """
    # Clean up review text
    heading('Cleaning up review text')
    movies.df['review'] = movies.df['review'].apply(MovieData.clean_text)
    review_text_array = np.asarray(movies.df.values[:, 0])
    pp.pprint(review_text_array)
    """
    """
    print('*' * 100)
    pp.pprint(movies.df.loc[:, 'sentiment'].values.shape)
    pp.pprint(movies.df.loc[:, 'review'].values.shape)
    print('*' * 100)
    exit()
    """
    (X_train, X_test, y_train, y_test) = train_test_split(
        movies.df.loc[:, 'review'].values,
        movies.df.loc[:, 'sentiment'].values,
        train_size=0.1,
        random_state=1
    )

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    stop = stopwords.words('english')

    vectorizer = TfidfVectorizer(
        strip_accents=False,
        lowercase=False,
        preprocessor=None
    )

    classifer = LogisticRegression(
        random_state=0
    )

    pipe = Pipeline(
        [
            ('vectorizer', vectorizer),
            ('classifer', classifer)
        ]
    )

    c_values = [10 ** c for c in range(3)]

    param_grid = [

        {
            'vectorizer__ngram_range': [(1, 1)],
            'vectorizer__stop_words': [stop, None],
            'vectorizer__tokenizer': [
                MovieData.tokenize, MovieData.tokenize_porter],

            'classifer__penalty': ['l1', 'l2'],
            'classifer__C': c_values
        },

        {
            'vectorizer__ngram_range': [(1, 1)],
            'vectorizer__stop_words': [stop, None],
            'vectorizer__tokenizer': [
                MovieData.tokenize, MovieData.tokenize_porter],
            'vectorizer__use_idf': ['False'],
            'vectorizer__norm': [None],

            'classifer__penalty': ['l1', 'l2'],
            'classifer__C': c_values
        }

    ]

    k = 5
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='accuracy',
        cv=k,
        verbose=1,
        n_jobs=-1
    )

    heading('Fitting GridSearchCV')
    gs.fit(X_train, y_train)
    print('\nBest parameters:\n')
    pp.pprint(gs.best_params_)
    print('\nCross validation accuracy: ', gs.best_score_)
    best_estimator = gs.best_estimator_
    print('\nBest estimator accuracy: ', best_estimator.score(X_test, y_test))


if __name__ == '__main__':
    main()
