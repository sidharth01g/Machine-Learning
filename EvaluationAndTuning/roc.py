import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
from scipy import interp

from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from utils.common import heading
from utils.common import show_error


def interpolate(features, replacement, strategy='mean'):
    features[features == replacement[0]] = np.nan
    imputer = Imputer(missing_values=replacement[1], strategy=strategy, axis=0)
    imputer.fit(features)
    return imputer.transform(features)


def get_data_cancer_sklearn():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    (X_train, X_test, y_train, y_test) = train_test_split(
        X, y, test_size=0.7, random_state=1
    )

    return (X_train, X_test, y_train, y_test)


def main():

    np.set_printoptions(threshold=np.nan)
    np.set_printoptions(linewidth=200)

    try:
        (X_train, X_test, y_train, y_test) = get_data_cancer_sklearn()
    except Exception as error:
        show_error('Getting data failed')
        heading('Exception Trace:')
        raise error

    n_folds = 4
    cross_validator = StratifiedKFold(
        y_train,
        n_folds=n_folds,
        random_state=1
    )

    heading('X_train shape')
    # pp.pprint(X_train)
    pp.pprint(X_train.shape)
    heading('X_test shape')
    # pp.pprint(X_test)
    pp.pprint(X_test.shape)

    feature_indices = [0, 1]
    X_train = X_train[:, feature_indices]

    x_axis = np.linspace(0, 1, 100)

    pipeline = Pipeline(
        [
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=1))
        ]
    )

    tpr_interpolated_cumulative = None

    for index, (train_indices, val_indices) in enumerate(cross_validator):
        pipeline = pipeline.fit(X_train[train_indices], y_train[train_indices])
        # Compute probability that each sample belongs to each of the classes
        # in the set
        probabilities = pipeline.predict_proba(X_train[val_indices])
        heading('Probabilities: Cross-val set %s' % str(index))
        pp.pprint(probabilities)
        pp.pprint(probabilities.shape)
        # pp.pprint(val_indices.shape)
        prediction = pipeline.predict(X_train[val_indices])
        print('Predictions: ', prediction)

        # Compute False Positive Rate, True Positive Rate, Thresholds
        (fpr, tpr, thresholds) = roc_curve(
            y_true=y_train[val_indices],
            y_score=probabilities[:, 1],
            pos_label=1
        )
        heading('Metrics for cross validation set %s' % str(index))
        print('Validation set shape: ', val_indices.shape)
        print('False positive rate (FPR): ', fpr)
        pp.pprint(fpr.shape)
        print('True positive rate (TPR): ', tpr)
        pp.pprint(tpr.shape)
        print('Thresholds: ', thresholds)
        pp.pprint(thresholds.shape)
        print(max(probabilities[:, 1]))

        area_under_curve = auc(fpr, tpr)
        print('Area under the curve: ', area_under_curve)
        """
        plt.plot(
            fpr,
            tpr,
            lw=1,
            label='ROC for fold: %s, AUC: %s' % (
                str(index), str(area_under_curve))
        )
        """
        fpr_interpolated = x_axis
        tpr_interpolated = interp(
            x=x_axis,
            xp=fpr,
            fp=tpr
        )
        if tpr_interpolated_cumulative is None:
            tpr_interpolated_cumulative = tpr_interpolated
        else:
            tpr_interpolated_cumulative += tpr_interpolated
        heading('Interpolated fpr, tpr:')
        print('FPR: ')
        pp.pprint(fpr_interpolated.shape)
        print('TPR: ')
        pp.pprint(tpr_interpolated.shape)
        plt.plot(
            fpr_interpolated,
            tpr_interpolated,
            lw=1,
            label='ROC for fold: %s, AUC: %s' % (
                str(index), str(area_under_curve))
        )

    tpr_interpolated_mean = tpr_interpolated_cumulative / n_folds
    # Make the first sample 0 to make the plot look neat
    tpr_interpolated_mean[0] = 0.0
    area_under_curve = auc(fpr_interpolated, tpr_interpolated_mean)
    plt.plot(
        fpr_interpolated,
        tpr_interpolated_mean,
        lw=3,
        label='Mean ROC, AUC: %s' % str(area_under_curve),
        color='black'
    )

    # Draw random guess line (0,0) to (1,1) on ROC plot
    random_guess_x = [0, 1]
    random_guess_y = [0, 1]
    plt.plot(
        random_guess_x,
        random_guess_y,
        lw=2,
        color='cyan',
        label='Random guessing',
        linestyle='--'
    )

    # Draw ideal ROC: points (0,0), (0,1) and (1,1)
    ideal_x = [0, 0, 1]
    ideal_y = [0, 1, 1]
    plt.plot(
        ideal_x,
        ideal_y,
        lw=2,
        color='green',
        label='Ideal classifier ROC',
        linestyle=':'
    )

    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristics')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    main()
