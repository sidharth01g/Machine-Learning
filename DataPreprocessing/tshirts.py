import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


class Example(object):

    def __init__(self):
        self.class_label_heading = 'Class Label'
        self.columns = ['Color', 'Size', 'Price', self.class_label_heading]
        self.data = [
            ['Green', 'M', 10.5, 'Class 1'],
            ['Red', 'L', 14.0, 'Class 2'],
            ['Blue', 'XL', 15.6, 'Class 1'],
        ]

        self.df = pd.DataFrame(self.data)
        self.df.columns = self.columns

    def get_dataframe(self):
        return self.df

    def map_ordinal_feature(self, feature_name, mapping):
        if not feature_name:
            raise Exception("Feature name not specified")
        if not mapping:
            raise Exception("Mapping not specified")
        if type(mapping) is not dict:
            raise Exception("Mapping not a dictionary")

        # Map feature specified by 'feature_name' using 'mapping'
        self.df[feature_name] = self.df[feature_name].map(mapping)

    def get_class_mapping(self):
        """Maps class labels to integer values

        """

        class_mapping = {
            label: index for index, label in enumerate(
                np.unique(self.df[self.class_label_heading]))
        }

        return class_mapping

    def map_class_labels(self, mapping=None):
        """If 'mapping' is provided, uses it to map class labels, else uses

        unique label enumeration

        """

        if not mapping:
            mapping = self.get_class_mapping()

        self.df[self.class_label_heading] = (
            self.df[self.class_label_heading].map(mapping)
        )

    def one_hot_encode(self, feature_names_list):
        """Encodes NOMINAL features using one-hot scheme.

        Keyword arguments:
        feature_indices_list -- list of indices of nominal features to
            transform using one-hot encoding

        """
        # X: array of feature vectors from the dataframe
        X = self.df[:].values
        le = LabelEncoder()

        # Get feature_indices_list from feature_names_list
        feature_indices_list = [
            self.columns.index(feature_name) for feature_name in (
                feature_names_list)]
        for index in feature_indices_list:
            # Encode labels corresponding to 'index' to integer values
            X[:, index] = le.fit_transform(X[:, index])

        print("Dataframe after integer labels encoding\n")
        print(X, '\n')

        ohe = OneHotEncoder(categorical_features=feature_indices_list)
        transformed_sparse_matrix = ohe.fit_transform(X)
        print("Transformed (sparse matrix):\n")
        print(transformed_sparse_matrix, '\n')
        transformed_feature_vectors = transformed_sparse_matrix.toarray()
        return transformed_feature_vectors

    @staticmethod
    def get_inverse_mapping(mapping):
        if type(mapping) is not dict:
            raise TypeError("Mapping not a dictionary")

        inverse_mapping = {value: key for key, value in mapping.items()}
        return inverse_mapping


def run():
    ex = Example()

    print("T-shirts dataframe:\n")
    print(ex.get_dataframe(), '\n')

    size_mapping = {
        'XL': 5,
        'L': 4,
        'M': 3
    }

    ex.map_ordinal_feature(feature_name='Size', mapping=size_mapping)
    print("Dataframe after mapping SIZE to integers:\n")
    print(ex.get_dataframe(), '\n')

    """
    inverse_size_mapping = {x: y for y, x in size_mapping.items()}
    ex.map_ordinal_feature(feature_name='Size', mapping=inverse_size_mapping)
    print("Dataframe after inverse-mapping SIZE:\n")
    print(ex.get_dataframe(), '\n')
    """

    print("Printing class label mapping:\n")
    original_mapping = ex.get_class_mapping()
    print(original_mapping, '\n')

    print("Dataframe after mapping class labels:\n")
    ex.map_class_labels()
    print(ex.get_dataframe(), '\n')

    """
    print("Dataframe after restoring original class label mapping:\n")
    inverse_mapping = Example.get_inverse_mapping(original_mapping)
    ex.map_class_labels(mapping=inverse_mapping)
    print(ex.get_dataframe(), '\n')
    """

    features_to_encode_one_hot = ['Color']
    encoded_vectors = ex.one_hot_encode(features_to_encode_one_hot)
    print("One-hot encoded feature vectors (Encoding: %s):\n" %
          str(features_to_encode_one_hot))
    print(encoded_vectors, '\n')


if __name__ == '__main__':
    run()
