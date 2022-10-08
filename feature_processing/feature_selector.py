import pandas as pd
import feature_processing.feature_processor as fp
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest
import sklearn.feature_selection as fs
from sklearn.pipeline import make_pipeline


class FeatureSelector:
    """
    Provides recursive feature elimination and select k best feature implementations for given training and testing
    datasets.
    """

    def __init__(self, training_data, testing_data):
        """
        :param training_data: the training data with all features and labels
        :param testing_data: the testing data with all features and labels
        """
        self.features = training_data.drop('HOME_TEAM_WINS', axis=1)
        self.labels = training_data['HOME_TEAM_WINS']

        self.test_features = testing_data.iloc[:, :-1]
        self.test_labels = testing_data.iloc[:, -1]

        self.scaling_functions = [fp.min_max_scale_features, fp.standard_scale_features, fp.max_abs_scale_features,
                                  fp.robust_scale_features, fp.power_transform_scale_features,
                                  fp.quantile_scale_features, fp.quantile_2_scale_features, fp.normalise_scale_features]

    def select_k_best(self, k=10, score_func=fs.f_classif, verbose=False):
        """
        Selects k best features for the give score function

        :param k the number of features to select
        :param score_func the score function to use to score the features
        :param verbose a boolean to determine if the function prints its results
        :return: the selected features' column names
        """
        selector = SelectKBest(score_func=score_func, k=k).fit(self.features, self.labels)
        selected_features = self.features.iloc[:, selector.get_support()].columns

        if verbose:
            print(selector.pvalues_)
            print(f"Selected {k} best features with {score_func.__name__}:")
            print(selected_features)

        return selected_features

    def recursive_feature_selection(self, estimator=LogisticRegression(class_weight='auto', max_iter=900, C=5),
                                    verbose=False):
        """
        Implements recursive feature elimination for the given estimator

        :param estimator the estimator used for RFECV and accuracy score
        :param verbose a boolean to determine if the function prints its results
        :return: the selected features' column names
        """
        logistic_pipeline = make_pipeline(StandardScaler(), estimator)
        scaled_features = StandardScaler().fit_transform(self.features)
        selector = RFECV(estimator, cv=5).fit(scaled_features, self.labels)

        selected_features = pd.DataFrame(self.features).iloc[:, selector.support_]

        model = logistic_pipeline.fit(selected_features, self.labels)
        test_features = pd.DataFrame(self.test_features).iloc[:, selector.support_]

        y_pred = model.predict(test_features)
        accuracy = accuracy_score(y_true=self.test_labels, y_pred=y_pred)

        model = logistic_pipeline.fit(self.features, self.labels)
        y_pred_all_features = model.predict(self.test_features)
        accuracy_all_features = accuracy_score(y_true=self.test_labels, y_pred=y_pred_all_features)

        if verbose:
            print(f'Model Accuracy with RFECV: {accuracy}')
            print(f"Params: {selector.get_params()}")
            print(f'Model Accuracy with all features: {accuracy_all_features}')
            print(f"Selected Columns = {selected_features.columns}")

        return selected_features.columns

    def test_with_all_scaling_methods(self, estimator=LogisticRegression(class_weight='auto', max_iter=900, C=5)):
        """
        Calculates and prints the accuracy of the given estimator for all scaling functions in feature_processor

        :param estimator the estimator used for accuracy score
        :return: None
        """
        for fn in self.scaling_functions:
            estimator.fit(fn(self.features), self.labels)
            y_pred = estimator.predict(fn(self.test_features))
            print(f'Model Accuracy with {fn.__name__} : {accuracy_score(y_true=self.test_labels, y_pred=y_pred)}')

    def get_k_best_train_test_split(self, k=10):
        """
        Calculates the k best features for the training data and returns the training and testing datasets with just the
        k best features selected

        :param k the number of features to select
        :return: the training and test feature values for the k best features
        """
        feature_cols = self.select_k_best(k=k)
        train_x = self.features[feature_cols]
        col_indices = [self.features.columns.get_loc(c) for c in feature_cols if c in self.features]
        test_x = self.test_features.iloc[:, col_indices]

        return train_x, test_x

    def get_rfe_train_test_split(self, estimator=LogisticRegression(class_weight='auto', max_iter=900, C=5)):
        """
        Calculates the best features for the training data using RFECV and returns the training and testing datasets
        with just the features selected

        :param estimator the estimator used for the recursive feature elimination
        :return: the training and test feature values for the selected features
        """
        feature_cols = self.recursive_feature_selection(estimator=estimator)
        train_x = self.features[feature_cols]
        col_indices = [self.features.columns.get_loc(c) for c in feature_cols if c in self.features]
        test_x = self.test_features.iloc[:, col_indices]

        return train_x, test_x
