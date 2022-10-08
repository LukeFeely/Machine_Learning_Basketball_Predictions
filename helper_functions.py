from enum import Enum
from typing import Union, Type

import numpy
import numpy as np
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class HyperParam(Enum):
    """
    Contains the possible hyper-parameters for our choice of models
    """
    C = 1
    POWER = 2
    GAMMA = 3
    K = 4


def cross_validate(model_type: Union[Type[LogisticRegression], Type[KNeighborsClassifier], Type[SVC]],
                   hyper_param: HyperParam, hyper_param_values_list, x_training_data, y_training_data, k_fold_splits=10,
                   penalty_type="l2", max_iter=None, weights=None) -> None:
    """

    :param model_type: the model for which we are cross-validating the hyper-param
    :param hyper_param: instance of HyperParam that we are cross-validating
    :param hyper_param_values_list: possible values for hyper-param
    :param x_training_data: the input training data
    :param y_training_data: the output labels for the training data
    :param k_fold_splits: number of k-folds to use (default to 10)
    :param penalty_type: penalty type for logistic (default l2)
    :param max_iter: set max iterations (optional)
    :param weights: set weights for SVC (optional)
    :return: None
    """

    print("cross validating....")
    x_training_data = np.array(x_training_data)
    y_training_data = np.array(y_training_data)
    log_loss_list = []  # using log loss for classification problem
    standard_dev = []  # standard deviation of log loss

    for param in hyper_param_values_list:
        print(f"validating for {hyper_param} {param}")

        model_params = {}
        if hyper_param == HyperParam.C:  # populate model_params with correct parameters for model and hyper-param
            model_params['C'] = param
        if max_iter is not None:
            model_params['max_iter'] = max_iter
        if weights is not None:
            model_params["weights"] = weights
        if model_type == LogisticRegression:
            model_params['penalty'] = penalty_type
        elif model_type == SVC and hyper_param == HyperParam.GAMMA:
            model_params['gamma'] = param
        elif model_type == KNeighborsClassifier:
            model_params["n_neighbors"] = param

        model = model_type(**model_params)  # create the specified model
        pipeline = make_pipeline(StandardScaler(), model)  # add normalisation to pipeline

        if hyper_param == HyperParam.POWER:  # create polynomial features if specified
            x_training_data = PolynomialFeatures(param).fit_transform(x_training_data)

        temp = []
        kf = KFold(n_splits=k_fold_splits)  # perform k-fold cross validation
        for train, test in kf.split(x_training_data):
            pipeline.fit(x_training_data[train], y_training_data[train].ravel())  # fit model for current split
            y_pred = pipeline.predict(x_training_data[test])  # get predictions
            temp.append(log_loss(y_training_data[test], y_pred))  # calculate log loss
        log_loss_list.append(numpy.array(temp).mean())  # add log loss mean to final list
        standard_dev.append(numpy.array(temp).std())  # add log loss standard deviation to list

    # plot using matplotlib and have error bars as standard_dev
    pyplot.errorbar(hyper_param_values_list, log_loss_list, yerr=standard_dev, label="Standard Deviation")
    pyplot.title(f"Prediction error vs {hyper_param} - {model_type.__name__}")
    pyplot.xlabel(f"{hyper_param} Value")
    pyplot.ylabel("Log loss")
    pyplot.legend()
    pyplot.show()
