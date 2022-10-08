import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import \
    accuracy_score, plot_confusion_matrix, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from helper_functions import cross_validate, HyperParam
from CSVGenerator import CSVGenerator
from feature_processing.feature_selector import FeatureSelector


def parse_input(user_input):
    """
    returns if the user has given y or not when prompted
    :param user_input:
    :return: true if input is y else false
    """
    user_input = user_input.lower()[0]
    if user_input == "y":
        return True
    return False


years_for_testing = [2015, 2016, 2017, 2018, 2019]  # list of seasons to test the models
should_run_cross_validation = False
should_scrape_data = parse_input(input("Do you want to scrape the data? (y/n)\n> "))
should_gen_csv = parse_input(input("Do you want to generate CSV? (y/n)\n> "))
model_accuracies = {"LOGISTIC": [], "SVC": [], "KNN": [], "LOGISTIC_RFECV": [], "LOGISTIC_KBEST": []}
model_precision = {"LOGISTIC": [], "SVC": [], "KNN": [], "LOGISTIC_RFECV": [], "LOGISTIC_KBEST": []}
model_recall = {"LOGISTIC": [], "SVC": [], "KNN": [], "LOGISTIC_RFECV": [], "LOGISTIC_KBEST": []}
model_f1 = {"LOGISTIC": [], "SVC": [], "KNN": [], "LOGISTIC_RFECV": [], "LOGISTIC_KBEST": []}

for index, year_for_testing in enumerate(years_for_testing):  # iterate over years to test
    print(f"\n-------  {index + 1}/{len(years_for_testing)}: testing on {year_for_testing}  --------\n")

    training_years = [2015, 2016, 2017, 2018, 2019]
    training_years.remove(year_for_testing)  # remove testing year

    first_iteration_of_k_fold = index == 0
    if should_scrape_data and first_iteration_of_k_fold:
        CSVGenerator(0).scrape_all_training_data(training_years)  # scrape the necessary data to files
        CSVGenerator(year_for_testing).generate_game_stats()  # generate stats for testing year

    if should_gen_csv:
        # generate the csv containing info for all training seasons
        CSVGenerator(0).generate_multiple_years(training_years)
        # make csv for testing year
        CSVGenerator(year_for_testing).generate()

    FILE_PATH_TEST = f"data/{year_for_testing}_games.csv"
    FILE_PATH_TRAIN = f"data/training_features/training_features_{str(training_years)[1:-1]}.csv"

    training_csv_dataframe = pd.read_csv(FILE_PATH_TRAIN)  # read csv files into dataframes
    testing_csv_dataframe = pd.read_csv(FILE_PATH_TEST)
    num_columns = len(training_csv_dataframe.columns)

    x_input_features = training_csv_dataframe.iloc[:, range(0, num_columns - 1)]  # get the input data for training
    y_output_data = training_csv_dataframe.iloc[:, [num_columns - 1]]  # get output/label data for training

    test_x_input_features = testing_csv_dataframe.iloc[:, range(0, num_columns - 1)]  # get the input data for testing
    test_y_output_data = testing_csv_dataframe.iloc[:, [num_columns - 1]]  # get output/label data for testing

    if year_for_testing == 2019 and should_run_cross_validation:  # only generate cross validation plots for 2018,
        # just to make things easy to see

        cross_validate(LogisticRegression, HyperParam.C, [0.001, 0.01, 0.1, 1, 5, 10, 15, 20], x_input_features,
                       y_output_data)
        cross_validate(SVC, HyperParam.GAMMA, [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.007], x_input_features,
                       y_output_data)
        cross_validate(SVC, HyperParam.C, [0.001, 0.01, 0.1, 1, 5, 10, 15, 20], x_input_features, y_output_data)
        cross_validate(KNeighborsClassifier, HyperParam.K, [1, 5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200],
                       x_input_features, y_output_data)
        cross_validate(KNeighborsClassifier, HyperParam.K, [1, 5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200],
                       x_input_features, y_output_data, weights="distance")
        cross_validate(LogisticRegression, HyperParam.POWER, [1, 2], x_input_features, y_output_data, max_iter=1500)

    logistic_model = LogisticRegression(class_weight='auto', max_iter=900, C=5)  # best LR from above plots
    svc_model = SVC(gamma=0.001, C=1)  # best SVC from above plots
    knn_model = KNeighborsClassifier(n_neighbors=125)  # best kNN from above plots

    logistic_pipeline = make_pipeline(StandardScaler(), logistic_model)  # add normalisation to pipeline
    logistic_pipeline.fit(x_input_features, np.array(y_output_data).ravel())  # fit model
    y_pred = logistic_pipeline.predict(test_x_input_features)  # get predictions
    logistic_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)  # find and update accuracies
    model_accuracies["LOGISTIC"].append(logistic_accuracy)
    print(f'Logistic Coefs : {logistic_pipeline.named_steps["logisticregression"].coef_}')
    print(f'Logistic Accuracy : {logistic_accuracy}')

    logistic_precision = precision_score(y_true=test_y_output_data, y_pred=y_pred)
    model_precision["LOGISTIC"].append(logistic_precision)
    print(f'Logistic Precision : {logistic_precision}')

    logistic_recall = recall_score(y_true=test_y_output_data, y_pred=y_pred)
    model_recall["LOGISTIC"].append(logistic_recall)
    print(f'Logistic Recall : {logistic_recall}')

    logistic_f1 = f1_score(y_true=test_y_output_data, y_pred=y_pred)
    model_f1["LOGISTIC"].append(logistic_f1)
    print(f'Logistic f1 : {logistic_f1}')

    if year_for_testing == 2019:
        coefficients_dataframe = pd.DataFrame({'feature': x_input_features.columns,
                                               'coef': logistic_pipeline.named_steps["logisticregression"].coef_[0]})
        coefficients_dataframe['coef'] = coefficients_dataframe['coef'].apply(abs)
        coefficients_dataframe = coefficients_dataframe.sort_values(by='coef', ascending=False)
        coefficients_dataframe_copy = coefficients_dataframe.copy()

        accuracies = []
        number_of_features = []
        for i in range(len(coefficients_dataframe_copy.index)):
            train_x = x_input_features[coefficients_dataframe_copy['feature']]
            col_indices = [x_input_features.columns.get_loc(c) for c in train_x.columns if c in x_input_features]
            test_x = test_x_input_features.iloc[:, col_indices]

            logistic_pipeline.fit(train_x, np.array(y_output_data).ravel())
            y_pred = logistic_pipeline.predict(test_x)
            logistic_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)

            accuracies.append(logistic_accuracy)
            number_of_features.append(len(coefficients_dataframe_copy.index))

            coefficients_dataframe_copy = coefficients_dataframe_copy[:-1]

        pyplot.rcParams["figure.figsize"] = (10, 8)
        pyplot.rcParams["font.size"] = 12

        fig, ax1 = pyplot.subplots()
        ax1.bar(coefficients_dataframe['feature'], coefficients_dataframe['coef'], color='c', label='Coefficients')
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Coefficient Magnitude')
        pyplot.xticks(rotation=60, ha='right')

        x_values = list(map(lambda x: x - 1, number_of_features))
        ax2 = ax1.twinx()
        ax2.plot(x_values, accuracies, color='m', label='Accuracy')
        ax2.set_ylabel('Accuracy')

        ax1.legend(loc=1)
        ax2.legend()
        pyplot.tight_layout()
        pyplot.show()

    svc_pipeline = make_pipeline(StandardScaler(), svc_model)  # add normalisation to pipeline
    svc_pipeline.fit(x_input_features, np.array(y_output_data).ravel())  # fit model
    y_pred = svc_pipeline.predict(test_x_input_features)  # get predictions
    svc_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)  # find and update accuracies
    model_accuracies["SVC"].append(svc_accuracy)
    print(f'SVC Accuracy : {svc_accuracy}')

    svc_precision = precision_score(y_true=test_y_output_data, y_pred=y_pred)
    model_precision["SVC"].append(svc_precision)
    print(f'SVC Precision : {svc_precision}')

    svc_recall = recall_score(y_true=test_y_output_data, y_pred=y_pred)
    model_recall["SVC"].append(svc_recall)
    print(f'SVC Recall : {svc_recall}')

    svc_f1 = f1_score(y_true=test_y_output_data, y_pred=y_pred)
    model_f1["SVC"].append(svc_f1)
    print(f'SVC f1 : {svc_f1}')

    knn_pipeline = make_pipeline(StandardScaler(), knn_model)  # add normalisation to pipeline
    knn_pipeline.fit(x_input_features, np.array(y_output_data).ravel())  # fit model
    y_pred = knn_pipeline.predict(test_x_input_features)  # get predictions
    knn_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)  # find and update accuracies
    model_accuracies["KNN"].append(knn_accuracy)
    print(f'KNN Accuracy : {knn_accuracy}')

    knn_precision = precision_score(y_true=test_y_output_data, y_pred=y_pred)
    model_precision["KNN"].append(knn_precision)
    print(f'KNN Precision : {knn_precision}')

    knn_recall = recall_score(y_true=test_y_output_data, y_pred=y_pred)
    model_recall["KNN"].append(knn_recall)
    print(f'KNN Recall : {knn_recall}')

    knn_f1 = f1_score(y_true=test_y_output_data, y_pred=y_pred)
    model_f1["KNN"].append(knn_f1)
    print(f'KNN f1 : {knn_f1}')

    feature_selector = FeatureSelector(training_csv_dataframe, testing_csv_dataframe)
    rfe_input_features, rfe_test_x_input_features = feature_selector.get_rfe_train_test_split()
    logistic_pipeline.fit(rfe_input_features, np.array(y_output_data).ravel())
    y_pred = logistic_pipeline.predict(rfe_test_x_input_features)
    rfe_logistic_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)
    model_accuracies["LOGISTIC_RFECV"].append(rfe_logistic_accuracy)
    print(f'Logistic Accuracy with RFE selected features: {rfe_logistic_accuracy}')

    k_best_input_features, k_best_test_x_input_features = feature_selector.get_k_best_train_test_split()
    logistic_pipeline.fit(k_best_input_features, np.array(y_output_data).ravel())
    y_pred = logistic_pipeline.predict(k_best_test_x_input_features)
    kbest_logistic_accuracy = accuracy_score(y_true=test_y_output_data, y_pred=y_pred)
    model_accuracies["LOGISTIC_KBEST"].append(kbest_logistic_accuracy)
    print(
        f'Logistic Accuracy with Kbest selected features: {kbest_logistic_accuracy}')

    pyplot.title('ROC Curves')
    pyplot.ylabel('True Positive Rate')
    pyplot.xlabel('False Positive Rate')

    if year_for_testing == 2019 and should_run_cross_validation:  # plot roc curves for 2018
        # split into x and y testing & training data
        x_train, x_test, y_train, y_test = train_test_split(x_input_features, y_output_data, test_size=0.2)
        knn_pipeline.fit(x_train, np.array(y_train).ravel())
        fpr, tpr, _ = roc_curve(y_test, knn_pipeline.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)  # get the area under the curve
        pyplot.plot(fpr, tpr, color="blue", label='KNN AUC = %0.8f' % roc_auc)  # plot the curve

        svc_pipeline.fit(x_train, np.array(y_train).ravel())
        fpr, tpr, _ = roc_curve(y_test, svc_pipeline.decision_function(x_test))
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, color="green", label='SVC AUC = %0.8f' % roc_auc)

        logistic_pipeline.fit(x_train, np.array(y_train).ravel())
        fpr, tpr, _ = roc_curve(y_test, logistic_pipeline.decision_function(x_test))
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, color="orange", label='Logistic Regression AUC = %0.8f' % roc_auc)

        baseline_pipeline = make_pipeline(StandardScaler(), DummyClassifier(strategy="most_frequent"))
        baseline_pipeline.fit(x_train, np.array(y_train).ravel())
        fpr, tpr, _ = roc_curve(y_test, baseline_pipeline.predict_proba(x_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        pyplot.plot(fpr, tpr, color="red", label='Baseline AUC = %0.8f' % roc_auc)

        pyplot.legend(loc='lower right')
        pyplot.show()

        best_pipeline = logistic_pipeline  # make confusion matrix for logistic regression model
        plot_confusion_matrix(best_pipeline, test_x_input_features, test_y_output_data)
        pyplot.title("Logistic Regression")
        pyplot.show()

        baseline_pipeline.fit(x_input_features, y_output_data)  # make confusion matrix for most_frequent model
        plot_confusion_matrix(baseline_pipeline, test_x_input_features, test_y_output_data)
        pyplot.title("Most Frequent Baseline")
        pyplot.show()

        baseline_accuracy = accuracy_score(y_pred=baseline_pipeline.predict(test_x_input_features),
                                           y_true=test_y_output_data)
        print(f"Baseline Accuracy: {baseline_accuracy}")
        print(f"\n-------  end of testing on {year_for_testing}  --------\n")

print("------ AVERAGE ACCURACY -------")
print("\nLogistic Regression:")
print(f"K-Fold Results : {np.array(model_accuracies['LOGISTIC'])}")
print(f"Mean Accuracy : {np.mean(model_accuracies['LOGISTIC'])}")
print(f"Variance:{np.var(model_accuracies['LOGISTIC'])}")
print(f"Mean Precision: {np.mean(model_precision['LOGISTIC'])}")
print(f"Mean Recall: {np.mean(model_recall['LOGISTIC'])}")
print(f"Mean f1: {np.mean(model_f1['LOGISTIC'])}")

print("\nSVC:")
print(f"K-Fold Results : {np.array(model_accuracies['SVC'])}")
print(f"Mean Accuracy : {np.mean(model_accuracies['SVC'])}")
print(f"Variance:{np.var(model_accuracies['SVC'])}")
print(f"Mean Precision: {np.mean(model_precision['SVC'])}")
print(f"Mean Recall: {np.mean(model_recall['SVC'])}")
print(f"Mean f1: {np.mean(model_f1['SVC'])}")

print("\nKNN:")
print(f"K-Fold Results : {np.array(model_accuracies['KNN'])}")
print(f"Mean Accuracy : {np.mean(model_accuracies['KNN'])}")
print(f"Variance:{np.var(model_accuracies['KNN'])}")
print(f"Mean Precision: {np.mean(model_precision['KNN'])}")
print(f"Mean Recall: {np.mean(model_recall['KNN'])}")
print(f"Mean f1: {np.mean(model_f1['KNN'])}")

print("\nLogistic Regression with RFECV:")
print(f"K-Fold Results : {np.array(model_accuracies['LOGISTIC_RFECV'])}")
print(f"Mean Accuracy : {np.mean(model_accuracies['LOGISTIC_RFECV'])}")
print(f"Variance:{np.var(model_accuracies['LOGISTIC_RFECV'])}")

print("\nLogistic Regression with KBest:")
print(f"K-Fold Results : {np.array(model_accuracies['LOGISTIC_KBEST'])}")
print(f"Mean Accuracy : {np.mean(model_accuracies['LOGISTIC_KBEST'])}")
print(f"Variance:{np.var(model_accuracies['LOGISTIC_KBEST'])}")
