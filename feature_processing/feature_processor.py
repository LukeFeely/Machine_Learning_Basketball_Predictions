import sklearn.preprocessing as pp

'''
The feature_processor abstracts the different scaling functions we tested. We set out default values here so they
do not need to be repeatedly set when using them.
'''


def min_max_scale_features(features):
    return pp.MinMaxScaler().fit_transform(features)


def standard_scale_features(features):
    return pp.StandardScaler().fit_transform(features)


def max_abs_scale_features(features):
    return pp.MaxAbsScaler().fit_transform(features)


def robust_scale_features(features):
    return pp.RobustScaler(quantile_range=(25, 75)).fit_transform(features)


def power_transform_scale_features(features):
    return pp.PowerTransformer().fit_transform(features)


def quantile_scale_features(features):
    return pp.QuantileTransformer(output_distribution='normal', n_quantiles=len(features)).fit_transform(features)


def quantile_2_scale_features(features):
    return pp.QuantileTransformer(n_quantiles=len(features)).fit_transform(features)


def normalise_scale_features(features):
    return pp.Normalizer().fit_transform(features)
