from sklearn.utils.estimator_checks import check_estimator

from timefiller import ExtremeLearningMachine


def test_ExtremeLearningMachine():
    check_estimator(ExtremeLearningMachine())
    check_estimator(ExtremeLearningMachine(ratio_features_projection=1.5))
    check_estimator(ExtremeLearningMachine(ratio_features_projection=2))
    check_estimator(ExtremeLearningMachine(n_features_projection=10))
