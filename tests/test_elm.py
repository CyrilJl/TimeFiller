from sklearn.utils.estimator_checks import check_estimator

from timefiller import ExtremeLearningMachineRegressor


def test_ExtremeLearningMachine():
    check_estimator(ExtremeLearningMachineRegressor(ratio_features_projection=1.5))
    check_estimator(ExtremeLearningMachineRegressor(ratio_features_projection=2))
    check_estimator(ExtremeLearningMachineRegressor(n_features_projection=10))
