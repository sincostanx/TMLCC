import numpy as np
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import mean_absolute_error

class LMAE(Metric):
    def __init__(self):
        self._name = "LMAE"
        self._maximize = False

    def __call__(self, y_true, y_score):
        mae = mean_absolute_error(y_true, y_score)
        return np.log(mae)