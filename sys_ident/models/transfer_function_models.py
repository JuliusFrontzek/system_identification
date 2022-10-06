import numpy as np
from sys_ident.models.base_model import BaseModel
from scipy.signal import lti, dlti


class BaseContinuousTFModel(BaseModel):
    """
    Base continuous transfer function model.
    The way the params list is provided defines model structure, including its order.
    The params list must be in accordance with the correct initialization described in 'scipy.signal.lti'
    """

    def lti(self, params: list) -> lti:
        return lti(*params)


class BaseDiscontinuousTFModel(BaseModel):
    """
    BaseDiscontinuous transfer function model.
    The way the params list is provided defines model structure, including its order.
    The params list must be in accordance with the correct initialization described in 'scipy.signal.dlti'
    """

    def dlti(self, params: list) -> dlti:
        return dlti(*params)


class FirstOrderContinuousTFModel(BaseContinuousTFModel):
    def lti(self, params: np.ndarray) -> lti:
        """
        Params:
            params:     1D-Numpy array of length 2. The first value represents the transfer
                        function's K value, the second one tau.
        """
        assert params.shape[0] == 2, "Params has the wrong sahpe"
        numerator = [params[0]]
        denominator = [params[1:], 1.0]
        return super().lti([numerator, denominator])


class SecondOrderContinuousTFModel(BaseContinuousTFModel):
    def lti(self, params: np.ndarray) -> lti:
        """
        Params:
            params:     1D-Numpy array of length 3. The first value represents the transfer
                        function's K value, the second one its omega value and the third one its ceta value
        """
        assert params.shape[0] == 3, "Params has the wrong sahpe"
        numerator = [params[0] * params[1] ** 2]
        denominator = [1.0, 2 * params[1] * params[2], params[1] ** 2]
        return super().lti([numerator, denominator])
