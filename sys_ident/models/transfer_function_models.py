import numpy as np
from sys_ident.models.base_model import BaseModel
from scipy.signal import lti, dlti


class ContinuousTFModel(BaseModel):
    """
    Continuous transfer function model.
    The way the params list is provided defines model structure, including its order.
    The params list must be in accordance with the correct initialization described in 'scipy.signal.lti'
    """

    def lti(self, params: list) -> lti:
        return lti(*params)


class DiscontinuousTFModel(BaseModel):
    """
    Discontinuous transfer function model.
    The way the params list is provided defines model structure, including its order.
    The params list must be in accordance with the correct initialization described in 'scipy.signal.dlti'
    """

    def dlti(self, params: list) -> dlti:
        return dlti(*params)
