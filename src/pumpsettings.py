import numpy as np


class PumpSettings:
    def __init__(self, basal_equation, isf_equation, icr_equation):
        """
        basal_equation: function that can be used to compute basal setting
        isf_equation: function that can be used to compute ISF setting
        icr_equation: function that can be used to compute ICR setting
        """
        self.basal_equation = basal_equation
        self.isf_equation = isf_equation
        self.icr_equation = icr_equation
