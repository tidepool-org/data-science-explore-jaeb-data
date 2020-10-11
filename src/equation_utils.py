import math


def jaeb_basal_equation(tdd, carbs):
    """ Basal equation fitted from Jaeb data """
    return 0.6507 * tdd * math.exp(-0.001498 * carbs)


def traditional_basal_equation(tdd):
    """ Traditional basal equation """
    return 0.5 * tdd


def jaeb_isf_equation(tdd, bmi):
    """ ISF equation fitted from Jaeb data """
    return 40250 / (tdd * bmi)


def traditional_isf_equation(tdd):
    """ Traditional ISF equation """
    return 1500 / tdd


def jaeb_icr_equation(tdd, carbs):
    """ ICR equation fitted from Jaeb data """
    return (1.31 * carbs + 136.3) / tdd


def traditional_icr_equation(tdd):
    """ Traditional ICR equation """
    return 500 / tdd
