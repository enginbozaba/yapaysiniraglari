import numpy as np

def mean_squared_error(y_test, y_pred, derivative=False):

    if derivative:

        output = np.mean(-(y_test - y_pred), axis=0)  # Boyut : 1 x En_Son_Katmandaki_Nöron

    else:

        output = np.mean(0.5 * np.power((y_test - y_pred), 2), axis=0)  # Boyut : 1 x En_Son_Katmandaki_Nöron
        # http://www.derinogrenme.com/2018/06/28/geri-yayilim-algoritmasina-matematiksel-yaklasim/

    return output


def binary_cross_entropy(y_test, y_pred, derivative=False):

    if derivative:

        output = - 1 * ((y_test / y_pred) + (1 - y_test / 1 - y_pred))
        # https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
        # Notation : https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
        # Derivative :https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right


    else:

        output = -1 * np.mean(y_test * np.log(y_pred) + (1 - y_test) * np.log(1 - y_pred))  # -(1/n) ((y_test*log(y_pred) + (1 - y_test)*log(1 - y_pred)))

    return output

