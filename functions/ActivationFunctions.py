import numpy as np


def Threshold( x , derivative=False ):

    if derivative:

        output = np.full((x.shape), 0)  # Threshold'(x)

    else :
        output = (x >= 0).astype(np.int)  # Threshold(x)

    return output


def Sigmoid( x , derivative=False ):

    if derivative:

        output = (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))  # s'(x)=s(x) * (1 - s(x))

    else:
        output = 1 / (1 + np.exp(-x))  # s(x)

    return output


def Tanh(x, derivative=False):# tanh is sometimes called hyperbolic tangent

    if derivative:

        output = 1 - np.power(np.sinh(x) / np.cosh(x), 2)  # tanh'(x) = 1- (tanh(x)^2)

    else:
        output = np.sinh(x) / np.cosh(x)  # tanh(x)

    return output


def ReLU(x, derivative=False):  #  ReLU = Rectified Linear Unit

    if derivative:

        output = (x >= 0).astype(np.int)  # ReLU'(x) = { 0  if x < 0,
                                           #             1  if x >= 0  }

    else:
        output = np.maximum(0, x)  # ReLU(x)

    return output


def leakyRELU(x, derivative=False):

    if derivative:

        output = 0.01 + ((x >= 0).astype(np.int) * 0.99)  # leaky_ReLU'(x) = {  0.01   if x < 0,
                                                          #                     1      if x >= 0  }

    else:

        output = np.maximum(0.01 * x, x)  # leaky_ReLU(x)

    return output


def Swish(x, derivative=False):

    if derivative:

        output = (1 + np.exp(-x)) + ((np.exp(-x) * x) / np.power(1 + np.exp(-x), 2))  # Swish'(x)

    else:

        output = x * 1 / (1 + np.exp(-x))  # Swish(x)

    return output


def Softmax(x, derivative=False):

    if derivative:

        # https://medium.com/@enginbozaba/softmax-fonksi%CC%87yonun-t%C3%BCrevi%CC%87-ve-matri%CC%87s-%C3%A7%C3%B6z%C3%BCm%C3%BC-b0197d1d019e

        x_full = np.full((x.size, x.size), x).T
        e_x_full = np.exp(x_full)
        np.fill_diagonal(e_x_full, 0)

        numerator = np.matrix.dot(e_x_full, np.exp(x.T))
        denominator = np.power(np.sum(np.exp(x.T)), 2)
        output = numerator / denominator


    else:

        output = np.exp(x) / np.sum(np.exp(x))  # Softmax

    return output
