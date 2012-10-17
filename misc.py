""" Misc useful functions for annclass. """
import numpy as np


def setup_bias(xs, ws, bias):
    """ Convert xs and ws to arrays, then add bias information to xs and
    ws thus standardizing the form and simplifying the calculation of the 
    activation functions. """
    
    xs = np.array(xs)
    ws = np.array(ws)
        ## Explicit typing just in case
        
    xs = np.append(1, xs)
    ws = np.append(bias, ws)

    return xs, ws