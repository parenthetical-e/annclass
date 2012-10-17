""" Some basic ANN activation rules, from Lectures 1 and 2. 

Notes: 
    * As seems to be the general custom bias was incorporated into xs an ws; 
    it is prepended to ws and a 1 is prepended to xs,
    see annclass.misc.setup_bias() for complete details. 
    * However if bias is None, it assumed that that xs and ws already include
    a bias term, as is the case when these functions are called from, for 
    example, the annclass.update module. 
"""
import numpy as np


def linear(xs, ws, bias):
    """ returns: sum(xs * ws) + b """
    
    if bias != None:
        xs, ws = setup_bias(xs, ws, bias)
  
    return np.sum(xs * ws)
        ## np.dot(np.transpose(xs), ws) would probably be
        ## more effcient (via LAPACK, or whatever) but the 
        ## above is more readable, to me.  Also, it is vectorized
        ## like this so should be reasonably fast as is.
        ## If I ever use this code for anything real, test the
        ## two implmentations.


def binary(xs, ws, bias):
    """ z = sum(xs * ws) - bias; if z > 0, return 1, else return 0. 
    Note: this is the activation function used in perceptrons. """
    
    if bias != None:
        xs, ws = setup_bias(xs, ws, -bias)
    
    z = np.sum(xs * ws)
    
    if z > 0:
        return 1
    else:
        return 0
 
   
def rectified(xs, ws, bias):
    """ z = sum(xs * ws) - bias; if z > 0, return z, else return 0. """

    if bias != None:
        xs, ws = setup_bias(xs, ws, -bias)
        
    z = np.sum(xs * ws)

    if z > 0:
        return z
            ## Making this non-linear, 
            ## i.e. it looks like ____/ 
            ## where the deflection starts at z
            ## and the underscores rest on 0
    else:
        return 0


def sigmoid(xs, ws, bias):
    """ z = sum(xs * ws) - bias; return the probability of an activation, 
    i.e.1.0 / (1.0 + np.exp(-z)). """

    if bias != None:
        xs, ws = setup_bias(xs, ws, bias)
        
    z = np.sum(xs * ws)

    return 1.0 / (1.0 + np.exp(-z))
        ## "The smooth dirivatives of the sigmoidal 
        ## makes learning (by grad des) easier."


def stocastic_binary(xs, ws, bias):
    """ Use the probabilistic value (p) from sigmoid() to sample a binomial 
    dist, i.e. return {0,1} contingent on p. """
    
    p = sigmoid(xs, ws, bias)
    
    return np.random.binomial(1, p)
    

def stocastic_rectified(xs, ws, bias):
    """ Use the real-valued output of rectified() as a rate parameter 
    to stochastically generate spikes counts from a poisson process. """
    
    rate = rectified(xs, ws, bias)
    
    return np.random.poisson(rate)
    
    