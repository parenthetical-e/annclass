""" Weight update rules. """
import numpy as np
from annclass.activation import binary, linear
from annclass.loss import delta, squared, perturb_ws, perturb_all_ws


def perceptron_update(xs, ws, y):
    """ The perceptron learning rule. """
    
    if y not in set((0,1)):
        raise ValueError("y must be either 0 or 1")
           
    h = binary(xs, ws, None)

    # ----
    # If they're not equal update and 
    # return the weights.

    # The loss function is:
    # If target is 0 when act is 1, add xs to ws
    # If target is 1 when act is 0, subtract xs from ws
    if h != y:
        if h == 0:
            ## if act and y are not the same,
            ## then is act is 0 y must be 1....
            ws_new = ws + xs
        elif h == 1:
            ## and the reverse as well.
            ws_new = ws - xs
        else:
            raise ValueError(
                    "activation (h) or target (y) values are illegal.")
            ## just in case, shouldn't get hit
    
    # If act and y were equal, ws and h returned unchanged.
    # Return the new wieghts and the new guess.    
    return ws_new, binary(xs, ws_new, None)


def linear_update(xs, ws, y, ep):
    """ Update rules for linear ANNs using the delta rule,
    i.e., delta = ep * (xs (h - y)), where h is hypothesis/activation
    (from annclass.activation.linear()).
    
    Note: in multilayer ANN an average of two good weights may be a bad 
    set of weights, i.e. they are not convex.  Unlike perceptrons, whose
    weights are proven to reach a 'good' set of weights, linear (or multilayer)
    ANNs are proven instead to have their predictions approach the target. Also 
    note that perceptrons are not proven to have their predictions approach the 
    target, yet somehow the weights are known-good.  An odd state of affairs.
    
    FYI, Linear neurons are linear filters in EE (with ep being the severity
    of filtration).  For linear cases and squared error functions there are of 
    course analytical solutions for the update rather than the 
    iterative procedure used here (e.g. OLS).  Iterative methods are used 
    so non-linear functions and non-squared error functions can be used, 
    and to better mimic the brain.
    """
    
    h = linear(xs, ws, None)
    delta_w = delta(xs, y, h, ep)  ## delta loss function
    ws_new = ws + delta_w
    
    # Return the new wieghts and the new guess.
    return ws_new, linear(xs, ws_new, None)


def logistic_update(xs, ws, y, ep):
    """ Update rules for logistic neuronal nets, with hidden layers. 
    Use backpropagation. """
    
    # Backpropgation lets you find features.
    pass

    
def perturbation_update(xs, ws, y, ep, all=True):
    """ Perturb one (if all is False) or all (if True) weights until
    performance improves. """
    
    # Pick a perturber
    if all:
        perturber = perturb_all_ws
    else:
        perturber = perturb_ws

    # Get the current guess
    h = logistic(xs, ws, None)
    
    # Init loop controllers
    cnt = 0
    stop_cnt = 5000
    
    # Loop until a random perturbation improves
    # the prediction or 5000 iterations have passed.
    # In the later case, return the intial weights.
    while(cnt < stop_cnt):
        
        # Create and test the new wieghts
        ws_new = perturber(ws, ep)
        h_new = logistic(xs, ws_new)
        if squared(y, h_new) < squared(y, h):
            return ws_new, h_new
        else:
            cnt += 1

    else:
        # The default return
        return ws, h


        
        