""" Weight update rules. """
import numpy as np
from annclass.activation import binary, linear


def perceptron_update(xs, ws, y):
    """ The proven good perceptron learning rule. """
    
    if y not in set((0,1)):
        raise ValueError("y must be either 0 or 1")
    
    xs = np.array(xs)
    ws = np.array(ws)
        ## Explicit typing just in case
       
    h = binary(xs, ws, None)

    # ----
    # If they're not equal update and 
    # return the weights.

    # The update rule:
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
        
    return ws_new, h
        ## If act and y were equal, 
        ## ws gets returned unchanged.


def linear_update(xs, ws, y, ep):
    """ Update rules for linear ANNs, use the delta rule to update the weights,
    i.e. ep * xs (h - y), where h is the linear activation 
    (from annclass.activation.linear).  
    
    Note: in multilayer ANN an average of two good weights may be a bad 
    set of weights, i.e. they are not convex.  Unlike perceptrons, whose
    weights are proven to reach a 'good' set of weights, linear (or multilayer)
    ANNs are proven instead to have their predictions approach the target. Also 
    note that perceptrons are not proven to have their predictions approach the 
    target, yet some how the weights are known-good.  Odd.
    
    FYI, Linear neurons are linear filters in EE (with ep being the severity
    of filtration).  For linear cases and squared error functions there are of 
    course analytical solutions for the update rather than the 
    iterative procedure used here (e.g. OLS).  Iterative methods are used 
    so non-linear functions and non-squared error functions can be used, 
    and to better mimic the brain.
    """
    
    h = linear(xs, ws, None)
    print(xs*(h-y))
    delta_w = float(ep) * (xs * (y - h))
    print(delta_w)
    ws_new = ws + delta_w
    
    return ws_new, h

