""" Weight update rules. """
import numpy as np
from annclass.activation import binary


def perceptron(xs, ws, y):
    """ The proven good perceptron learning rule. """
    
    if y not in set((0,1)):
        raise ValueError("y must be either 0 or 1")
    
    xs = np.array(xs)
    ws = np.array(ws)
        ## Explicit typing just in case
       
    act = binary(xs, ws, None)

    # ----
    # If they're not equal update and 
    # return the weights.

    # The update rule:
    # If target is 0 when act is 1, add xs to ws
    # If target is 1 when act is 0, subtract xs from ws
    if act != y:
        if act == 0:
            ## if act and y are not the same,
            ## then is act is 0 y must be 1....
            ws = ws + xs
        elif act == 1:
            ## and the reverse as well.
            ws = ws - xs
        else:
            raise ValueError(
                    "activation (binary()) or target (y) values are illegal.")
            ## just in case, shouldn't get hit
        
    return ws
        ## If act and y were equal, 
        ## ws gets returned unchanged.
