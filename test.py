""" A top-level testing script. """
import numpy as np
from annclass.misc import setup_bias
from annclass.update import perceptron_update, linear_update
from annclass.activation import linear


def print_perceptron_update(xs, ws, bias, y):
    """ Prints initial and updated ws. """
    
    # Convert xs ans ws to the proper form
    # for use in perceptron
    xs, ws = setup_bias(xs, ws, bias)
    ws_new, h = perceptron_update(xs, ws, y)

    print("ws_intial: {0}\nws_new: {1}".format(ws, ws_new))


def print_linear_update(xs, ws, bias, y, ep):
    """ Prints initial and updated ws, and h. """
    
    # Convert xs ans ws to the proper form
    # for use in perceptron
    xs = np.array(xs)
    ws = np.array(ws) 
        # Not using setup_bias so do this by hand.
    
    ws_new, h = linear_update(xs, ws, y, ep)
    h_new = linear(xs, ws_new, None)
    
    print("ws_intial: {0}\nws_new: {1}".format(ws, ws_new))
    print("h_intial: {0}\nh_new: {1}".format(h,h_new))
    

if __name__ == "__main__":
    
    # ----
    # Lecture 2.1 quiz:
    print("----\nLecture 2.1 quiz:\n----")
    xs = [0.5, -0.5]
    ws = [2, -1]
    bias = 0.5
    y = 0

    print_perceptron_update(xs, ws, bias, y)
    print("ws_new should be [-0.5, 1.5, -0.5]")
        ## correct quiz answer of 
        ##   ws_new is [1.5, -0.5] with a bias of -0.5
        ## suggests perceptron code is working
        ## correctly ....so far anyway
    # ----

    # ----
    # Lecture 3.1 quiz:
    print("----\nLecture 3.1 quiz:\n----")
    xs = [2, 5, 3]
    ws = [50, 50, 50]
    bias = None
    y = 850
    ep = 1/35.

    print_linear_update(xs, ws, bias, y, ep)
    print("ws_new should be [70, 100, 80]")
    print("h_new should be 880")
        ## correct quiz answer of 
        ##   ws_new is [1.5, -0.5] with a bias of -0.5
        ## suggests perceptron code is working
        ## correctly ....so far anyway
    # ----