""" A top-level testing script. """
import numpy as np
from annclass.misc import setup_bias
from annclass.update import perceptron


def print_perceptron_update(xs, ws, bias, y):
    """ Prints initial and updated ws. """
    
    # Convert xs ans ws to the proper form
    # for use in perceptron
    xs, ws = setup_bias(xs, ws, bias)
    ws_new = perceptron(xs, ws, y)

    print("ws_intial: {0}\nws_new: {1}".format(ws, ws_new))


if __name__ == "__main__":
    
    # ----
    # Lecture 2.1 quiz:
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
