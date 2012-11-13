""" A submodule of classes for modifying loss function behavior. 

They're loss function agonistic, though it is assumed lossfunction returns
a real value representing the change in weight for the current example (i.e. delta_wi in the syntactical convention of loss.py).

NOTE: Using classes as these approaches need to maintain a state and classes 
are an easy if expensive way to do that. """
import numpy as np
from copy import deepcopy


# NOTE: It might be nice if step() had a consistent signature...
class Momentum():
    """ Add momentum to weight changes. """
    
    def __init__(self, alpha, lossfunction, *args):
        self.lossfunction = lossfunction
        self.alpha = alpha
        self.delta_tminus = self.alpha * self.lossfunction(*args)
    
    
    def step(self, *args):
        delta_wi = (self.alpha * self.delta_tminus) - self.lossfunction(*args)
        self.delta_tminus = deepcopy(delta_wi)

        return delta_wi


class Nesterov():
    """ Use the Nesterov (1983) momentum method -- step than correct,
    rather then step with a past, as in momentum (which is sort of like a 
    pre-correction). """
    
    def __init__(self, lossfunction, *args):
        self.lossfunction = lossfunction
        
        pass
        # TODO the lecture did not give implementation details..
        # look these up.


    def step(self, *args):
        pass
        

class Adaptive():
    """ Implements adaptive learnings rates as described in lecture 6d."""
    
    def __init__(self, gain, lossfunction, *args):
        self.lossfunction = lossfunction
        self.gain = gain
        pass
 
    
    def step(self, *args):
        pass

