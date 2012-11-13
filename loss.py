""" A menagerie of loss functions.  All functions return weight changes,
i.e. delta_ws. """
import numpy as np


def squared(y, h):
    """ The SSE loss function. """
    
    return 0.5 * sum((y - h)**2)
    
    
def delta(xs, y, h, ep):
    """ The delta rule. """
    
    return float(ep) * (xs * (y - h))
        ## From lecture 3.1:
        ## The delta rule follows from the patial dirivative of
        ## the squared residuals with each wieght (w_i).
        ## I.e., 
        ## if SSE = 0.5 * sum((y - h)**2)
        ## 
        ## Then the SSE changes with infentesimal weight changes as....
        ## 
        ## dSSE / dw_i = 0.5 sum(dy/dw_i * dE/dy) (by chain rule)
        ## which becomes following calculus-ification:
        ##
        ## float(ep) * (xs * (y - h)), our delta rule.
        ## 
        ## ...Which is also the OLS update.
        ## For small ep, the delta rule with minimize the SSE
        ## however correlated data (xs) slows learning dramatically.
        ## 
        ## Too large an ep leads to unstablity, 
        ## too small and it takes forever.  
        ## ....Parameter selection is hard


def logistic(xs, y, h):
    """ The loss for logistic neurons. """
    
    return (y * (y - 1)) * (xs * (y - h))
        ## Define z:
        ## z = bias + sum_i(xs * xs); read sum_i as sum over i
        ## so
        ## dz/dw_i = x_i; dz/dx_i = w_i
        ## which is directly from their 0 zero order linear relation
        ##     
        ## Less obvious is the fact that 
        ## dy/dz = y * (1 - y), 
        ## see the 2:22 mark in lecture 3.3
        ## for the derivation
        ## 
        ## But what we want is how changes with the wieghts, i.e.
        ## 
        ## dy/dw_i but ia the chain rule we can just
        ## 
        ## So we combine the above to find it.
        ## State the problem as a chain:
        ## dy/dw_i = dz/dw_i * dy/dz = x_i * y * (1 - y)
        ## 
        ## Which following differentiation (again assuming a SSE as in
        ## the delta case above) becomes:
        ## sum_i( xs * (y - h) * y * (y - 1))
        ##
        ## Which is the delta rule (xs * (y - h))
        ## multiplied by the the slope of the logistic funcion
        ## (y * (y - 1)).
    

# TODO test all perturbation functions
def perturb_ws(ws, ep, seedval=42):
    """ Uniform random perturbation of a randomly selected weight,
    scaled by ep. """
    
    # Where to perturb:
    location = np.random.random_integers(0, ws.size - 1)
    
    # Init delta_ws, then perturb an entry
    delta_ws = np.zeros_like(ws)
    
    # Flatten, insert perturbation (ep * rand), and rehape.
    delta_ws_tmp = delta_ws.flatten()
    delta_ws_tmp[location] = float(ep) * np.random.rand()
    delta_ws = delta_ws_tmp.reshape(delta_ws.shape)

    return delta_ws


def perturb_all_ws(ws, ep, seedval=42):
    """ Uniform random perturbation of all ws, scaled by ep - 
    i.e., do a random search for good weights. """

    # Seed setup for consistent generation
    np.random.seed(seedval)
    
    return float(ep) * np.random.random_sample(ws.shape)
        ## Generate random wieght changes, 
        ## rescale them by ep


def perturb_al_ws_neighborhood(ws, ep, seedval=42):
    """ Return two sets of random perturbations. One positive
    and one one negative allowing for exploration of ep sized neighborhoods
    around ws. """    
    
    delta_w = perturb_all_ws(ws, ep, seedval)
    return delta_w, -1 *changes delta_w


