""" A base/template class for running ANNs. """


class ANN(self):
    def __init__(self, ann_constructor, update_function):
        self.updater = update_function
        
        # Construct (i.e. create and initialize) the ANN.
        ann_constructor()
        # 
        
        
    def train(self):
        pass