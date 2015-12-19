import math
import numpy as np

@np.vectorize
def g(x):
    'Activation function'
    return 1/(1+math.exp(-x))
    
@np.vectorize
def dg(g):
    'calculates dg/dx by given g(x)'
    return g*(1-g)

class Network:
    'Primitive activation linear neural network'
    
    def __init__(self, layer_sizes):
        'init network specified by give sizes and initialize it by random wights'
        self.layers=[]
        self.activations = {}
        # init by random values
        for i in range(1, len(layer_sizes)):
            random_array = np.random.rand(layer_sizes[i], layer_sizes[i-1] + 1)
            self.layers.append(np.asmatrix(random_array))
            
    def result(self):
        'calculated result of forward propogation'
        return self.activations[len(self.layers)]

    def forward(self, x):
        'calculate the result of forward propogation'
        self.activations[0] = np.matrix.transpose(x)
        for l in range(0, len(self.layers)):
            inp = np.vstack([np.asmatrix([1]), self.activations[l]])
            z = self.layers[l] * inp
            self.activations[l+1] = g(z)
        return self.result()
        
    def backward(self, y):
        'make a correction of weights based on expectations for last result'
        
        @np.vectorize
        def distance(y, a):
            'Distance between elements'
            return ((y - a) ** 2)/2
        
        y = np.matrix.transpose(y)
        E = distance(y, self.result());
        Etotal = sum(E)
        print(Etotal)
        
        # Backward pass for output layer
        
        

net = Network([5, 10, 2])

print(net.forward(np.asmatrix([1,2,3,4,5])))

net.backward(np.asmatrix([0,1]))
