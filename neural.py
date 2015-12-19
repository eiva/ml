import math
import numpy as np

@np.vectorize
def g(x):
    'Activation function'
    return 1/(1+math.exp(-x))
    
def dg(g):
    'calculates dg/dx by given g(x)'
    return g*(1-g)

class Network:
    'Primitive activation linear neural network'
    
    def __init__(self, layer_sizes : list):
        'init network specified by give sizes and initialize it by random wights'
        self.fi=[] # weights of network
        self.a = {} # output activation from neurons
        self.delta = {} # delta function
        self.df = {} # corection for weights

        # init by random values
        for i in range(1, len(layer_sizes)):
            random_array = np.random.rand(layer_sizes[i], layer_sizes[i-1] + 1)
            self.fi.append(np.asmatrix(random_array))
            
    def result(self):
        'calculated result of forward propogation'
        return self.a[len(self.fi)]

    def forward(self, x):
        'calculate the result of forward propogation'
        self.a[0] = np.matrix.transpose(x)
        for l in range(0, len(self.fi)):
            inp = np.vstack([np.asmatrix([1]), self.a[l]])
            z = self.fi[l] * inp
            self.a[l+1] = g(z)
        return self.result()
        
    def backward(self, y):
        'make a correction of weights based on expectations for last result'
        y = np.matrix.transpose(y)

        @np.vectorize
        def distance(y, a):
            'Distance between elements'
            return ((y - a) ** 2)/2

        # Calculate backpropogation step for output layer

        @np.vectorize
        def calcdeltal(y, a):
            'Calculate delta on layer L (output layer)'
            return - (y - a) * dg(a)

        L = len(self.fi)
        delta = calcdeltal(y, self.a[L])
        
        self.delta[L] = delta;

        a = np.vstack([[1], self.a[L-1]])
        df = delta * np.matrix.transpose(a)
        self.df[L] = df
        f = self.fi[L-1]

        # Calculate backprop for all hidden layer

        def calcsum(delta, fi):
            'calculate dE/dA^(l-1)'
            sum = 0
            todo
        
        E = distance(y, self.result());
        Etotal = sum(E)
        print(Etotal)
        
        

net = Network([5, 10, 2])

net.forward(np.asmatrix([1,2,3,4,5]))

net.backward(np.asmatrix([0,1]))
