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
        self.fi={} # weights of network
        self.a = {} # output activation from neurons
        self.delta = {} # delta function
        self.df = {} # corection for weights
        self.L = len(layer_sizes) # number of layers: including input and output

        # init by random values
        for l in range(1, self.L):
            random_array = np.random.rand(layer_sizes[l], layer_sizes[l-1] + 1)
            fi = np.asmatrix(random_array)
            self.fi[l] = fi 
            # print("Layer : ", l, ", Generated matrix shape: ", np.shape(fi))
            
    def result(self):
        'calculated result of forward propogation'
        return self.a[self.L-1]

    def forward(self, x):
        'calculate the result of forward propogation'
        self.a[0] = np.matrix.transpose(x)
        for l in range(1, self.L):
            inp = np.vstack([[1], self.a[l-1]])
            z = self.fi[l] * inp
            self.a[l] = g(z)
        #print ("Result = ", self.result())
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

        L = self.L - 1
        delta = calcdeltal(y, self.a[L])
        
        self.delta[L] = delta;

        a = np.vstack([[1], self.a[L-1]])
        df = delta * np.matrix.transpose(a)
        self.df[L] = df
        
        # Calculate backprop for all hidden layers

        for l in range(L-1, 0, -1):
            #print (l)
            fi_l = self.fi[l+1]
            delta_l = self.delta[l+1]
            sum_d_fi = np.matrix.transpose(fi_l) * delta_l

            #sum_d_fi = np.matrix.transpose(np.matrix.transpose(delta_l) * fi_l)

            @np.vectorize
            def calcdelta(s, a):
                'Calculate delta on layer l (hidden layer)'
                return s * dg(a)

            a = np.vstack([[1], self.a[l]])
            delta = calcdelta(sum_d_fi, a)
            delta = delta[1:,:] # throw away first index (TODO: need to skip it on delta sums calculation)
            self.delta[l] = delta

            a = np.vstack([[1], self.a[l-1]])
            df = delta * np.matrix.transpose(a)

            self.df[l] = df
            #print("Layer: ", l)
            #print("Correction: ", df)
            #print("df = ", np.shape(self.df[l]))
            #print("f = ", np.shape(self.df[l]))

        # adjust weights

        for l in range(1, self.L):
            self.fi[l] -= 10*self.df[l]
               
        E = distance(y, self.result());
        Etotal = sum(E)
        print(Etotal)
        
        

net = Network([2, 10, 11, 10, 1])

for i in range(0, 10000):
    net.forward(np.asmatrix([1, 1]))
    net.backward(np.asmatrix([1]))
    net.forward(np.asmatrix([0, 0]))
    net.backward(np.asmatrix([0]))
    net.forward(np.asmatrix([1, 0]))
    net.backward(np.asmatrix([0]))
    net.forward(np.asmatrix([0, 1]))
    net.backward(np.asmatrix([0]))
