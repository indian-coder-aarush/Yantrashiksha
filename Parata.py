import cupy as cp
import Tanitra
import math

class Parata:

     def __init__(self,input_shape = None):
         self.params = {}
         self.input_shape = input_shape
         self.output_shape = None

     def forward(self):
         raise NotImplementedError

class PraveshParata:

    def __init__(self,input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.params = {}

    def forward(self,data):
        if data.shape == self.input_shape:
            return data
        else:
            raise ValueError("Shape of input layer and the input data should be the same.")

class GuptaParata:

    def __init__(self, n_neurons, activation, input_shape =  None):
        if isinstance(n_neurons, tuple):
            raise ValueError("You can only have a 1d NirgamParata or the number of neurons should be an integer.")
        self.output_shape = (n_neurons,)
        self.params = {}
        self.n_neurons = n_neurons
        self.input_shape = None
        self.activation = activation
        self.input_output_learned = False

    def forward(self, input):
        if not self.input_output_learned:
            self.input_shape = input.shape
            self.params = {"weights": Tanitra.Tanitra(cp.random.randn(math.prod(self.input_shape), self.n_neurons) *
                                                      1 / cp.sqrt(math.prod(self.input_shape))),
                "biases": Tanitra.Tanitra(cp.random.randn(self.n_neurons) * 0.01)}
            self.input_output_learned  = True
        input = input.flatten()
        if Tanitra.length(input) != math.prod(self.input_shape):
            raise RuntimeError("you can only input an Tanitra of the shape of input_neurons")
        output =  input @ self.params['weights'] + self.params['biases']
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        else:
            raise RuntimeError("Invalid activation function was Input")
        return output

class NirgamParata:

    def __init__(self, n_neurons, activation, input_shape =  None):
        if isinstance(n_neurons, tuple):
            raise ValueError("You can only have a 1d NirgamParata or the number of neurons should be an integer.")
        self.output_shape = (n_neurons,)
        self.params = {}
        self.n_neurons = n_neurons
        self.input_shape = None
        self.activation = activation
        self.input_output_learned = False

    def forward(self, input):
        if not self.input_output_learned:
            self.input_shape = input.shape
            self.params = {"weights": Tanitra.Tanitra(cp.random.randn(math.prod(self.input_shape), self.n_neurons) *
                                                      1 / cp.sqrt(math.prod(self.input_shape))),
                "biases": Tanitra.Tanitra(cp.random.randn(self.n_neurons) * 0.01)}
            self.input_output_learned = True
        input = input.flatten()
        if Tanitra.length(input) != math.prod(self.input_shape):
            raise RuntimeError("you can only input an Tanitra of the shape of input_neurons")
        output =  input @ self.params['weights'] + self.params['biases']
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        elif self.activation == 'linear':
            pass
        else:
            raise RuntimeError("Invalid activation function was Input")
        return output

class Samasuchaka:

    def __init__(self,normalization_type):
        self.normalization = normalization_type
        if self.normalization == 'z-score':
            self.X_mean = None
            self.y_mean = None
            self.X_std = None
            self.y_std = None
        elif self.normalization == 'min-max':
            self.X_min = None
            self.y_min = None
            self.X_max = None
            self.y_max = None
        else:
            raise ValueError("Invalid normalizer was input. Choose one out of 'z-score','min-max'.")

    def learn(self,X,y):
        if self.normalization == 'z-score':
            self.X_mean  = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.y_mean = y.mean(axis=0)
            self.y_std = y.std(axis=0)
        if self.normalization == 'min-max':
            self.X_min = X.min(axis = 0)
            self.X_max = X.max(axis = 0)
            self.y_min = y.min(axis = 0)
            self.y_max = y.max(axis = 0)

    def forward(self,X,y):
        if self.X_mean is None or self.y_mean is None:
            raise RuntimeError("Samasuchaka must call `learn(X, y)` before `forward`")
        if self.normalization == 'z-score':
            return (X - self.X_mean) / (self.X_std+1e-8) , (y - self.y_mean) / (self.y_std+1e-8)
        if self.normalization == 'min-max':
            return (X-self.X_min)/(self.X_max-self.X_min+1e-8),(y-self.y_min)/(self.y_max-self.y_min+1e-8)

class ConvLayer2D:

    def __init__(self,stride,filters,kernel_size,activation,input_shape=None,padding_constant = 0,padding_mode = None,
                 padding_width = 0):
        self.input_output_learned = False
        self.stride = stride
        self.filters = filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.params = {}
        self.padding = padding_mode
        self.padding_width = padding_width
        self.padding_constant = padding_constant
        self.activation = activation
        self.output = None

    def forward(self,X):
        output = Tanitra.Tanitra([])
        if not self.input_output_learned:
            self.input_shape = X.shape
            self.output = (self.filters, (self.input_shape[0] - self.kernel_size) / self.stride + 1,
                           (self.input_shape[1] -self.kernel_size) / self.stride + 1)
            for i in range(self.filters):
                self.params['kernels'+str(i)] = Tanitra.Tanitra(cp.random.randn(self.kernel_size,self.kernel_size)
                                                /self.input_shape[0]*self.input_shape[1])
            self.input_output_learned = True
        for i in self.params:
            output = output.append(Tanitra.convolution2d(X , self.params[i],self.stride, padding_mode = self.padding,
                                            pad_width=self.padding_width,constant_values=self.padding_constant))
        if self.activation == 'relu':
            output = Tanitra.relu(output)
        elif self.activation == 'sigmoid':
            output = Tanitra.sigmoid(output)
        else:
            raise ValueError("invalid activation")
        return output

class MaxPoolingLayer2D:

    def __init__(self,stride,pool_window,padding_mode = None,pad_width = 0,pad_constants = 0,input_shape = None):
        self.params = {}
        self.stride = stride
        self.pool_window = pool_window
        self.padding = padding_mode
        self.pad_width = pad_width
        self.pad_constants = pad_constants
        self.input_shape = None
        self.output = None
        self.input_output_learned = False

    def forward(self,X):
        output = Tanitra.Tanitra([])
        for i in range(Tanitra.length(X)):
            output  = output.append(Tanitra.pooling2d(X[i],self.pool_window,self.stride,padding_mode=self.padding,
                                            pad_width=self.pad_width,constant_values=self.pad_constants))
        return output