import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
import time

class AdamANN_clf(BaseEstimator):
    '''
    ANN classifier with N layers, using Adam optimization and mini-batch 
    gradient descent.
    
    Architecture of the NN : 
    (Sigmoid or Tanh or ReLU) x N-1 times + (Sigmoid or Softmax) for output.
    
    The last layer's activation function is chosen automatically.
    -------------------------------------------------------------------------
    Hyperparameters
    -------------------------------------------------------------------------
        - hidden_units : List of number of units for the hidden layers
        - hidden_func : Activation function to be used for the hidden layers,
                        'sigmoid', 'tanh' or 'relu'
        - alpha : Regularization parameter of weight decay (l2)
        - epoch : Number of iteration through the training dataset
        - learning_rate : Learning rate for the gradient descent
        - learn_decay : Decay rate of the learning_rate parameter, if set to x
                    learning_rate will be decreased linearly to 1/x of his 
                    original value
        - batch_size : Size of the mini-batch for splitting into batches the 
                        training data for training
        - beta1 : Parameter for weight averaging the gradient descent
        - beta2 : Parameter for RMS prop.
        - hot_start : If True, refitting the estimator won't reinitialize the 
                        parameters
        - verbose : If True, print cost function value 10 times during training
        - grad_check : If True, chech implementation of backprop and print a number,
                    if lower than 1e-5, the implementation is right.
    -------------------------------------------------------------------------
    Method
    -------------------------------------------------------------------------
        - fit(X, y) : 
                Input
                    - X : Input data, numpy array shape (m, n_x) or (m,)
                    - y : Input target, numpy array of shape (m,)
                
        - predict(X) :
                Input
                    - X : Input data (m, n_x)
                Output
                    - y_pred : Predicted labels for X (m,)
    '''
    def __init__(self, hidden_units, hidden_func='relu', alpha=0, epoch=100, 
                 learning_rate=0.001, learn_decay=0, batch_size=256, beta1=0.9, 
                 beta2=0.999, hot_start=False, verbose=True, grad_check=False):
        
        self.alpha = alpha
        self.hidden_units = hidden_units
        self.hidden_func = hidden_func
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.learn_decay = learn_decay
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.hot_start = hot_start
        self.verbose = verbose
        self.grad_check = grad_check

        self.L = len(hidden_units) + 1
        
    def fit(self, X, y):
        '''
        Train the Artificial Neural Network.
        '''
        alpha = self.alpha
        hidden_units = self.hidden_units
        hidden_func = self.hidden_func
        epoch = self.epoch
        learning_rate = self.learning_rate
        learn_decay = self.learn_decay
        batch_size = self.batch_size
        hot_start = self.hot_start
        verbose = self.verbose
        grad_check = self.grad_check
        
        self.parameters, self.v_grads, self.s_grads = self.NN_model(X, y, hidden_units, hidden_func,
             alpha, epoch, learning_rate, learn_decay, batch_size, hot_start, verbose, grad_check)
        
        return self
    
    def predict(self, X):
        '''
        Predict target values of input X and return the prediction
        '''
        parameters = self.parameters
        hidden_func = self.hidden_func
        output_func = self.output_func
        
        y_pred = self.MakePrediction(X, parameters, hidden_func, output_func)
        
        return y_pred
    
    def ReLU(self, Z):
        '''
        Compute the ReLU of the matrix Z
        '''
        relu = np.maximum(0, Z)
        
        return relu
    
    def Sigmoid(self, Z):
        '''
        Compute the sigmoid of the matrix Z
        '''
        sigmoid = 1/(1+np.exp(-Z))
        
        return sigmoid
    
    def Softmax(self, Z):
        '''
        Compute the Softmax of the matrix Z
        '''
        exp_Z = np.exp(Z)
        softmax = exp_Z/np.sum(exp_Z, axis=0)
        
        return softmax
    
    def InitializeParameters(self, n_units_list, hidden_func):
        '''
        Initialize the parameters values W and b for each layers.
        --------
        Input  
                - n_units_list : list of number of units for each layers, input and
                                output included.
        Output
                - parameters : dictionnary of the parameters W and b
                               for each layers
        '''
        L = len(n_units_list) -1   # number of layers
        parameters = {}
        
        for l in range(1, L+1):
            n_l_prev = n_units_list[l-1]    # number of units in layer l-1
            n_l = n_units_list[l]           # number of units in layer l
            
            # initialize the parameters values randomly for W and 0 for b
            xavier_init = np.sqrt(1/n_l_prev)
            parameters['W' + str(l)] = np.random.randn(n_l, n_l_prev) * xavier_init
            if hidden_func=='relu':
                parameters['W' + str(l)] * np.sqrt(2)
                
            parameters['b' + str(l)] = np.zeros((n_l, 1))
            
            
        return parameters
    
    def ForwardProp(self, X, parameters, hidden_func, output_func):
        '''
        Compute the prediction matrix A3.
        --------
        Input
                - X : Matrix of input (n_x, m)
                - parameters : dictionnary of parameters W and b, for each layers
                - hidden_func : Activation function to be used for the hidden layers
                - output_func : Activation function to be used for the last layer
        Output
                - AL : Output matrix of last layer (n_y, m)
                - cache : Dictionnary of the A and Z, to use them during backprop
        '''
        L = len(parameters)//2
        cache = {}
        Al_prev = X
        for l in range(1,L):
            # get the parameters from the parameters dict
            Wl = parameters['W' + str(l)]
            bl = parameters['b' + str(l)]
            
            # compute forward propagation
            Zl = Wl.dot(Al_prev) + bl
            if hidden_func=='sigmoid':
                Al = self.Sigmoid(Zl)
            if hidden_func=='tanh':
                Al = np.tanh(Zl)
            if hidden_func=='relu':
                Al = self.ReLU(Zl)
            
            # write into the cache dict the Z and A
            cache['Z' + str(l)] = Zl
            cache['A' + str(l)] = Al
            
            # set Al_prev for next iter
            Al_prev = Al
            
        # compute forward prop for last layer
        WL = parameters['W' + str(L)]
        bL = parameters['b' + str(L)]
        ZL = WL.dot(Al_prev) + bL
        if output_func=='softmax':
            AL = self.Softmax(ZL)
        if output_func=='sigmoid':
            AL = self.Sigmoid(ZL)
            
        return AL, cache
    
    def ComputeCost(self, Y, AL, parameters, output_func, alpha):
        '''
        Compute the cost function.
        --------
        Input
                - Y : Target matrix (n_y, m)
                - AL : Output matrix of last layer (n_y, m)
                - output_func : Activation function to be used for the last layer
        Output
                - cost : the cost function computed for Y and AL
        '''
        m = Y.shape[1]
        
        
        
        if output_func=='sigmoid':
            loss = - Y * np.log(AL) - (1-Y) * np.log(1 - AL)
            # sum the loss through the m examples
            cost = np.sum(loss)/m
            
        if output_func=='softmax':
            loss = - Y * np.log(AL)
            # sum the loss through the m examples
            cost = np.sum(loss)/m
        
        # compute regularization part
        if alpha != 0:
            L = len(parameters)//2
            regul = 0
            for l in range(1, L+1):
                Wl = parameters['W'+str(l)]
                regul += (alpha/m) * np.sum(Wl**2)
            cost =  cost + regul
        
        
        return cost
    
    
    def BackProp(self, X, Y, AL, parameters, cache, hidden_func, output_func, alpha):
        '''
        Compute the gradients of the cost for the parameters W, b of each layers
        --------
        Input
                - X : Training data matrix of shape (n_x, m)
                - Y : Target data matrix of shape (n_y, m)
                - AL : Output from last layer
                - parameters : Parameters of the model W and b
                - cache : Cache data of Z and A
                - hidden_func : Activation function to be used for the hidden layers
                - output_func : Activation function to be used for the last layer
        Output
                - grads : dictionnary of the derivatives of the cost function
                          for each parameters
        '''
        m = X.shape[1]              # m = number of training examples
        L = len(parameters)//2       # L number of layer
        
        grads = {}
        
        # last layer
        # get dZL, depending of last layer activation fuction
        if output_func=='sigmoid':
            dZL = AL - Y
        if output_func=='softmax':
            dZL = AL - Y
        
        # get AL_prev to compute the gradients
        AL_prev = cache['A'+str(L-1)]
        WL = parameters['W'+str(L)]
        dWL = (1/m) * (dZL.dot(AL_prev.T) + alpha * 2 * WL)
        dbL = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        
        # write the gradients in grads dictionnary
        grads['dW'+str(L)] = dWL
        grads['db'+str(L)] = dbL
        
        # set dZl to dZL to be use as dZl_next for the first iter of the loop
        dZl = dZL
        
        # layer L-1 to 1
        for l in range(L-1,0,-1):
            # compute dAl
            Wl_next = parameters['W'+str(l+1)]
            dZl_next = dZl
            dAl = Wl_next.T.dot(dZl_next)
            
            # compute dZl
            Zl = cache['Z' + str(l)]
            if hidden_func=='sigmoid':
                sig = self.Sigmoid(Zl)
                dZl = sig * (1 - sig) * dAl
            if hidden_func=='tanh':
                dZl = (1 - np.tanh(Zl)**2) * dAl
            if hidden_func=='relu':
                dZl = ((Zl > 0)*1) * dAl
            
            # get Al_prev
            if l>1:
                Al_prev = cache['A'+str(l-1)]
            if l == 1:
                Al_prev = X
            
            Wl = parameters['W'+str(l)]
            # compute the gradients
            dWl = (1/m) * (dZl.dot(Al_prev.T) + alpha * 2 * Wl)
            dbl = (1/m) * np.sum(dZl, axis=1, keepdims=True)
            
            # write the gradients in grads dictionnary
            grads['dW'+str(l)] = dWl
            grads['db'+str(l)] = dbl
        
        return grads
    
    def UpdateParameters(self, parameters, grads, v_grads, s_grads, learning_rate, t):
        '''
        Update the parameters by gradient descent
        ---------
        Input
                - parameters : dictionnary of parameters W, b of each layer
                - grads : dictionnary of gradient of the cost function
                          for each parameters W, b of each leayer
                - learning_rate : learning rate to use for updating the parameters
        Output
                - parameters : parameters updated after gradient descent
        '''
        epsilon = 1e-8
        L = len(parameters)//2           # L number of layer
        v_grads_corrected = self.MomentumGradDescent(grads, v_grads, t)
        s_grads_corrected = self.RMSprop(grads, s_grads, t)
        
        for l in range(1, L+1):
            dW_Adam = v_grads_corrected['dW' + str(l)] / (np.sqrt(s_grads_corrected['dW' + str(l)]) + epsilon)
            parameters['W' + str(l)] -= learning_rate * dW_Adam
            
            db_Adam = v_grads_corrected['db' + str(l)] / (np.sqrt(s_grads_corrected['db' + str(l)]) + epsilon)
            parameters['b' + str(l)] -= learning_rate * db_Adam
            
        return parameters, v_grads, s_grads
    
    def InitializeAdamParameters(self):
        s_grads = {}
        v_grads = {}
        for l in range(1, self.L+1):
            s_grads['dW' + str(l)] = 0
            s_grads['db' + str(l)] = 0
            v_grads['dW' + str(l)] = 0
            v_grads['db' + str(l)] = 0
        
        return v_grads, s_grads
    
    def MomentumGradDescent(self, grads, v_grads, t):
        '''
        Compute gradient with momentum
        '''
        beta1 = self.beta1
        L = len(grads)//2           # L number of layer
        v_grads_corrected = {}
        
        for l in range(1, L+1):
            v_grads['dW' + str(l)] = beta1 * v_grads['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
            v_grads_corrected['dW' + str(l)] = v_grads['dW' + str(l)] / (1 - beta1**t) # bias correction
            
            v_grads['db' + str(l)] = beta1 * v_grads['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]
            v_grads_corrected['db' + str(l)] = v_grads['db' + str(l)] / (1 - beta1**t) # bias correction
        
        
        return v_grads_corrected
    
    def RMSprop(self, grads, s_grads, t):
        '''
        Compute Root Mean Squared prop
        '''
        L = len(grads)//2
        beta2 = self.beta2
        s_grads_corrected = {}
        
        for l in range(1, L+1):
            s_grads['dW' + str(l)] = beta2 * s_grads['dW' + str(l)] + (1 - beta2) * grads['dW' + str(l)]**2
            s_grads_corrected['dW' + str(l)] = s_grads['dW' + str(l)] / (1 - beta2**t) # bias correction
            
            s_grads['db' + str(l)] = beta2 * s_grads['db' + str(l)] + (1 - beta2) * grads['db' + str(l)]**2
            s_grads_corrected['db' + str(l)] = s_grads['db' + str(l)] / (1 - beta2**t) # bias correction
        
        return s_grads_corrected
    
    def PreProcess_X_Y(self, X, y):
        '''
        Transform the input training data into the correct form.
        ----------
        Input
                - X : Input data, numpy array shape (m, n_x) or (m,)
                - Y : Input target, numpy array of shape (m,)
        Output
                - X : New input data of shape (n_x, m)
                - Y : New input target of shape (n_y, m)
        '''
        # check that y is a 1D numpy array
        assert isinstance(y, (pd.core.series.Series, np.ndarray)) and len(y.shape) == 1, \
            'y must be a 1 dimensional numpy array or Series'
        # check that X is a 1D or 2D numpy array
        assert isinstance(X, (pd.core.series.Series, pd.core.frame.DataFrame, np.ndarray)) \
            and (len(y.shape) == 2 or len(y.shape) == 1), \
            'X must be a 1 or 2 dimensional numpy array or Series/Dataframe'
        
        # Make sure the data is in a numpy array form
        X = np.array(X)
        y = np.array(y)
            
        # if X is a 1D array, transform it in a 2D array (m,1)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # get the number of features n_x and number of examples m of X
        m = X.shape[0]
        
        # transform Y to a 2 dim array (m, n_y)
        K = len(np.unique(y)) # get number of classes
        if K==2:
            self.output_func = 'sigmoid'     # set activation function for last layer
            Y = y.reshape(-1,1)    # reshape Y into (m,1)
            
        if K>2:
            self.output_func = 'softmax'  # set activation function for last layer
            Y_dummy = np.zeros((m, K))
            for i in range(len(y)):
                Y_dummy[i, int(y[i])] = 1   # Y_dummy : (m, K)
            
            Y = Y_dummy
            
        X, Y = X.T, Y.T
        
        return X, Y
    
    def MakeMiniBatches(self, X, Y, batch_size):
        '''
        
        '''
        m = X.shape[1]
        idx = np.arange(0, m)
        np.random.shuffle(idx)
        
        # Shuffle X and Y (the same way)
        X_shuffled, Y_shuffled = X[:, idx], Y[:, idx]
        
        n_batch = m//batch_size
        
        minibatches = []
        a = 0
        for batch in range(n_batch):
            X_batch = X_shuffled[:, a:a+batch_size]
            Y_batch = Y_shuffled[:, a:a+batch_size]
            a += batch_size
            minibatches.append((X_batch, Y_batch))
            
        if m % batch_size !=0:
            X_batch = X_shuffled[:, a:]
            Y_batch = Y_shuffled[:, a:]
            minibatches.append((X_batch, Y_batch))
            
        return minibatches
    
    def NN_model(self, X, y, hidden_units, hidden_func, \
             alpha, epoch, learning_rate, learn_decay, batch_size, hot_start, verbose, grad_check):
        '''
        Train a Neural Network of 3 layers (2 layers ReLU and 1 sigmoid for the output).
        ----------
        Input
                - X : input training dataset (m, n_x)
                - y : target of the training dataset (m,)
                - hidden_units : list of number of units for the hidden layers
                - hidden_func : Activation function to be used for the hidden layers
                - output_func : Activation function to be used for the last layer
                - epoch : number of iteration
                - learning_rate : learning rate for the gradient descent
                - verbose : if True, print cost function value every 100 epoch
        Output
                - parameters : dictionnary of the trained parameters W, b for each layers
        '''
        t0 = time.time()
        
        # reshape and transform X and y
        X, Y = self.PreProcess_X_Y(X,y)
        
        # get architecture of the NN
        n_x = X.shape[0]
        n_y = Y.shape[0]
        n_units_list = [n_x] + hidden_units + [n_y]
        
        # initialize the parameters
        if not hot_start:
            parameters = self.InitializeParameters(n_units_list, hidden_func)
            v_grads, s_grads = self.InitializeAdamParameters()
            self.n_iter = 0
        
        if hot_start:
            try:
                parameters = self.parameters
                v_grads = self.v_grads 
                s_grads = self.s_grads
            except:
                parameters = self.InitializeParameters(n_units_list, hidden_func)
                v_grads, s_grads = self.InitializeAdamParameters()
                self.n_iter = 0
        
        # set output function (set during InitializeParameters)        
        output_func = self.output_func  
        
        if grad_check:
                AL, cache = self.ForwardProp(X, parameters, hidden_func, output_func)
                grads = self.BackProp(X, Y, AL, parameters, cache, hidden_func, output_func, alpha)
                error_ratio = self.GradCheck(X, Y, parameters, grads, hidden_func, output_func, alpha, n_units_list)
                print('Verification of gradient : ', error_ratio)
        
        
        
        # split data in mini batches of size batch_size
        minibatches = self.MakeMiniBatches(X, Y, batch_size)
        
        # initialize a list to plot the evolution of the cost function
        cost_list = []
        
        # n_cost :  number of time the cost will be computed and saved
        if epoch >= 100:
            cost_step = epoch//100
            print_step = epoch//10     # number of time it will be printed if verbose = True
        
        elif epoch >= 10:
            cost_step = 1
            print_step = epoch//10
        
        else:
            cost_step = 1
            print_step = 1
        
        x_iter = [] # to keep count of the iteration when cost is computed, for plotting
        
        for i in range(epoch):
            learning_rate = self.learning_rate / (1 + learn_decay*i/epoch) # decay of learning_rate
            for X, Y in minibatches:
                self.n_iter += 1
                
                # compute the forward propagation
                AL, cache = self.ForwardProp(X, parameters, hidden_func, output_func)
                
                # compute the back propagation
                grads = self.BackProp(X, Y, AL, parameters, cache, hidden_func, output_func, alpha)
                
                # update the parameters
                t = self.n_iter
                
                parameters, v_grads, s_grads = self.UpdateParameters(parameters, grads, v_grads, s_grads, learning_rate, t)
                self.parameters, self.v_grads, self.s_grads = parameters, v_grads, s_grads
            
            if  i%cost_step == 0:
                # compute the cost function
                AL, _ = self.ForwardProp(X, parameters, hidden_func, output_func)
                cost = self.ComputeCost(Y, AL, parameters, output_func, alpha)
                cost_list.append(cost)
                x_iter.append(i)
                
                if verbose and (i%print_step == 0):
                    print('Cost function after epoch {} : {}'.format(i, cost))
        
        self.learning_rate = learning_rate # update self.learning_rate for hot start
        
        cost = self.ComputeCost(Y, AL, parameters, output_func, alpha)
        cost_list.append(cost)
        x_iter.append(i)
        print('Cost function after epoch {} : {}'.format(i+1, cost))
        print('Time : %.3f s' % (time.time()-t0))
        
        # print the cost function for each iterations
        plt.figure()
        plt.plot(x_iter, cost_list)
        plt.title('Cost function')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost Function')
        
        return parameters, v_grads, s_grads
    
    def MakePrediction(self, X, parameters, hidden_func, output_func):
        '''
        Make prediction of the data X
        ---------
        Input
                - X : Input data (m, n_x)
                - parameters : parameters W, b of each layers of the NN model
        Output
                - y_pred : Predicted labels for X (m,)
        '''
        X = X.T
        A3, _ = self.ForwardProp(X, parameters, hidden_func, output_func)
        
        if output_func=='softmax':
            y_pred = np.argmax(A3, axis=0)
    
        if output_func=='sigmoid':
            y_pred = ((A3 >0.5)*1).reshape(-1)
            
        return y_pred

    def GradCheck(self, X, Y, parameters, grads, hidden_func, output_func, alpha, n_units_list):
        '''
        Check implementation of backprop by comparing the gradient from backprop
        with the gradient estimated using the cost function.
        '''
        
        epsilon = 1e-7
        
        # Gradient computed by backprop
        vec_grads = self.gradients_to_vector(grads)
        
        vec_parameters, keys = self.dictionary_to_vector(parameters)
        
        vec_grads_approx = np.zeros(vec_parameters.shape)
        for i in range(len(vec_parameters)):
            
            # Add epsilon to one element of the parameters
            vec_parameters_plus = vec_parameters.copy()
            vec_parameters_plus[i] += epsilon
            dict_parameters_plus = self.vector_to_dictionary(vec_parameters_plus, n_units_list)
            
            AL_plus, _ = self.ForwardProp(X, dict_parameters_plus, hidden_func, output_func)
            cost_plus = self.ComputeCost(Y, AL_plus, dict_parameters_plus, output_func, alpha)

            # Do the same for cost_minus
            vec_parameters_minus = vec_parameters.copy()
            vec_parameters_minus[i] -= epsilon
            dict_parameters_minus = self.vector_to_dictionary(vec_parameters_minus, n_units_list)
            
            AL_minus, _ = self.ForwardProp(X, dict_parameters_minus, hidden_func, output_func)
            
            cost_minus = self.ComputeCost(Y, AL_minus, dict_parameters_minus, output_func, alpha)
        
            grad = (cost_plus - cost_minus)/(2*epsilon)
            
            vec_grads_approx[i] = grad
            #print(keys[i], vec_grads_approx[i] - vec_grads[i], vec_grads_approx[i], vec_grads[i])
        
        
        assert vec_grads.shape == vec_grads_approx.shape

        num = np.linalg.norm(vec_grads_approx - vec_grads)
        denum = np.linalg.norm(vec_grads_approx + vec_grads)
        
        error_ratio = num / denum
        
        return error_ratio
    
    
    def dictionary_to_vector(self, parameters):
        """
        Roll all our parameters dictionary into a single vector satisfying our specific required shape.
        """
        count = 0
        L = len(parameters)//2
        keys = []
        for l in range(1, L+1):
            for par in ('W', 'b'):
                key = par + str(l)

                # flatten parameter
                new_vector = np.reshape(parameters[key], (-1))
                
                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector))
                count = count + 1
                
                for i in new_vector:
                    keys.append(key)
    
        return theta, keys
    
    def vector_to_dictionary(self, theta, n_units_list):
        """
        Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
        """
        parameters = {}
        n2 = 0
        L = len(n_units_list) - 1
        for l in range(1, L+1):
            n_l_prev = n_units_list[l-1]
            n_l = n_units_list[l]
            
            n0 = n2
            n1 = n0 + n_l * n_l_prev
            n2 = n1 + n_l
            
            parameters['W'+str(l)] = theta[n0:n1].reshape(n_l, n_l_prev)
            parameters['b'+str(l)] = theta[n1:n2].reshape(n_l, 1)

        return parameters
    
    def gradients_to_vector(self, grads):
        """
        Roll all our gradients dictionary into a single vector satisfying our specific required shape.
        """
        
        count = 0
        L = len(grads)//2
        
        for l in range(1, L+1):
            for par in ('dW', 'db'):
                key = par + str(l)
                # flatten parameter
                new_vector = np.reshape(grads[key], (-1))
                
                if count == 0:
                    theta = new_vector
                else:
                    theta = np.concatenate((theta, new_vector))
                count = count + 1
    
        return theta