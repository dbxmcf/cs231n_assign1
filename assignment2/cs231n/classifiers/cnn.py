from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # w: Filter weights of shape (F, C, HH, WW)
        # C = input_dim[0]
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        Hout, Wout = input_dim[1], input_dim[2]
        #Hout = 1 + (input_dim[1] - PH) / S
        #Wout = 1 + (input_dim[2] - PW) / S
        indim_W2 = num_filters*Hout/2*Wout/2
        print("indim_W2=",indim_W2)
        self.params['W2'] = weight_scale * np.random.randn(indim_W2,hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)
        #pass
        #in_dims = [input_dim] + hidden_dim
        #out_dims = hidden_dim + [num_classes]
        #all_dims = [input_dim] + hidden_dims + [num_classes]
        #for idx,(in_dim,out_dim) in enumerate(zip(in_dims,out_dims)):
        #    strW = 'W' + str(idx+1)
        #    strb = 'b' + str(idx+1)
        #    
        #    #out_dim = all_dims[idx+1]
        #    self.params[strW] = weight_scale * np.random.randn(in_dim, out_dim)
        #    self.params[strb] = np.zeros(out_dim)
        #    #print strW, strb, self.params[strW].shape, self.params[strb].shape
        #    # last layer does not have gamma and beta
        #    if self.use_batchnorm and (idx < self.num_layers-1):
        #        str_gamma = 'gamma' + str(idx+1)
        #        str_beta = 'beta' + str(idx+1)
        #        #print str_gamma, str_beta
        #        self.params[str_gamma] = np.ones(out_dim)
        #        self.params[str_beta]  = np.zeros(out_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        #a1, cache1 = conv_forward_naive(X, W1, b1, conv_param)
        #a1_rl, cache1_rl = relu_forward(a1)
        #a1_rl_mp, cache1_rl_mp = max_pool_forward_naive(a1_rl, pool_param)
        #print("a1_rl_mp.shape=",a1_rl_mp.shape)
        #a1_rl_mp_flt = a1_rl_mp.reshape(a1_rl_mp.shape[0],-1)reg = self.reg
        #print("a1_rl_mp_flt.shape=",a1_rl_mp_flt.shape)
        
        reg = self.reg
        a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        #print("a1.shape=",a1.shape)
        a1f = a1.reshape(a1.shape[0],-1)
        #print("a1f.shape=",a1f.shape)
        a2, cache2 = affine_relu_forward(a1f, W2, b2)
        scores, cache3 = affine_forward(a2, W3, b3)
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        
        da2, dW3, db3 = affine_backward(dscores, cache3)
        grads['W3'] = dW3 + reg*W3
        grads['b3'] = db3
        da1f, dW2, db2 = affine_relu_backward(da2, cache2)
        grads['W2'] = dW2 + reg*W2
        grads['b2'] = db2
        da1 = da1f.reshape(a1.shape)
        dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)
        grads['W1'] = dW1 + reg*W1
        grads['b1'] = db1
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
