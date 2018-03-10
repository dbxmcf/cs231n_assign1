import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  for i in xrange(num_train):
    scores = X[i].dot(W)

    #print X[i].shape, W.shape
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      #if i==499:
      #  print 'margin[',499,']=',margin
      if margin > 0:
        loss += margin
        # print dW[j].shape, X[i].T.shape
        dW[:,j] += X[i].T
        dW[:,y[i]] += -X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW /= num_train
  dW += 2.0*reg*W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  score = X.dot(W)
  idx_x = xrange(num_train)
  correct_class_score = score[idx_x, y].reshape(num_train,-1)
  margin = score - correct_class_score + 1
  #print margin.shape
  margin[idx_x,y] = 0.0
  mask = margin>0
  loss = np.sum(margin[mask])/num_train
  loss += reg * np.sum(W * W)
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  margin1 = mask.astype(int)
  #print margin1.shape
  #dW = (X.T).dot(margin1)
  #print dW.shape
  #a = np.sum(dW,axis=1)
  #print a.shape
  #idx_d = np.array(xrange(W.shape[0]))
  #print idx_d
  msum = margin1.sum(axis=1)
  margin1[idx_x,y] = -msum
  dW = (X.T).dot(margin1)
  dW /= num_train
  dW += 2.0*reg*W
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
