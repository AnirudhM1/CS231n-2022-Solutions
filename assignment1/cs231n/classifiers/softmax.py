from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for x_i, y_i in zip(X, y):
      logits = x_i.dot(W)
      logits -= np.max(logits) # For numerical stability
      probabilities = np.exp(logits) / np.sum(np.exp(logits))
      loss += -np.log(probabilities[y_i]).item()

      diff = probabilities
      diff[y_i] -= 1
      dW += np.expand_dims(x_i, -1) @ np.expand_dims(diff, 0)

    loss /= X.shape[0]
    loss += reg * np.square(W).sum()

    dW /= X.shape[0]
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    logits = X @ W
    logits -= np.max(logits, axis=1, keepdims=True) # For numerical stability
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    loss = -np.mean(np.log(probabilities[np.arange(X.shape[0]), y]))
    loss += reg * np.square(W).sum()

    diff = probabilities
    diff[np.arange(X.shape[0]), y] -= 1

    dW = X.T @ diff
    dW /= X.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
