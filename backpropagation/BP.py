import numpy as np

def loss_function(y, y_pred):
    return -(y*np.log(y_pred)).sum()

class backPropagation():
    def __init__(self, lr):
        self.learning_rate = lr
        self.ReLU_derivative = lambda x : np.where(x > 0., 1., 0.)

    def backward(self, predictions, targets, x, weights, memory):
        grad_loss_y = predictions - targets

        # Gradients of W3 and b3
        grad_b3 = np.sum(grad_loss_y, axis=0)
        grad_W3 = memory["y_2"].T @ grad_loss_y
    
        # Gradients of W2 and b2
        aux2 = grad_loss_y @ weights["W_3"].T
        delta_2 = aux2 * self.ReLU_derivative(memory["z_2"])
    
        grad_b2 = np.sum(delta_2, axis=0)
        grad_W2 = memory["y_1"].T @ delta_2
    
        # Gradients of W1 and b1
        aux1 = delta_2 @ weights["W_2"].T
        delta_1 = aux1 * self.ReLU_derivative(memory["z_1"])
    
        grad_b1 = np.sum(delta_1, axis=0)
        grad_W1 = x.T @ delta_1
    
        # Update weights through gradient descent
        weights["W_3"] -= self.learning_rate * grad_W3
        weights["b_3"] -= self.learning_rate * grad_b3
        weights["W_2"] -= self.learning_rate * grad_W2
        weights["b_2"] -= self.learning_rate * grad_b2
        weights["W_1"] -= self.learning_rate * grad_W1
        weights["b_1"] -= self.learning_rate * grad_b1