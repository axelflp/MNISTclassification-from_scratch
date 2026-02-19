import numpy as np
np.random.seed(0)

class MLP():
    def __init__(self, config):
        # Here we save the hidden states to compute the backpropagation 
        self.memory = dict()
        # Here we save the weights to learn, Ws are initialized with Xavier normal initalization
        self.model_weights = {
            "W_1": np.random.randn(config['input_size'], config['hidden_size_1'])*np.sqrt(2./(config['input_size']+config['hidden_size_1'])),
            "b_1": np.zeros((1, config['hidden_size_1'])),
            "W_2": np.random.randn(config['hidden_size_1'], config['hidden_size_2'])*np.sqrt(2./(config['hidden_size_1']+config['hidden_size_2'])),
            "b_2": np.zeros((1, config['hidden_size_2'])),
            "W_3": np.random.randn(config['hidden_size_2'], config['output_size'])*np.sqrt(2./(config['hidden_size_2']+config['output_size'])),
            "b_3": np.zeros((1, config['output_size']))}
        self.ReLU = lambda x : np.maximum(0., x)

    def update_weights(self, new_weights):
        self.model_weights = new_weights
        
    def forward(self, x):
        # First layer
        self.memory["z_1"] = x @ self.model_weights["W_1"] + self.model_weights["b_1"]
        self.memory["y_1"] = self.ReLU(self.memory["z_1"])
        # Second layer
        self.memory["z_2"] = self.memory["y_1"] @ self.model_weights["W_2"] + self.model_weights["b_2"]
        self.memory["y_2"] = self.ReLU(self.memory["z_2"])
        # Third layer
        self.memory["z_3"] = self.memory["y_2"] @ self.model_weights["W_3"] + self.model_weights["b_3"]
        
        return self.softMax(self.memory["z_3"])# , self.self.memory, self.model_weights

    def eval(self, x):
        # First layer
        z_1 = x @ self.model_weights["W_1"] + self.model_weights["b_1"]
        y_1 = self.ReLU(z_1)
        # Second layer
        z_2 = y_1 @ self.model_weights["W_2"] + self.model_weights["b_2"]
        y_2 = self.ReLU(z_2)
        # Third layer
        z_3 = y_2 @ self.model_weights["W_3"] + self.model_weights["b_3"]
        
        return np.argmax(self.softMax(z_3), axis=1)

    def softMax(self, x):
        x = np.exp(x)
        return x/(x.sum(axis=1, keepdims=True))