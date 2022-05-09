import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class _MLPBase(nn.Module):
    """
    Base class for all the Multi Layer Perceptron (MLP) modules.
    
    Parameters
    ----------
    input_dim: int
        Input dimensions (1D, 2D or 3D). Note that the batch dimension is not counted.
    
    layer_sizes: list of ints
        List of sizes of each MLP layer (the first one is the in_features).
        
    add_bias: None, bool or list of bools
        Flag for adding a bias term to each MLP layer.
        Note that if batch normalization is used then there is no need to add bias.
        
    apply_bn: None, bool or list of bools
        Flag for applying batch normalization to each MLP layer.
        
    activation_fn: None, str or list of strs
        Strings that specify the activation functions for each MLP layer.
    
    dropout_probs: None, float or list of floats in range [0.0, 1.0]
        Dropout probabilities (for dropping elements).
    """
    
    # Dictionary of available activation functions (you can add more)
    ACTIVATIONS = {"": None,
                   "relu": nn.ReLU(),
                   "lrelu": nn.LeakyReLU(),
                   "tanh": nn.Tanh(),
                   "sigmoid": nn.Sigmoid(),
                   "softmax": nn.Softmax()}
    
    def __init__(self, input_dim, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs):
        
        if not isinstance(input_dim, int) or input_dim < 1 or input_dim > 3:
            raise ValueError("The input_dim can be an int from {1, 2, 3}.")
                               
        # Checks for layer_sizes
        if not isinstance(layer_sizes, list) or not all(isinstance(s, int) and s > 0 for s in layer_sizes):
            raise TypeError("The layer_sizes must be a list of positive integers.")
        else:
            self.num_layers = len(layer_sizes) - 1
            if self.num_layers < 1:
                raise ValueError("The layer_sizes must contain at least two element.")
                
        # Checks for add_bias
        if add_bias is None or isinstance(add_bias, bool):
            add_bias = [add_bias] * self.num_layers
        elif isinstance(add_bias, list):
            if len(add_bias) != self.num_layers:
                raise ValueError(
                    f"The list add_bias should have size {self.num_layers} (same as number of layers).")
            elif not all(isinstance(b, bool) for b in add_bias):
                raise TypeError("The entries of add_bias must be all booleans.")
        else:
            raise TypeError(
                f"The add_bias can be None, a bool or a list of size {self.num_layers} containing booleans.")
            
        # Checks for apply_bn
        if apply_bn is None or isinstance(apply_bn, bool):
            apply_bn = [apply_bn] * self.num_layers
        elif isinstance(apply_bn, list):
            if len(apply_bn) != self.num_layers:
                raise ValueError(
                    f"The list apply_bn must have size {self.num_layers} (same as number of layers).")
            elif not all(isinstance(b, bool) for b in apply_bn):
                raise TypeError("The entries of the apply_bn must be all booleans.")
        else:
            raise TypeError(
                f"The apply_bn can be None, a bool or a list of size {self.num_layers} containing booleans.")

        # Checks for activation_fn
        activation_keys = set(self.ACTIVATIONS.keys())
        if activation_fn is None:
            activation_fn = [False] * self.num_layers
        elif isinstance(activation_fn, str):
            if activation_fn not in activation_keys:
                raise ValueError(f"The string activation_fn must be one of the {activation_keys}.")
            else:
                activation_fn = [activation_fn] * self.num_layers
        elif isinstance(activation_fn, list):
           if len(activation_fn) != self.num_layers:
               raise ValueError(
                   f"The list activation_fn must have size {self.num_layers} (same as number of layers).")
           elif not all(isinstance(a, str) and a in activation_keys for a in activation_fn):
               raise ValueError(f"The entries of the activation_fn must be strings from {activation_keys}.")
        else:
            raise TypeError(
                f"The activation_fn can be None, a strting or a list of size {self.num_layers} "
                f"containing strings from {activation_keys}.")

        # Checks for dropout_probs
        if dropout_probs is None:
            dropout_probs = [False] * self.num_layers
        elif isinstance(dropout_probs, float):
            if dropout_probs < 0.0 or dropout_probs > 1.0:
                raise ValueError("The float dropout_probs should be in range [0.0, 1.0].")
            else:
                dropout_probs = [dropout_probs] * self.num_layers
        elif isinstance(dropout_probs, list):
            if len(dropout_probs) != self.num_layers:
                raise ValueError(
                    f"The list dropout_probs must have size {self.num_layers} (same as number of layers).")
            elif not all(isinstance(p, float) and p >= 0.0 and p <= 1.0 for p in dropout_probs):
                raise ValueError("The entries of the dropout_probs must be floats in range [0.0, 1.0].")
        else:
            raise TypeError(
                f"The dropout_probs can be None, a float or a list of size {self.num_layers} "
                f"containing floats in range [0.0, 1.0].")    
        
        super().__init__()
        self.input_dim = input_dim
        self.in_features = layer_sizes[0]
        self.out_features = layer_sizes[-1]
        self.layer_sizes = layer_sizes
        self.add_bias = add_bias
        self.apply_bn = apply_bn
        self.activation_fn = activation_fn
        self.dropout_probs = dropout_probs
        
        self.net = self._build_model()
        
        
    def _build_model(self):

        layers = OrderedDict()
        for i in range(1, len(self.layer_sizes)):
            # 1D input of size (batch_size, in_features)
            if self.input_dim == 1:
                linear = nn.Linear(in_features=self.layer_sizes[i-1],
                                   out_features=self.layer_sizes[i],
                                   bias=self.add_bias[i-1])
            # 2D input of size (batch_size, in_features, num_points)
            elif self.input_dim == 2:
                linear = nn.Conv1d(in_channels=self.layer_sizes[i-1],
                                   out_channels=self.layer_sizes[i],
                                   kernel_size=1,
                                   bias=self.add_bias[i-1])
            # 3D input of size (batch_size, in_features, num_groups, num_samples_max)
            else:
                linear = nn.Conv2d(in_channels=self.layer_sizes[i-1],
                                   out_channels=self.layer_sizes[i],
                                   kernel_size=1,
                                   bias=self.add_bias[i-1])
                
            # i-th fully-connected layer
            layer = OrderedDict()
            layer[f"linear{i}"] = linear
                
            # i-th batch normalization layer
            if self.apply_bn[i-1]:
                if self.input_dim == 1 or self.input_dim == 2:
                    # bn with default values eps=1e-05, momentum=0.1, affine=True
                    bn = nn.BatchNorm1d(self.layer_sizes[i])              
                else:
                    bn = nn.BatchNorm2d(self.layer_sizes[i])      
                layer[f"bn{i}"] = bn
                
            # i-th activation function
            if self.activation_fn[i-1]:
                activation = self.ACTIVATIONS[self.activation_fn[i-1]]          
                layer[f"activation{i}"] = activation
                
            # i-th dropout layer
            if self.dropout_probs[i-1]:
                dropout = nn.Dropout(p=self.dropout_probs[i-1])     
                layer[f"dropout{i}"] = dropout
                
            # Gather them into the i-th MLP layer
            layers[f"layer{i}"] = nn.Sequential(layer)
            
        return nn.Sequential(layers)
        
                
    def _check_input(self, x):       
        
        # Check the dimensionality of the input. Note that the first dimension is reserved for the batch.
        if x.dim() != self.input_dim + 1:
            raise ValueError(f"The input must be a {self.input_dim + 1}-dimensional tensor "
                             f"but got a {x.dim()}-dimensional tensor instead.")
        if x.shape[1] != self.in_features:
            raise ValueError(f"The number of input features must be {self.in_features} "
                             f"but got x.shape[1] = {x.shape[1]} instead.")
            
        # Check if the batch normalization can be applied.
        # Note that BN statistics are computed per feature/channel, over all the other dimensions.
        if self.net.training:
            size_prod = np.prod([x.shape[i] for i in range(x.dim()) if i != 1])
            if any(self.apply_bn) and size_prod == 1:
                raise ValueError("Expected more than 1 values per feature for the batch "
                                 "normalization to be applied (in training mode).")
            

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features) or (batch_size, in_features, num_points)
            or (batch_size, in_features, num_groups, num_samples_max).

        Returns
        -------
        y : torch.Tensor
            Output tensor of shape same as the input's shape.
        """
        self._check_input(x)
        y = self.net(x)
        return y
        

class MLP1d(_MLPBase):
    """
    Multi Layer Perceptron (MLP) for 1D input of shape (batch_size, in_features).
    """        
    def __init__(self, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs):
        super().__init__(1, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs)    


# PointMLP
class MLP2d(_MLPBase):
    """
    Multi Layer Perceptron (MLP) for 2D input of shape (batch_size, in_features, num_points).
    Applied identically and independently to all the points.
    """        
    def __init__(self, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs):
        super().__init__(2, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs)
     

# PointGroupMLP
class MLP3d(_MLPBase):
    """
    Multi Layer Perceptron (MLP) for 3D input of shape (batch_size, in_features, num_groups, num_group_points).
    Applied identically and independently to all the points of each group.
    """        
    def __init__(self, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs):
        super().__init__(3, layer_sizes, add_bias, apply_bn, activation_fn, dropout_probs)
   


if __name__ == "__main__":
    
    in_features = 3
    out_features = 13
    x = torch.rand(2, in_features)
    mlp_1d = MLP1d(layer_sizes=[in_features, 5, out_features],
                   add_bias=False,
                   apply_bn=True,
                   activation_fn=['relu', 'sigmoid'],
                   dropout_probs=[0.6, 0.0])
    # mlp_1d.eval()
    y = mlp_1d(x)
    print(y.shape)
           
    x = torch.rand(2, in_features, 10)
    mlp_2d = MLP2d(layer_sizes=[in_features, 5, out_features],
                   add_bias=True,
                   apply_bn=False,
                   activation_fn=['relu', 'sigmoid'],
                   dropout_probs=None)
    y = mlp_2d(x)
    print(y.shape)

    x = torch.rand(2, in_features, 4, 4)
    mlp_3d = MLP3d(layer_sizes=[in_features, 5, out_features],
                   add_bias=False,
                   apply_bn=True,
                   activation_fn=['relu', 'sigmoid'],
                   dropout_probs=[0.6, 0.0])
    y = mlp_3d(x)
    print(y.shape)
    