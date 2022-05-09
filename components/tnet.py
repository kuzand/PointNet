from mlp import MLP1d, MLP2d
import torch
import torch.nn as nn


class _TNetBase(nn.Module):
    """
    Base class for the Transformation Network (TNet).
    
    Each point-cloud of a batch is transformed by its own transformation
    matrix of dimension mat_size x mat_size.
    
    Note that only the first mat_size features of each point are transformed.
    E.g. if the points have 6 features (x, y, z, r, g, b) and mat_size = 3,
    then only the (x, y, z) will be transformed.
        
    Parameters
    ----------       
    mat_size : int
        Size of the (square) transformation matrix.
          
    point_mlp : MLP2d
        Point MLP that is applied on each point, before the aggregation function.

    agg_fn : str 
        Aggregation function from {'max', 'avg'}.
        
    mlp : MLP1d
        MLP applied on the global feature vector, after the aggregation function.
    """
    
    # Aggregation is done over the points (num_points -> 1)
    AGG = {"max": nn.AdaptiveMaxPool1d(output_size=1),
           "avg": nn.AdaptiveAvgPool1d(output_size=1)}
    
    def __init__(self, mat_size, point_mlp, agg_fn, mlp):
        
        # Checks
            
        if not isinstance(mat_size, int) or mat_size < 1:
            raise TypeError("The mat_size must be a positive integer > 0.")
                    
        if not isinstance(mlp, MLP1d):
            raise TypeError("The mlp must be an instance of the MLP1d class.")
        
        if mlp.out_features != mat_size**2:
            raise ValueError(
                f"The output size of the mlp must be equal to mat_size^2 = {mat_size**2}.")
            
        if not isinstance(point_mlp, MLP2d):
            raise TypeError("The point_mlp must be an instance of the MLP2d class.")
            
        if point_mlp.out_features != mlp.in_features:
            raise ValueError("point_mlp.out_features should be equal to mlp.in_features.")
            
        if point_mlp.in_features < mat_size:
            raise ValueError("point_mlp.in_features should be greater or equal to mat_size.")
            
        if not isinstance(agg_fn, str) or agg_fn not in self.AGG.keys():
            raise TypeError(f"The agg_fn must be a string from {set(self.AGG.keys())}.")
                 
        # Initialization
        super().__init__()
        self.mat_size = mat_size
        self.point_mlp = point_mlp
        self.agg_fn = self.AGG[agg_fn]
        self.mlp = mlp
        
        # The transformation matrix is initialized as identity transformation.
        nn.init.constant_(self.mlp.net[-1][0].weight, 0.0)
        nn.init.eye_(self.mlp.net[-1][0].bias.view(mat_size, mat_size))      
                   

    def forward(self, x):
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features, num_points) with in_features >= mat_size.

        Returns
        -------
        x_transformed : torch.Tensor
            Transformed input x, same shape as input.
            
        transform_mat : torch.Tensor
            Transformation matrix of shape (batch_size, mat_size, mat_size).
        """
        
        batch_size = x.shape[0]
 
        transform_mat = self.point_mlp(x)  # (batch_size, point_mlp.out_features, num_points)
        transform_mat = self.agg_fn(transform_mat).squeeze(dim=-1)  # (batch_size, point_mlp.out_features)
        transform_mat = self.mlp(transform_mat)  # (batch_size, mat_size**2)
        transform_mat = transform_mat.view(batch_size, self.mat_size, self.mat_size)  # (batch_size, mat_size, mat_size)
        
        if x.shape[1] == self.mat_size:
            x_transformed = torch.bmm(transform_mat, x)
        else:
            # In this case x.shape[1] > self.mat_size, so we transform only the first mat_size features
            x_transformed = torch.cat([torch.bmm(transform_mat, x[:, :self.mat_size, :]), x[:, self.mat_size:, :]], dim=1)
            
        return x_transformed, transform_mat



class TNet(_TNetBase):
    """
    The original TNet from the PointNet paper.
    """
    
    def __init__(self, in_features, mat_size, agg_fn="max"):
        
        point_mlp = MLP2d(layer_sizes=[in_features, 64, 128, 1024],
                          add_bias=False,
                          apply_bn=True,
                          activation_fn="relu",
                          dropout_probs=None)
        
        mlp = MLP1d(layer_sizes=[1024, 512, 256, mat_size**2],
                    add_bias=[False, False, True],
                    apply_bn=[True, True, False],
                    activation_fn=["relu", "relu", ""],
                    dropout_probs=None)
        
        super().__init__(mat_size, point_mlp, agg_fn, mlp)
        
    
 
if __name__ == "__main__":
    
    x = torch.rand(2, 3, 100)
    
    tnet_3d = TNet(in_features=3, mat_size=2, agg_fn="max")
    tnet_3d.eval()
    
    x_transformed, transform_mat = tnet_3d(x)
    print(f"x_transformed shape: {tuple(x_transformed.shape)}")
    print(f"transform_mat shape: {tuple(transform_mat.shape)}")