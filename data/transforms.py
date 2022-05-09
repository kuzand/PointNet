import numpy as np
import torch
import torch.nn as nn
from data_utils import check_pc


def normalize_pc(pc):
    """
    Centers and normalizes the input point-cloud pc (its (x,y,z) coordinates) into unit sphere.

    Parameters
    ----------    
    pc : torch.Tensor
        The input point-cloud tensor of shape (in_features, num_points) with in_features = 3.
        The features are the (x,y,z) coordinates of the points.
        
    Returns
    -------
    pc_normalized : torch.Tensor
        The centered and normalized point-cloud tensor of shape (in_features, num_points) with in_features = 3,
    """
    
    check_pc(pc, tensor_dims=2, xyz_only=True)
    
    centroid = torch.mean(pc, dim=-1, keepdim=True)
    pc_centered = pc - centroid
    
    dists = torch.sqrt(torch.sum(pc_centered**2, dim=0))
    pc_normalized = pc_centered / torch.max(dists)
    
    return pc_normalized


def angle_axis_to_rot_mat(angle, axis_vec):
    """
    Returns rotation matrix computed from axis_vec and angle.
    
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    
    Parameters
    ----------
    angle : float
        Angle of rotation in radians.
    axis_vec : np.ndarray or list
        Axis vector to rotate around.

    Returns
    -------
    R : torch.Tensor
        Rotation matrix of shape (3, 3).
    """

    # Arguments check
    if not isinstance(angle, float):
        raise TypeError("The angle should be a float.")
        
    if isinstance(axis_vec, list):
        if len(axis_vec) != 3:
            raise ValueError("The axis_vec should have length 3.")
        axis_vec = np.array(axis_vec)
    elif isinstance(axis_vec, np.ndarray):
        if axis_vec.ndim != 1 or len(axis_vec) != 3:
            raise ValueError("The axis_vec should be a 1D numpy array of length 3.")
    else:
        raise TypeError(
            "The axis_vec can be a 1D numpy array or a list of length 3.")
        
    # Normalize
    norm = np.linalg.norm(axis_vec)
    if norm > 0:
        u = axis_vec / norm
    else:
        raise ValueError("The given axis_vec is a zero vector.")
    
    u_x = u[0]
    u_y = u[1]
    u_z = u[2]
    
    cos_val = np.cos(angle)
    sin_val =  np.sin(angle)
    
    u_cross = np.array([[ 0.0, -u_z,  u_y],
                        [ u_z,  0.0, -u_x],
                        [-u_y,  u_x,  0.0]])
    
    u_outer = np.array([[ u_x * u_x,  u_x * u_y, u_x * u_z],
                        [ u_x * u_y,  u_y * u_y, u_y * u_z],
                        [ u_x * u_z,  u_y * u_z, u_z * u_z]])
    
    I = np.eye(3)
    
    # Compute rotation matrix
    R = cos_val * I + sin_val * u_cross + (1 - cos_val) * u_outer  # (3, 3)
    R = torch.tensor(R, dtype=torch.float32)
    
    return R


def rotate_pc(pc, angle, axis_vec):
    """
    Rotates the input point-cloud pc by an angle around a axis_vec in 3D space.
    
    Note that the features of the pc are assumed to be the (x, y, z) coordinates of the points.
    
    Parameters
    ----------
    pc : torch.Tensor
        Point-cloud tensor of shape (in_features, num_points) with in_features = 3.
    angle : float
        Angle of rotation in radians.
    axis_vec : list or np.ndarray
        3D axis-vector to rotate around.

    Returns
    -------
    pc_rotated : torch.Tensor
        Rotated point-cloud.
    """
    
    check_pc(pc, tensor_dims=2, xyz_only=True)
    
    if angle == 0.0:
        return pc
    
    # Create the rotation matrix
    R = angle_axis_to_rot_mat(angle, axis_vec)  # (3, 3)
    
    # Rotate the input point cloud
    pc_rotated = torch.matmul(R, pc)

    return pc_rotated


def translate_pc(pc, translation_vec):
    """
    Translates the input point-cloud pc by a translation_vec.
    
    Parameters
    ----------
    pc : torch.Tensor
        Point-cloud tensor of shape (in_features, num_points) with in_features = 3.
    translation_vec : list or torch.Tensor
       3D translation vector.

    Returns
    -------
    pc_translated : torch.Tensor
        Translated point-cloud.
    """
    
    check_pc(pc, tensor_dims=2, xyz_only=True)
    
    if isinstance(translation_vec, list):
        if len(translation_vec) != 3:
            raise ValueError("translation_vec should be of length 3.")
        translation_vec = torch.tensor(translation_vec)
    elif isinstance(translation_vec, torch.Tensor):
        if translation_vec.dim() != 1 or len(translation_vec) != 3:
            raise ValueError("translation_vec should be 1d torch tensor of length 3.")
    else:
        raise TypeError("translation_vec can be a 1d torch tensor or a list of length 3.")
    
    pc_translated = pc + translation_vec.unsqueeze(1)
    
    return pc_translated
    

class Compose:
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, pc):
        for t in self.transforms:
            pc = t(pc)
        return pc


    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
      

class ToTensor(nn.Module):
    """
    Converst the input point-cloud pc from numpy ndarray to torch tensor.
    """
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self, pc):    
        """
        Parameters
        ----------    
        pc : torch.Tensor
            The input point-cloud tensor of shape (in_features, num_points).

        Returns
        -------
        pc_tensor : torch.Tensor
            Normalized point-cloud tensor of shape (in_features, num_points).
        """
        
        check_pc(pc, tensor_dims=2, xyz_only=False)
        
        pc_tensor = torch.from_numpy(pc).contiguous()
        return pc_tensor
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
    
    
class Normalize(nn.Module):
    """
    Centers and normalizes the input point-cloud pc into unit sphere.
    """
    
    def __init__(self):
        super().__init__()
        
        
    def forward(self, pc):    
        """
        Parameters
        ----------    
        pc : torch.Tensor
            The input point-cloud tensor of shape (in_features, num_points) with in_features = 3.

        Returns
        -------
        pc_normalized : torch.Tensor
            Normalized point-cloud tensor of shape (in_features, num_points) with in_features = 3.
        """
        
        check_pc(pc, tensor_dims=2, xyz_only=True)
        
        pc_normalized = normalize_pc(pc)
        return pc_normalized
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"
        
        
class RandomSample(nn.Module):
    """
    Randomly samples num_samples points from the input point-cloud pc.
    """
    
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        

    def forward(self, pc):    
        """
        Parameters
        ----------    
        pc : torch.Tensor
            The input point-cloud tensor of shape (in_features, num_points).

        Returns
        -------
        pc_sampled : torch.Tensor
            Sampled point-cloud tensor of shape (in_features, num_points).
        """
        
        check_pc(pc, tensor_dims=2, xyz_only=False)
        num_points = pc.shape[1]
        if self.num_samples > num_points:
            raise ValueError(f"num_samples must not exceed the number of input points ({num_points}).")
        
        sample_indices = np.random.choice(num_points, self.num_samples, replace=False)
        pc_sampled = pc[:, sample_indices]
        return pc_sampled
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={self.num_samples})"
    

class Shuffle(nn.Module):
    """
    Shuffles the order of the points of the input point-cloud pc.
    """
    
    def __init__(self, seed):
        super().__init__()
        self.seed = seed
        if seed:
            self._generator = torch.Generator().manual_seed(self.seed)
        else:
            self._generator = None

        
    def forward(self, pc):
        """
        Parameters
        ----------
        pc : torch.Tensor
            Input point-cloud tensor of shape (in_features, num_points).

        Returns
        -------
        pc_shuffled : torch.Tensor
            Input point-cloud with the order of points shuffled,
            tensor of shape (in_features, num_points).
        """
        
        check_pc(pc, tensor_dims=2, xyz_only=False)
        
        num_points = pc.shape[1]
        indices = torch.randperm(num_points, generator=self._generator)
        pc_shuffled = pc[:, indices]
        
        return pc_shuffled


    def __repr__(self):
        return f"{self.__class__.__name__}(seed={self.seed})"
    

class RandomScale(nn.Module):
    """
    Randomly scales the input point-cloud pc by a scale uniformly sampled
    from the given range [scale_low, scale_high).
 
    Parameters
    ----------
    scale_low : float
        Lower boundary of the interval from which the scale is sampled.
        
    scale_high : float
        Upper boundary of the interval from which the scale is sampled.
    """
    
    def __init__(self, scale_low=0.8, scale_high=1.2):
        
        if not isinstance(scale_low, float) or scale_low < 0.0:
            raise ValueError("The scale_low should be a non-negative float.")
            
        if not isinstance(scale_high, float) or scale_high <= 0.0:
            raise ValueError("Thr scale_high should be a positive float.")
            
        if scale_low > scale_high:
            raise ValueError("It should be scale_low <= scale_high.")
              
        super().__init__()
        self.scale_low = scale_low
        self.scale_high = scale_high
        
    
    def forward(self, pc):
        """
        Parameters
        ----------
        pc : torch.Tensor
            Input point-cloud tensor of shape (in_features, num_points).

        Returns
        -------
        pc_scaled : torch.Tensor
            Randomly scaled point-cloud tensor
            of shape (in_features, num_points).
        """        
        
        check_pc(pc, tensor_dims=2, xyz_only=True)
        
        self._scale = np.random.uniform(low=self.scale_low, high=self.scale_high, size=None)
        
        pc_scaled = pc * self._scale
        
        return pc_scaled
        

    def __repr__(self):
        return f"{self.__class__.__name__}(scale_low={self.scale_low}, scale_high={self.scale_high})"


class RandomJitter(nn.Module):
    """
    Randomly jitters the input point-cloud pc, i.e. adds gausian noise to each
    point's coordinate independently.
    
    Each point of pc is jittered by a different noise.

    The noise is gaussian with zero mean and std given by the user. The noise
    can be clipped to the interval [-clip, clip].
    
    Parameters
    ----------
    std : float
        Standard deviation of the Gaussian noise.
    clip : float
        Value to clip (limit) the noise to the interval [-clip, clip].
    """
    
    def __init__(self, std=0.01, clip=0.05):
        
        if not isinstance(std, float) or std < 0.0:
            raise ValueError("std should be a non-negative float.")
            
        if not isinstance(clip, float) or clip < 0.0:
            raise ValueError("clip should be a non-negative float.")
            
        super().__init__()
        self.std = std
        self.clip = clip
        
            
    def forward(self, pc):
        """
        Parameters
        ----------
        pc : torch.Tensor
            Input point-cloud tensor of shape (in_features, num_points) with in_features = 3.

        Returns
        -------
        pc_jittered : torch.Tensor
            Randomly jittered point-cloud tensor of shape (in_features, num_points) with in_features = 3.
        """
        
        check_pc(pc, tensor_dims=2, xyz_only=True)
        
        noise = np.random.normal(loc=0.0, scale=self.std, size=pc.shape)
        noise = np.clip(noise, a_min=-self.clip, a_max=self.clip)
        noise = torch.tensor(noise, dtype=torch.float32)
        
        pc_jittered = pc + noise
        
        return pc_jittered
        

    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std}, clip={self.clip})"
    
    
class RandomTranslation(nn.Module):
    """
    Randomly translates the input point-cloud pc by a translation vector
    obtained by uniformly sampling its components from translation_range
    for each axis independently.
    
    Note that each point of pc is translated by the same translation vector.
    
    Parameters
    ----------
    translation_range : float or list
        Range of translations for each axis. If translation_range is a float,
        then the range will be [-translation_range, translation_range] for
        each axis.
    """
    
    def __init__(self, translation_range):
        
        if isinstance(translation_range, float):
            if translation_range < 0.0:
                raise ValueError("If the translation_range is a single float, it must be positive.")
            self.translation_range = [-translation_range, translation_range]
        elif isinstance(translation_range, list):
            if len(translation_range) != 2:
                raise ValueError("If the translation_range is a list, it must have length 2.")
            if translation_range[0] > translation_range[1]:
                self.translation_range = [translation_range[1], translation_range[0]]
            else:
                self.translation_range = translation_range
        else:
            raise TypeError("The translation_range can be a list of two floats or a single (positive) float.")
        
        super().__init__()
    
        
    @staticmethod
    def get_params(translation_range):
        """
        Returns a random translation vector.
        
        Returns
        -------
        translation_vec : torch.Tenosr
            Translatin vector.
        """
        
        translation_vec = np.random.uniform(low=translation_range[0], high=translation_range[1], size=3)
        
        return torch.tensor(translation_vec, dtype=torch.float32)
    
    
    def forward(self, pc):
        """
        Parameters
        ----------
        pc : torch.Tensor
            Input point-cloud tensor of shape (in_features, num_points) with in_features = 3.

        Returns
        -------
        pc_translated : torch.Tensor
            Randomly translated point-cloud tensor of shape (in_features, num_points) with in_features = 3.
        """
        
        # check_pc(pc, tensor_dims=2, xyz_only=True) -> this check is done in translate_pc
        
        self._translation_vec = self.get_params(self.translation_range)
        pc_translated = translate_pc(pc, self._translation_vec)
        
        return pc_translated


    def __repr__(self):
        return f"{self.__class__.__name__}(translation_range={self.translation_range})"
    

class RandomRotation(nn.Module):
    """
    Randomly rotates the input point-cloud pc by an angle uniformly sampled
    from angle_range around a given axis_vec (if not given then it's random selected).
    
    Note that each point of pc is rotated by the same rotation matrix.
    
    Parameters
    ----------
    angle_range : float or list
        Range of angles (in radians) to select from. If angle_range is a float
        then the range of angles will be [-angle_range, angle_range].
    axis_vec : None or np.ndarray or list
        Axis vector to rotate around. If axis is None then it will be selected randomly.
    """
    
    def __init__(self, angle_range, axis_vec=None):
        
        if isinstance(angle_range, float):
            if angle_range < 0.0:
                raise ValueError("If the angle_range is a single float, it must be positive.")
            self.angle_range = [-angle_range, angle_range]
        elif isinstance(angle_range, list):
            if len(angle_range) != 2:
                raise ValueError("If the angle_range is a list, it must have length 2.")
            if angle_range[0] > angle_range[1]:
                self.angle_range = [angle_range[1], angle_range[0]]
            else:
                self.angle_range = angle_range
        else:
            raise TypeError("The angle_range should be a list of two floats or a single (positive) float.")
        
        super().__init__()
        self.axis_vec = axis_vec
        
        
    @staticmethod
    def get_params(angle_range, axis_vec):
        """
        Returns parameters for a random rotation.
        
        Returns
        -------
        angle : float
            Angle of rotation in radians.
        axis : np.ndarray
            Axis vector to rotate around.
        """
        
        angle = np.random.uniform(low=angle_range[0], high=angle_range[1], size=None)
        
        if axis_vec is None:
            axis_vec = np.random.uniform(low=0, high=1, size=3)
            
        return angle, axis_vec
    
    
    def forward(self, pc):
        """
        Parameters
        ----------
        pc : torch.Tensor
            Input point-cloud tensor of shape (in_features, num_points) with in_features = 3.

        Returns
        -------
        pc_rotated : torch.Tensor
            Randomly rotated point-cloud tensor of shape (in_features, num_points) with in_features = 3.
        """
        
        # check_pc(pc, tensor_dims=2, xyz_only=True) -> this check is done in rotate_pc
        
        self._angle, self._axis_vec = self.get_params(self.angle_range, self.axis_vec)
        pc_rotated = rotate_pc(pc, self._angle, self._axis_vec)
        
        return pc_rotated
    

    def __repr__(self):
        return f"{self.__class__.__name__}(angle_range=[{self.angle_range[0]:.3f}, {self.angle_range[1]:.3f}], axis_vec={self.axis_vec})"
    