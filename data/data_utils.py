import time
import trimesh
import torch


def check_pc(pc, tensor_dims, xyz_only=False):
    """
    Checks if the given point-cloud pc is a 2D tensor of shape (in_features, num_points) if tensor_dims = 2,
    or a 3D tensor of shape (batch_size, in_features, num_points) if tensor_dims = 3.
    
    If xyz_only = True then it also checks if point features are only the point coordinates x, y, z,
    i.e. if in_features == 3.
    
    Parameters
    ----------
    pc : torc.Tensor
    tensor_dims : int from {1, 2}
    xyz_only : boolean, optional

    Returns
    -------
    None.
    """
    
    if not isinstance(tensor_dims, int) or tensor_dims not in {2, 3}:
        raise ValueError("The tensor_dims should be an int from {2, 3}.")
        
    if not isinstance(pc, torch.Tensor) or pc.dim() != tensor_dims:
        raise ValueError("The input should be a 2D or 3D torch tensor of shape (in_features, num_points) "
                         "or (batch_size, in_features, num_points) respectively.")
       
    if xyz_only:
        if pc.dim() == 2 and pc.shape[0] != 3:
            raise ValueError("The shape of the input 2D point-cloud should be (3, num_points).")
        elif pc.dim() == 3 and pc.shape[1] != 3:
            raise ValueError("The shape of the input 3D point-cloud should be (batch_size, 3, num_points).")


def get_timestamp():
    """
    Returns the local time as a str of the form 'month.day.year.hour.minute.second'.
    """
    lt = time.localtime()
    timestamp = str(lt.tm_mday).zfill(2) + "."
    timestamp += str(lt.tm_mon).zfill(2) + "."
    timestamp += str(lt.tm_year) + "."
    timestamp += str(lt.tm_hour).zfill(2) + "."
    timestamp += str(lt.tm_min).zfill(2) + "."
    timestamp += str(lt.tm_sec).zfill(2)
    
    return timestamp


def get_pc(object_path, num_points):
    """
    Returns a point-cloud as a torch.Tensor from a .off file.
    
    Parameters
    ----------
    path : str
        Path to a .off file containing the point-cloud.
        
    num_points : int
        Number of points to sample from the point-cloud.
    
    Returns
    -------
    pc : torch.Tensor
        The output point-cloud tensor of shape (in_features, num_points)
    """
    
    pc = trimesh.load(object_path).sample(num_points)  # (num_points, in_features)
    pc = torch.tensor(pc, dtype=torch.float32).transpose(1, 0)  # (in_features, num_points)
    
    return pc


def split_indices(split_lengths, shuffle=False, seed=None):
    """
    Creates indices [0, 1, ..., sum(split_lengths) - 1]
    and splits them into chunks of lengths specified by split_lengths.
    
    Before splitting, the indices can be optionally shuffled.
    
    Usefull for producing indices for splitting datasets into train,
    validation and test datasets.
    
    Example
    -------
    >>> split_indices([3, 2, 1, 2, 2])
    [[0, 1, 2], [3, 4], [5], [6, 7], [8, 9]]
    
    >>> split_indices([3, 2, 1, 2, 2], shuffle=True)
    [[8, 6, 7], [9, 2], [3], [1, 5], [4, 0]]
    
    Parameters
    ----------
    split_lengths : list
        Lengths of the splits to be produced.
        
    shuffle : boolean, optional
        Flag for shuffling the indices before splitting them. The default is False.
        
    seed : None or int, optional
        Random seed for reproducible results. The default is None.

    Returns
    -------
    indices_chunked : list of lists
        List of chunked indices.
    """
    
    if shuffle:
        generator = torch.Generator().manual_seed(seed) if seed else None
        indices = torch.randperm(sum(split_lengths), generator=generator)
    else:
        indices = torch.arange(sum(split_lengths))
    
    indices_chunked = [inds_chunk.tolist() for inds_chunk in torch.split(indices, split_lengths)]
    
    return indices_chunked