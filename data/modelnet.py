from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from data_utils import get_pc


def modelnet_to_hdf5(data_dir, output_path, compression=None, num_points=2048, normalize=True):
    """
    Converts the ModelNet10/40 train and test datasets into a single .hdf5 file.
    
    The output .hdf5 will have two groups with two datasets each:
        'train/data': tensor of shape (num_train_pcs, in_features, num_points), 
        'train/labels': tensor of shape (num_train_pcs),
        'test/data': tensor of shape (num_test_pcs, in_features, num_points),
        'test/labels': tensor of shape (num_test_pcs).
        
    Additionally, we save the class-to-id mapping as an attribute of the .hdf5
    
    Data can be downloaded from https://modelnet.cs.princeton.edu/ to the data_dir,
    structured as follows:
        data_dir/
        ├── class_0/
        │   ├── test/
        │   │   ├── pc_0_0.off
        │   │   ├── pc_0_1.off
        │   │   └── ...
        │   └── train/
        │       ├── pc_0_x.off
        │       ├── pc_0_y.off
        │       └── ...
        ├── class_1/
        │   ├── test/
        │   │   ├── pc_1_0.off
        │   │   ├── pc_1_1.off
        │   │   └── ...
        │   └── train/
        │       ├── pc_1_z.off
        │       ├── pc_1_w.off
        │       └── ...
        └── ...

    Each input point-cloud is uniformly sampled to num_points points
    and (optionally) centered and normalized into unit sphere.
    """
    
    data = dict()
    data["train"] = torch.tensor([])
    data["test"] = torch.tensor([])
    
    labels = dict()
    labels["train"] = []
    labels["test"] = []
    
    out = h5py.File(output_path, "w")
    idx = 0
    for subdir in Path(data_dir).iterdir():
        if subdir.is_dir():
            print(idx, subdir.name)
            out.attrs[subdir.name] = idx
            for mode in ["train", "test"]:
                for f in Path(data_dir, subdir.name, mode).glob("*.off"):
                    pc = get_pc(f, num_points=num_points, normalize=normalize).unsqueeze(0)  # (1, in_features, num_points)
                    data[mode] = torch.cat([data[mode], pc])
                    labels[mode].append(idx)
            idx += 1
    
    # Train group
    out.create_dataset("train/data", data=data["train"], compression=compression, dtype="float32")
    out.create_dataset("train/labels", data=torch.tensor(labels["train"]), compression=compression, dtype="uint8")
    
    # Test group
    out.create_dataset("test/data", data=data["test"], compression=compression, dtype="float32")
    out.create_dataset("test/labels", data=torch.tensor(labels["test"]), compression=compression, dtype="uint8")
    
    out.close()
    

class ModelNetDataset(Dataset):
    """
    ModelNet dataset from a .hdf5 file.
    """
    
    def __init__(self, h5_path, train, indices=None, transform=None):

        self.h5_path = h5_path
        self.train = train
        self._mode = "train" if train else "test"
        self.indices = indices
        self.transform = transform
        
        file = h5py.File(Path(h5_path), "r")
        self.data = self.hdf5_select(file[f"{self._mode}/data"], indices)  # (num_pcs, in_features, num_points)
        self.labels = self.hdf5_select(file[f"{self._mode}/labels"], indices)  # (num_pcs)
        self.id_to_class = {int(i): c for c, i in file.attrs.items()}
        file.close()
        
        self.num_classes = len(self.id_to_class)
        
    
    @staticmethod
    def hdf5_select(dset, indices):
        """
        Selects rows at the given indices from the input hdf5 dataset and returns
        them as single torch tensor.
        """
        
        if indices is None:
            return torch.tensor(dset[:])
        else:
            try:
                return torch.tensor(dset[indices])
            except TypeError:
                indices_unique_sorted, indices_rev = np.unique(indices, return_inverse=True)            
                dset_array = dset[indices_unique_sorted]
                return torch.tensor(dset_array[indices_rev])
            
    
    def get_weights(self):
        """
        Returns the weights (inverse of the number of occurrences) for each sample
        in the dataset.

        Returns
        -------
        weights : list of floats
            A list of weights for each sample in the dataset.
        """
        
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        self.label_counts = dict(zip(unique_labels, counts))
        weights = [1 / self.label_counts[lab.item()] for lab in self.labels]
        
        return weights
        
        
    def __getitem__(self, idx):
        
        pc = self.data[idx]
        label = self.labels[idx]
                
        if self.transform is not None:
            pc = self.transform(pc)
        
        return pc, label
        
        
    def __len__(self):
        return len(self.data)
    


if __name__ == "__main__":
    
    # modelnet_to_hdf5(data_dir="data/ModelNet10", output="data/modelnet10_2048_normalized.hdf5", num_points=2048, normalize=True)
      
    import transforms
    from data_utils import split_indices
    
    # Create datasets
    transform = transforms.Compose([transforms.RandomSample(num_samples=100),
                                    transforms.RandomRotation(angle_range=np.pi/2, axis_vec=[0, 0, 1]),
                                    transforms.RandomTranslation(translation_range=1.0),
                                    transforms.RandomJitter(std=0.01, clip=0.05),
                                    transforms.RandomScale(scale_low=0.8, scale_high=1.2),
                                    transforms.Shuffle(seed=3214124)])
       
    train_indices, valid_indices = split_indices(split_lengths=[3000, 991], shuffle=True, seed=42)

    train_dataset = ModelNetDataset(h5_path="datasets//modelnet10_2048_normalized.hdf5",
                                    train=True,
                                    indices=train_indices,
                                    transform=transform)
    
    valid_dataset = ModelNetDataset(h5_path="datasets//modelnet10_2048_normalized.hdf5",
                                    train=True,
                                    indices=valid_indices,
                                    transform=None)
    
    test_dataset = ModelNetDataset(h5_path="datasets//modelnet10_2048_normalized.hdf5",
                                   train=False,
                                   transform=None)
    
    print(f"Sizes of train dataset: {len(train_dataset)}, val dataset: {len(valid_dataset)}, test dataset: {len(test_dataset)}")

    
    # Save the id_to_class mapping as a pickle file.
    id_to_class = train_dataset.id_to_class
    # import pickle
    # with open("data/modelnet10_2048_id_to_class.pkl", 'wb') as f:
    #     pickle.dump(id_to_class, f)
    
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=32,
                                  sampler=WeightedRandomSampler(train_dataset.get_weights(), len(train_dataset)))
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    
    test_dataloader = DataLoader(test_dataset, batch_size=32)
    
    
    # Check the class balance of the train dataset
    labels_all = []
    for pcs, labels in train_dataloader:
        labels_all.extend(labels.tolist())
    unique_labs, counts = np.unique(labels_all, return_counts=True)
    for lab, c in zip(unique_labs, counts):
        print(f"{id_to_class[lab]}: {c}")
    
    # Plot a point cloud
    from matplotlib import pyplot as plt
    train_iter = iter(train_dataloader)
    pcs, labels = next(train_iter)
    i = np.random.choice(pcs.shape[0])
    pc = pcs[i]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc[0], pc[1], pc[2])
    ax.set_title(id_to_class[labels[i].item()])
    ax.set_axis_off()
    plt.show()