import time
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

ROOT_DIR = Path.cwd().parent.parent
MODULES = ["components", "models", "data"]
for module in MODULES:
    module_path = Path(ROOT_DIR, module)
    if module_path not in map(Path, sys.path):
        sys.path.append(str(module_path))

from pointnet_clf import PointNetClf
from modelnet import ModelNetDataset
from data_utils import get_timestamp
from arg_parser import parse_args


def plot_confusion_mat(confusion_mat, id_to_class, save_to=""):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(confusion_mat, interpolation="nearest", cmap=plt.cm.Blues)
    tick_marks = list(id_to_class.keys())
    classes = list(id_to_class.values())
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix")
    
    for (i, j), val in np.ndenumerate(confusion_mat):
        ax.text(j, i, confusion_mat[i, j].item(), horizontalalignment="center", color="black")
    
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)


def evaluate():
    
    t0 = time.time()
    
    # Parse command line arguments
    args = parse_args()
        
    # Build test dataloader
    test_dataset = ModelNetDataset(h5_path=args.hdf5_path, train=False, transform=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_test)

    NUM_CLASSES = test_dataset.num_classes
    NUM_VAL_BATCHES = len(test_dataloader)
    NUM_VAL_PCS = len(test_dataloader.dataset)

    # Build model
    pointnet_clf = PointNetClf(in_features=3, num_classes=NUM_CLASSES)
    
    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    pointnet_clf.load_state_dict(checkpoint["model_state_dict"])
    
    pointnet_clf.to(args.device)
    pointnet_clf.eval()
    
    loss = 0
    acc = 0
    confusion_mat = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    with torch.no_grad():
        for pcs, labels in test_dataloader:
            pcs, labels = pcs.to(args.device), labels.to(args.device)
            pred, _, _ = pointnet_clf(pcs)
            
            loss_batch = F.nll_loss(pred, labels)
            loss += loss_batch.item()

            pred_labels = pred.argmax(dim=1)
            correct = pred_labels.eq(labels).sum().item()
            acc += correct
            
            for i, j in zip(labels, pred_labels):
                confusion_mat[i.item(), j.item()] += 1
            
    loss /= NUM_VAL_BATCHES
    acc /= NUM_VAL_PCS
    print(f"Test loss: {loss:.3f}, accuracy: {acc:.3f} [ELT: {time.time() - t0 : .2f} sec]\n")
        
    print(confusion_mat)
    if args.plot_confusion_mat:
        plot_confusion_mat(confusion_mat, test_dataset.id_to_class, Path(args.plots_dir, f"confusion_matrix_{get_timestamp()}.png"))
    

if __name__ == "__main__":
    evaluate()