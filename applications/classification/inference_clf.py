from pathlib import Path
import sys
import pickle
import torch

ROOT_DIR = Path.cwd().parent.parent
MODULES = ["components", "models", "data"]
for module in MODULES:
    module_path = Path(ROOT_DIR, module)
    if module_path not in map(Path, sys.path):
        sys.path.append(str(module_path))

from data_utils import get_pc
from pointnet_clf import PointNetClf
from transforms import normalize_pc
from arg_parser import parse_args



def inference():
    
    # Parse command line arguments
    args = parse_args()
    
    # Load the id_to_class map
    with open(args.id_to_class_path, "rb") as f:
        id_to_class = pickle.load(f)
    
    # Load a point cloud
    pc = get_pc(args.object_path, num_points=2048)  # ( 3, 2048)
    # Should normalize (as was done during training)
    pc = normalize_pc(pc).unsqueeze(0)  # (1, 3, 2048)
    pc = pc.to(args.device)
    
    # Build model
    pointnet_clf = PointNetClf(in_features=3, num_classes=args.num_classes)
    checkpoint = torch.load(Path(args.checkpoint_path), map_location=args.device)
    pointnet_clf.load_state_dict(checkpoint["model_state_dict"])
    pointnet_clf.to(args.device)
    pointnet_clf.eval()
    
    # Make predictions (top k)
    pred, _, _ = pointnet_clf(pc)
    probs = torch.exp(pred).squeeze(dim=0)
    top_probs, top_ids = torch.topk(probs, args.top_k)
    top_classes = [id_to_class[i.item()] for i in top_ids]

    for c, p in zip(top_classes, top_probs):
        print(f"{c} : {p:.3f}")

    return top_probs, top_classes
    
    

if __name__ == "__main__":
    inference()