import logging
import time
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

ROOT_DIR = Path.cwd().parent.parent
MODULES = ["components", "models", "data"]
for module in MODULES:
    module_path = Path(ROOT_DIR, module)
    if module_path not in map(Path, sys.path):
        sys.path.append(str(module_path))
        
from pointnet_clf import PointNetClf
from modelnet import ModelNetDataset
from data_utils import split_indices, get_timestamp
import transforms
from arg_parser import parse_args


def create_logger(log_path):
    
    logger = logging.getLogger("train_clf_logger")
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO) 
    file_formatter = logging.Formatter(fmt='[%(asctime)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO) 
    stream_formatter = logging.Formatter('%(message)s')
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    return logger


def plot_losses(train_losses, val_losses, save_to=""):

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(train_losses, label='train loss')
    ax.plot(val_losses, label='val loss')
    ax.set_xlim(0, len(train_losses))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    plt.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)


def loss_clf(pred, labels, transform_mats=None, reg_weights=None):
    """
    Loss for the object classification: nll loss + regularization terms for transformation matrices.
    
    pred: log-probabilities, shape (batch_size, num_classes)
    labels: ints, shape (batch_size)
    transform_mats: list of transformation matrices from tnets, of shape (batch_size, K, K)
    reg_weights: list of regularization weights for each transformation matrix
    """  
    
    
    compute_reg = True
    if transform_mats is None and reg_weights is None:
        compute_reg = False
    elif isinstance(transform_mats, torch.Tensor) and isinstance(reg_weights, float):
        transform_mats = [transform_mats]
        reg_weights = [reg_weights]
    elif isinstance(transform_mats, list) and isinstance(reg_weights, list):
        if len(transform_mats) != len(reg_weights):
            raise ValueError("transform_mats and reg_weights should have the same length")
        else:
            if not all(isinstance(t, torch.Tensor) for t in transform_mats) or not all(isinstance(w, float) for w in reg_weights):
                raise TypeError("transform_mats must contain torch.Tensors whereas reg_weights  must contain floats.")
    else:
        raise TypeError(
            "transform_mats and reg_weights can be None and None, or torch.Tensor and float, "
            "or a list of torch.Tensor and a list of floats, respectively")
        
    loss = F.nll_loss(pred, labels)
        
    if compute_reg:
        for i in range(len(transform_mats)):
            w_i = reg_weights[i]
            if w_i > 0.0:
                t_i = transform_mats[i]
                I = torch.eye(t_i.shape[1], device=t_i.device).unsqueeze(dim=0)  #.expand(t_i.shape)
                diff = torch.bmm(t_i, t_i.transpose(2,1)) - I
                reg_i = torch.mean(torch.sum(diff**2, dim=(1,2)))
                loss += w_i * reg_i
        
    return loss


def train_one_epoch(args, logger, epoch, train_dataloader, model, optimizer, scheduler):
    
    t0 = time.time()
        
    model.train()
    
    num_train_batches = len(train_dataloader)
    num_train_pcs = len(train_dataloader.dataset)
    
    loss_epoch = 0
    acc_epoch = 0
    for batch_idx, (pcs, labels) in enumerate(train_dataloader, 1):
        pcs, labels = pcs.to(args.device), labels.to(args.device)
        
        # Zero the gradients for every batch
        optimizer.zero_grad()
        
        # Make predictions for this batch. Note that pred is log-probabilities
        pred, trans_0, trans_1 = model(pcs)
        
        # Compute the (batch) loss and its gradients
        loss_batch = loss_clf(pred, labels, [trans_0, trans_1], args.reg_weights)
        loss_batch.backward()
        
        # Update weights
        optimizer.step()
        
        # Computer the epoch's mean loss and accuracy
        loss_epoch += loss_batch.item()
        
        pred_labels = pred.argmax(dim=1)  # We don't need to convert log-probabilities pred to probabilities because log is strictly increasing and we are computing max()
        correct = pred_labels.eq(labels).sum().item()
        acc_epoch += correct
        
        # Print batch loss and accuracy
        if args.print_every_batch > 0 and batch_idx % args.print_every_batch == 0:
            print(f"[Batch: {batch_idx}/{num_train_batches}] train loss: {loss_batch:.3f}, accuracy: {correct / pcs.shape[0]:.3f}")
        
    loss_epoch = loss_epoch / num_train_batches
    acc_epoch = acc_epoch / num_train_pcs
    logger.info(f"Epoch {epoch} train loss: {loss_epoch:.3f}, accuracy: {acc_epoch:.3f} [ELT: {time.time() - t0 : .2f} sec]")
    
    # Update learning rate
    scheduler.step()
    
    # Save checkpoint
    if args.save_checkpoint_every and epoch % args.save_checkpoint_every == 0:
        checkpoint_path = Path(args.checkpoint_save_dir, f"checkpoint_clf_ep{epoch}_dt{get_timestamp()}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}, checkpoint_path)
        logger.info(f"Saved checkpoint {checkpoint_path}.")
        
    return loss_epoch, acc_epoch


def valid_one_epoch(args, logger, epoch, valid_dataloader, model):
    
    t0 = time.time()
    
    model.eval()
    
    num_val_batches = len(valid_dataloader)
    num_val_pcs = len(valid_dataloader.dataset)
    
    loss_epoch = 0
    acc_epoch = 0
    with torch.no_grad():
        for pcs, labels in valid_dataloader:
            pcs, labels = pcs.to(args.device), labels.to(args.device)
            pred, _, _ = model(pcs)
            
            loss_batch = F.nll_loss(pred, labels)
            loss_epoch += loss_batch.item()

            pred_labels = pred.argmax(dim=1)
            correct = pred_labels.eq(labels).sum().item()
            acc_epoch += correct
    
    loss_epoch = loss_epoch / num_val_batches
    acc_epoch = acc_epoch / num_val_pcs
    logger.info(f"Epoch {epoch} valid loss: {loss_epoch:.3f}, accuracy: {acc_epoch:.3f} [ELT: {time.time() - t0 : .2f} sec]")
    
    return loss_epoch, acc_epoch


def train():
    
    t0 = time.time()
    
    # Parse command line arguments
    args = parse_args()
    print(f"Training on: {args.device}")
    
    # Create logger
    logger = create_logger(Path(args.log_dir, "training_clf.log"))
    logger.info("TRAINING...")
    logger.info(f"ARGS: {vars(args)}")

    # Build train and validation dataloaders
    if args.data_augment:
        transform = transforms.Compose([transforms.RandomRotation(angle_range=np.pi/12, axis_vec=[0, 0, 1]),
                                        transforms.RandomTranslation(translation_range=0.1),
                                        transforms.RandomJitter(std=0.01, clip=0.05),
                                        transforms.RandomScale(scale_low=0.8, scale_high=1.2),
                                        transforms.Shuffle(seed=args.seed)])
    else:
        transform = None
    
    logger.info(f"Transforms: {transform}")
    
    train_indices, valid_indices = split_indices(split_lengths=[3000, 991], shuffle=True, seed=args.seed)
    
    train_dataset = ModelNetDataset(h5_path=args.hdf5_path,
                                    train=True,
                                    indices=train_indices,
                                    transform=transform)
    
    valid_dataset = ModelNetDataset(h5_path=args.hdf5_path,
                                    train=True,
                                    indices=valid_indices,
                                    transform=None)    

    logger.info(f"Created train and valid datasets from {args.hdf5_path}.")
    
    if args.balance:
        sampler = WeightedRandomSampler(train_dataset.get_weights(), len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size_train, shuffle=True)
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_valid)

    NUM_CLASSES = train_dataset.num_classes

    # Build model
    pointnet_clf = PointNetClf(in_features=3, num_classes=NUM_CLASSES)
    pointnet_clf.to(args.device)
    
    # Build optimizer
    optimizer = torch.optim.Adam(pointnet_clf.parameters(), lr=args.lr_init, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # Resume training (optional)
    start_epoch = args.start_epoch
    if args.resume_path:
        checkpoint = torch.load(Path(args.resume_path))
        logger.info(f"Resuming training... Loaded checkpoint {args.hdf5_path}.")
        start_epoch = checkpoint["epoch"] + 1
        pointnet_clf.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Start training
    train_losses = []
    valid_losses = []
    end_epoch = start_epoch + args.num_epochs
    for epoch in range(start_epoch, end_epoch):
        logger.info(f"\n------------ Epoch {epoch} / {end_epoch - 1} ------------")
        
        # Train and validate epoch
        train_loss_epoch, train_acc_epoch = train_one_epoch(args, logger, epoch, train_dataloader, pointnet_clf, optimizer, scheduler)
        valid_loss_epoch, valid_acc_epoch = valid_one_epoch(args, logger, epoch, valid_dataloader, pointnet_clf)
        
        train_losses.append(train_loss_epoch)
        valid_losses.append(valid_loss_epoch)
    
    if args.plot_losses:
        plot_losses(train_losses, valid_losses, Path(args.plots_dir, f"losses_plot_{get_timestamp()}.png"))
        
    logger.info(f"\nDONE TRAINING. Elapsed time: {time.time() - t0 : .2f} sec]")


         
if __name__ == "__main__":
    train()