import argparse
from pathlib import Path


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--hdf5_path", type=str, default="",
                        help="path to a dataset .hdf5 file")
    
    parser.add_argument("--object_path", type=str, default="",
                        help="Path to an object .off file")
    
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to a checkpoint .pth file for resuming the training from it).")
    
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Path to a checkpoint .pth file for loading a model for evaluation or inference.")
    
    parser.add_argument("--id_to_class_path", type=str, default="",
                        help="Path to a id_to_class .pkl file")
    
    parser.add_argument("--checkpoint_save_dir", type=str, default="",
                        help="Dir for saving model checkpoints during training.")
    
    parser.add_argument("--log_dir", type=str, default="",
                        help="Dir for logging.")
    
    parser.add_argument("--plots_dir", type=str, default="",
                        help="dir for saving plots.")
    
    parser.add_argument("--start_epoch", type=int, default=1,
                        help="Start epoch number (useful for resuming training) [default: 1]")
    
    parser.add_argument("--num_epochs", type=int,
                        help="Number of epochs to train [default: None]")
    
    parser.add_argument("--batch_size_train", type=int, default=32,
                        help="Batch size for training [default: 32].")
    
    parser.add_argument("--batch_size_valid", type=int, default=32,
                        help="Batch size for validation [default: 32].")    

    parser.add_argument("--batch_size_test", type=int, default=32,
                        help="batch size for testing [default: 32].")
    
    parser.add_argument("--balance", type=int, default=0, choices=[0, 1],
                        help="Flag for weighted sampling for balancing the training dataset [default: 0]")
    
    parser.add_argument("--data_augment", type=int, default=0, choices=[0, 1],
                        help="Flag data augmentation [default: 0]")
    
    parser.add_argument("--lr_init", type=float, default=0.001,
                        help="Initial learning rate [default: 0.001]")
    
    parser.add_argument("--lr_step_size", type=int, default=10,
                        help="Period of learning rate decay [default: 10]")
    
    parser.add_argument("--lr_gamma", type=float, default=0.1,
                        help="Multiplicative factor of learning rate decay [default: 0.1]")
    
    parser.add_argument("--reg_weights", type=float, nargs="*", default=[0.0, 0.001],
                        help="Weights for the TNets regularization terms [default: [0.0, 0.001]]")
    
    parser.add_argument("--save_checkpoint_every", type=int, default=1,
                        help="Save checkpoint every given epoch [default: 1]")
    
    parser.add_argument("--print_every_batch", type=int, default=0,
                        help="Print training stats every given batch [default: 0]")
    
    parser.add_argument("--plot_confusion_mat", type=int, default=0, choices=[0, 1],
                        help="flag for plotting confusion matrix [default: 0]")
    
    parser.add_argument("--plot_losses", type=int, default=0, choices=[0, 1],
                        help="Flag for plotting losses w.r.t epochs [default: 0]")
    
    parser.add_argument("--device", type=str, default="", choices=["cpu", "cuda"],
                        help="Device to use (cpu or cuda).")
    
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of object classes [default: 10]")
    
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top k classes to predict [default: 5]")
    
    parser.add_argument("--seed", type=int, default=1235976,
                        help="random seed [default: 42]")

    args = parser.parse_args()
    
    
    # Check some arguments
    if args.device == "":
        raise ValueError("The device argument must be provided ('cpu' or 'cuda').")
    
    if args.hdf5_path and not Path(args.hdf5_path).is_file():
        raise FileNotFoundError(f"The hdf5_path = {args.hdf5_path} not found.")
    
    if args.object_path and not Path(args.object_path).is_file():
        raise FileNotFoundError(f"The object_path = {args.object_path} not found.")
        
    if args.checkpoint_path and not Path(args.checkpoint_path).is_file():
        raise FileNotFoundError(f"The checkpoint_path = {args.checkpoint_path} not found.")
        
    if args.id_to_class_path and not Path(args.id_to_class_path).is_file():
        raise FileNotFoundError(f"The id_to_class_path = {args.id_to_class_path} not found.")
        
    if args.resume_path and not Path(args.resume_path).is_file():
        raise FileNotFoundError(f"The resume_path = {args.resume_path} file not found.")
        
    if args.checkpoint_save_dir and not Path(args.checkpoint_save_dir).is_dir():
        Path(args.checkpoint_save_dir).mkdir()
        
    if args.log_dir and not Path(args.log_dir).is_dir():
        Path(args.log_dir).mkdir()
        
    if args.plots_dir and not Path(args.plots_dir).is_dir():
        Path(args.plots_dir).mkdir()
        

    # Convert int flags to boolean
    args.plot_confusion_mat = bool(args.plot_confusion_mat)
    args.data_augment = bool(args.data_augment)
    args.plot_losses = bool(args.plot_losses)
    args.balance = bool(args.balance)
        
    return args
    