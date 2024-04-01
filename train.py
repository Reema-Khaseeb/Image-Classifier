import argparse
import torch
import train_utils
import os

def main():
    # Setup command line arguments
    parser = argparse.ArgumentParser(description='Train a new network + \
                                    on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_dir', type=str, help='Directory of the dataset')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints',
                        default='checkpoints')
    parser.add_argument('--arch', type=str,
                        help='Architecture [vgg16, resnet18, ...]',
                        default='vgg16')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate', default=0.003)
    parser.add_argument('--hidden_units', type=int,
                        help='Hidden units', default=1024)
    parser.add_argument('--output_size', type=int,
                        help='Number of output classes', default=41)
    parser.add_argument('--dropout', type=float, help='Dropout rate', default=0.5)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5)
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    # Parse command line arguments
    args = parser.parse_args()

    # Check if save_dir exists, create if not
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Initialize the model, criterion, and optimizer with parsed arguments
    model, criterion, optimizer = train_utils.initialize_model(architecture=args.arch,
                                                               hidden_units=args.hidden_units,
                                                               output_size=args.output_size,
                                                               dropout=args.dropout,
                                                               learning_rate=args.learning_rate)

    # Setup device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load data
    dataloaders = train_utils.load_data(args.data_dir)

    # Train the model
    train_utils.train_model(model, dataloaders, criterion, optimizer,
                            args.epochs, device)

    # Save the model checkpoint
    train_utils.save_checkpoint(model, args.save_dir + '/checkpoint.pth',
                                args.arch,
                                dataloaders['train'].dataset.class_to_idx)


if __name__ == '__main__':
    main()