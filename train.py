import argparse
import train_utils  # utility file for functions used in training

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
parser.add_argument('data_dir', type=str, help='Directory of the dataset')
parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints', default='checkpoints')
parser.add_argument('--arch', type=str, help='Architecture [vgg16, resnet18, ...]', default='vgg16')
parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)
parser.add_argument('--hidden_units', type=int, help='Hidden units', default=512)
parser.add_argument('--epochs', type=int, help='Number of epochs', default=20)
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args = parser.parse_args()

# Train the model with the provided arguments
train_utils.train_model(args)
