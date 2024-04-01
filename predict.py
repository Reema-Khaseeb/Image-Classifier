import argparse
import torch
from utils.predict_utils import load_checkpoint, predict


# Parse command line arguments
parser = argparse.ArgumentParser(description='Predict flower name from an image with a trained network.')
parser.add_argument('input', type=str, help='Path to the image')
parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
parser.add_argument('--top_k', type=int, help='Return top KK most likely classes', default=3)
parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping categories to real names')
parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

args = parser.parse_args()

# Predict the class for the image using the provided checkpoint
predict_utils.predict_class(args)
