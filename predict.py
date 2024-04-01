import argparse
import torch
from predict_utils import load_checkpoint, predict

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image with a trained network')
    parser.add_argument('input_image', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, help='Return top K most likely classes', default=5)
    parser.add_argument('--category_names', type=str, help='JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.input_image, model, args.top_k, device)

    if args.category_names:
        import json
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[cls] for cls in classes]

    print('Predicted Classes:', classes)
    print('Class Probabilities:', probs)

if __name__ == '__main__':
    main()
