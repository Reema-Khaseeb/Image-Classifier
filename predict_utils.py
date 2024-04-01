import torch
from torchvision import models, transforms
from PIL import Image


def load_checkpoint(filepath):
    """load the checkpoint and rebuild the model

    Args:
        filepath (string)

    Returns:
        model
    """
    checkpoint = torch.load(filepath)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Model architecture {checkpoint['arch']} not recognized.")
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array.
    '''
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize such that the smallest side is 256 pixels
        transforms.CenterCrop(224),  # Crop the center 224x224 portion
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])

    # Open the image and apply the transformations
    image = Image.open(image_path)
    image_tensor = transform(image)
    
    return image_tensor


def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    model.to(device)
    model.eval()
    
    image = process_image(image_path)
    # Add batch dimension and send to device
    image = image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(image)
    
    # probs, classes = torch.exp(output).topk(topk)
    
    # return probs[0].tolist(), [model.class_to_idx[str(cls)] for cls in classes[0].tolist()]

    probs, indices = torch.exp(output).topk(topk)
    probs = probs.cpu().numpy()[0]  # Convert to array and squeeze
    indices = indices.cpu().numpy()[0]  # Convert to array and squeeze
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    class_names = [idx_to_class[index] for index in indices]
    return probs, class_names
