import torch
from torchvision import datasets, transforms, models
from torch import nn, optim


def load_data(data_dir):
    """
    Load the training, validation, and test datasets from the given directory.
    Apply the necessary transformations and return the dataloaders.
    """
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(root=f'{data_dir}/{x}',
                                transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
        }

    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x],
                                       batch_size=64, shuffle=True)
        for x in ['train', 'valid', 'test']
        }
    
    return dataloaders


def initialize_model(architecture='vgg16', hidden_units=1024,
                     output_size=41, dropout=0.5, learning_rate=0.003):
    """
    Initializes the model, criterion, and optimizer for training.
    
    Args:
        architecture (str): Name of the model architecture.
        hidden_units (int): Number of units in the hidden layer of the classifier.
        output_size (int): Number of output classes.
        dropout (float): Dropout rate in the classifier.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        model: The initialized model with the classifier replaced.
        criterion: The loss function.
        optimizer: The optimizer for training.
    """
    # model = getattr(models, architecture)(pretrained=True)
    if architecture == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Architecture '{architecture}' not recognized. "
                 f"Please use a supported architecture.")

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier
    input_size = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )
    
    # Define the loss function
    criterion = nn.NLLLoss()
    
    # Define the optimizer (only train the classifier parameters)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer


def train_model(model, dataloaders, criterion, optimizer, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 40
    
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(model, path, architecture, class_to_idx):
    checkpoint = {
        'arch': architecture,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'classifier': model.classifier
    }
    torch.save(checkpoint, path)