import os

from copy import deepcopy
from collections import OrderedDict

from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.autograd import Variable

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models

# Had to rename file from workspace-utils to workspace_utils
from workspace_utils import active_session


criterion = nn.CrossEntropyLoss()


def get_device(use_gpu):
    if use_gpu and not torch.cuda.is_available():
        print("GPU isn't available, falling back to CPU")
    return torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")


def get_dataloaders(data_dir):
    train_dir = f'{data_dir.rstrip("/")}/train'
    valid_dir = f'{data_dir.rstrip("/")}/valid'
    test_dir = f'{data_dir.rstrip("/")}/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
        'valid': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]),
    }

    data_dirs = {
        'test': test_dir,
        'train': train_dir,
        'valid': valid_dir

    }

    datasets = ['test', 'train', 'valid']

    # TODO: Load the datasets with ImageFolder
    image_datasets = {d: ImageFolder(data_dirs[d], data_transforms[d]) for d in datasets}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {d: DataLoader(image_datasets[d], batch_size=64, shuffle=True, num_workers=0) for d in datasets}
    return dataloaders, image_datasets['train'].class_to_idx


def get_model(name):
    inputs = None
    model = None
    _hidden_layers = []
    if name == 'vgg16':
        model = models.vgg16(pretrained=True)
        inputs = model.classifier[0].in_features
    if name == 'densenet121':
        model = models.densenet121(pretrained=True)
        inputs = model.classifier.in_features
    if model == None:
        print(f"Model {name} isn't supported")
        return None, None
    model.model_name = name
    return model, inputs


def get_classifier(inputs, outputs, hidden_layers):
    layers = OrderedDict()
    all_layers = [inputs] + hidden_layers
    for idx in range(len(all_layers) - 1):
        i, o = all_layers[idx], all_layers[idx + 1]
        layers.update({f'fc{idx}': nn.Linear(i, o)})
        layers.update({f'relu{idx}': nn.ReLU()})
        layers.update({f'drop{idx}': nn.Dropout(0.5)})
    
    layers.update({f'fc{len(all_layers) - 1}': nn.Linear(hidden_layers[-1], outputs)})
    layers.update({f'doutput': nn.LogSoftmax(dim=1)})
    return nn.Sequential(layers)


def get_model_and_optimizer(_model_name, _lr, _num_outputs, device, _hidden_layers=None):
    _model, _num_inputs = get_model(_model_name)
    if not _model:
        return None, None, None
    
    _classifier = get_classifier(_num_inputs, _num_outputs, _hidden_layers)
    # Freeze params
    for param in _model.parameters():
        param.requires_grad = False

    # update classifer of model
    _model.classifier = _classifier
    params = [p for p in _model.parameters() if p.requires_grad]

    _optimizer = optim.Adam(params, lr=_lr)
    _model.to(device)
    return _model, _optimizer, _hidden_layers

def validate(_model, _loader, device):
    _loss = 0
    _accuracy = 0
    _model.eval()
    _loader_len = len(_loader)
    with torch.no_grad():
        for inputs, labels in _loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = _model.forward(inputs)
            batch_loss = criterion(logps, labels)

            _loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            _accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    _model.train()
    return (_loss / _loader_len), (_accuracy / _loader_len) * 100


def train_model(_model, _optimizer, _dataloaders, device,
                epochs=1, print_every=20, target_accuracy=70):
    _trainloader = _dataloaders['train']
    _validloader = _dataloaders['valid']
    steps = 0
    running_loss = 0
    best_model = (0, {})
    print("Training model...")
    for epoch in range(epochs):
        for inputs, labels in _trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            _optimizer.zero_grad()

            logps = _model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            _optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                _valid_loss, _valid_accuracy = validate(_model, _validloader, device)
                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Valid loss: {_valid_loss:.3f}.. "
                    f"Valid accuracy: {_valid_accuracy:.2f}%"
                )
                acc, _ = best_model
                # save model with best accuracy to avoid overfitting...
                if _valid_accuracy > acc:
                    best_model = (_valid_accuracy, deepcopy(_model.state_dict()))
                
                running_loss = 0
    _, state_dict = best_model
    _model.load_state_dict(state_dict)
    print("Finished training model.")
    

def save_model(_model, _lr, _epochs, _optimizer, _num_outputs, _hidden_layers, save_dir):
    checkpoint_path = f'{save_dir.rstrip("/")}/{_model.model_name}_checkpoint.pth'
    state = {
        'class_to_idx' : _model.class_to_idx,
        'epochs': _epochs,
        'hidden_layers': _hidden_layers,
        'learning_rate': _lr,
        'model_name': _model.model_name,
        'model_state': _model.state_dict(),
        'optimizer_state' : _optimizer.state_dict(),
        'num_outputs': _num_outputs
    }
    torch.save(state, checkpoint_path)
    print(f"Model saved to file: {checkpoint_path}")

    
def load_model(checkpoint_path, device):
    print("Loading model..")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file {checkpoint_path} doesn't exist.")
        return None, None, None
    # Issue when you save model with gpu and try to load on cpu
    # https://stackoverflow.com/questions/55759311/runtimeerror-cuda-runtime-error-35-cuda-driver-version-is-insufficient-for
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    state = torch.load(checkpoint_path, map_location=map_location)
    _num_outputs = state['num_outputs']
    _model_name = state['model_name']
    _hidden_layers = state['hidden_layers']
    _lr = state['learning_rate']
    
    _model, _optimizer, _ = get_model_and_optimizer(_model_name, _lr, _num_outputs, device, _hidden_layers)
    

    _model.load_state_dict(state['model_state'])
    _model.class_to_idx = state['class_to_idx']
    _optimizer.load_state_dict(state['optimizer_state'])
    print("Model loaded.")
    return _model, _optimizer, state['epochs']


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = image.resize((256, 256))
    crop_size = (256 - 224) / 2
    end = 256 - crop_size
    image = image.crop((crop_size, crop_size, end, end))
    image = np.array(image) / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return np.transpose(image, (2, 0, 1))

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file\
    img = Image.open(image_path)
    img = process_image(img)
    img = Variable(torch.from_numpy(img).float()).to(device)
    img = img.unsqueeze(0)
    # set model to evaluation mode
    model.eval()
    op = model.forward(img).cpu()
    topk = torch.exp(op).data.topk(topk)
    probs = np.array(topk[0]).flatten()
    indexes = np.array(topk[1]).flatten()
    idx_to_class = {i: c for c, i in model.class_to_idx.items()}
    return probs, [idx_to_class[i] for i in indexes]