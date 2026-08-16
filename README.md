# PyTorch Animal Image Classifier

A convolutional neural network built with PyTorch to classify images as birds, cats, dogs, or horses. The project filters CIFAR-10 to four animal classes, trains a custom CNN, evaluates it on held-out images, saves the trained weights, and performs confidence-scored inference on a supplied image.

## Features

- Custom four-block convolutional neural network
- CIFAR-10 download and four-class filtering
- Image resizing, normalization, and random horizontal flipping
- Mini-batch training with Adam and cross-entropy loss
- Test accuracy after every epoch
- Per-class precision, recall, F1, and confusion matrix
- CPU or CUDA device selection
- Model checkpoint saving and loading
- Confidence-scored inference on a local image
- Reproducible random seeding

## Technology

- Python
- PyTorch
- Torchvision
- scikit-learn
- Pillow

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Run the project:

```bash
python project3.py
```

On the first run, Torchvision downloads CIFAR-10 and the program trains the model for eight epochs. The trained state dictionary is saved as `animal_cnn.pth`. Later runs load that checkpoint before evaluation and inference.

The example inference image must be named `test_animal_image.jpg` and located beside the Python file. Replace it with another image using the same filename to test a different input.

## Model architecture

The network contains convolutional blocks with batch normalization, ReLU activations, and max pooling. Adaptive average pooling reduces the final feature map, and a dropout-regularized linear layer produces logits for the four classes.

## Dataset

The application uses the bird, cat, dog, and horse categories from CIFAR-10. Training images receive random horizontal flips; test and inference images use deterministic resizing and normalization.

## Evaluation

The program reports training accuracy and held-out test accuracy during training. After training or checkpoint loading, it prints a classification report and confusion matrix. Record the final metrics from a clean run in this README before presenting the project as a benchmarked model.

## Repository notes

The downloaded dataset and trained checkpoint are excluded from Git by default. If you decide to distribute a checkpoint, use an appropriate artifact store or Git LFS and document the exact training configuration.
