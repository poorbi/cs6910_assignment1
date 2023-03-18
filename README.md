# cs6910_assignment1
# q1.py
This file was used to plot one sample image from each class
# train.py file
This is the main file(python script) that is asked to upload. It allows one to enter commandline arguments as specified in the assignment description.

When run without the commandline arguments the file runs for default parameters that have been set to the best values of hyperparameters found from tuning.

When run with the commandline arguments the file runs for values that have been specified.

The following codes have been commented as they have been run once and their plots have been put on report. These need to be uncommented to run specific features if you feel like testing.

1. Sweep : line 1097 of wandb agent has been commented out, if you want to run sweeps then uncomment this line.
2. Confusion Matrix Plot : lines 1106 to 1120 need to be uncommented out to plot confusion matrix.
3. Plots of Cross Entropy vs. Mean Squared Error : lines 1124 to 1135 need to be uncommented.
4. 3 Configurations for MNIST DATASET : You need to set the dataset value to 'mnist' through commandline args and then uncomment out lines 1139 to 1155.

# sweep.yaml file
This file contains method used for sweep and choices for the various hyperparameters needed for tuning. 

# Best Parameters
epochs = 10
batch size = 32
loss function = 'cross_entropy'
optimizer = 'nadam'
learning rate = 1e-3
weight decay constant = 0
weight initialization = 'Xavier'
number of hidden layers = 3
hiddel layer size = 128
activation function = 'ReLU'

# Best Results Obtained
Train accuracy = 91.43%
Validation accuracy = 89.08%
Test accuracy = 88.01%
