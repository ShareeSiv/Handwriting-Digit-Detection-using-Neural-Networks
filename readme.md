## Handwriting digit detection Neural Networks

I tested 3 CNN (Convolution Neural Network) and 1 NN (Neural Network) on the MNIST database. I ran each neural network for 10 epochs.

The performances for each neural network:

- CNN - Average loss:  0.0004, Accuracy 9901/10000  (99%)
- Simpler CNN - Average loss:  0.0003, Accuracy 9912/10000  (99%)
- Even simpler CNN - Average loss:  0.0003, Accuracy 9919/10000  (99%)
- Simple NN - Average loss:  0.0009, Accuracy 9718/10000  (97%)

From this I saw that the CNN and the Simple CNN I implemented were overkill for this task and even the simple NN was sufficient. This is could due the 'ease' (as it is a 32x32 greyscale image) of the task or how large the dataset.