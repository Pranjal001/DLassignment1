# DLassignment1

# Neural Network

This repository has a trian.py file which trains a Deep learning Neural Network which trains over Mnist_fashion as well as Mnist datasets. This is made by using only Numpy.It accomodate various configurations of parameters which is used for classification.Parameters such as Activation functions,number of layers , layer size, loss functions and many more are used.

# Attributes:

  hidden_layers : number neurons per layers
  activation : Activation functions such as sigmoid , tanh , ReLu
  loss_type : cross entropy and MSE
  dataset : type of data sets(mnist and fashion mnist)
  input_size : size of input layer(784 considering mnist datasets)
  output_size : size of output layer
  epochs : Number of iterations for training
  batch_size : batch size for training
  initialiser : type of initialisation

# Class:

# NeuralNet :
  This layer is Created for generation of the Neural Network and assigning layer values such as weights and biases etc. This contains methods such as Feedforwardpropagate and Backwardpropagate.

# weightsAndBiasLayer:
  This class is used by the NeuralNet class to help create weights and biases corresponding to each layer. Also this layer contains all possibl eactivation funtions and thier differentiation functions which help in forward and backward propagation

# optimisation_funcitons:
  This Class takes neural nets object and other required variables as arguments and helps as the update rule in training the neural network.

# super_neural :
  This class is super class which contains methods such as train_neural which trains the neural network with customizable batch configs. Other helping methods such as Cross_entropy, Mean_squared_loss and accuracy are used to to evaluate the models performance.Also a plot_confusion_matrix method which plots the confusion matrix for a given argument.

# Training :
  Trian(train.py) the model by parsing arguments the arguments used are provided in the code and by default the values are set to best model performance parameters. 

# helper methods :
  data_selector : selects a dataset(mnist or fashion_mnist) which return train,test and validation tests
  one_hot_vector and reshape_1D are used for preprocessing the dataset train,test and validation set values.

  
