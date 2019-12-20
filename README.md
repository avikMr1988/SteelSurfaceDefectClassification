# Steel Surface Defect Classification

Implements a simple CNN Module for working on Steel Strip Surface Defect Classification Database.
            
            Test Accuracy on Augmented Data Set Achieved is : 0.923.
            Accuracy Metric Used is : Sklearn.metrics.accuracy_score
          
Uses a three layered convolutional network Each layer using maxpooling for downsampling of Images. Three fully connected layers are being used to get the desired prediction accuracy.


Specifications of Network :

              PyTorch Sequential Layer : Conv2D(Kernel Size : 5, Padding : 2, Stride : 1, Activation : Tanh, Nr. Channels: 32),               Maxpooling : (Kernel: 2, Stride  : 2)
              PyTorch Sequential Layer : Conv2D(Kernel Size : 5, Padding : 2, Stride : 1, Activation : Tanh, Nr. Channels: 64),               Maxpooling : (Kernel: 2, Stride  : 2)
              PyTorch Sequential Layer : Conv2D(Kernel Size : 5, Padding : 2, Stride : 1, Activation : Tanh, Nr. Channels:                    128),Maxpooling : (Kernel: 2, Stride  : 2)
              Fully Connected Layers : 4*4*128 - > 1000, 1000 - > 500, 500 -> 9
              
 Hyper Parameters used for training :
               
               Nr of Epochs : 5
               Learning rate : 0.001
               Momentum : 0.9
               Batch Size : 50
               Optimizer : Adam
               Loss : CrossEntropy

               
Software Requirements:

                                    Python : 3.6.9
                                    Torch  : 1.3.1
                                    PIL    : 6.2.1
                                    GDown  : 3.8.3 
Code Usage :

Run MainScript.Py : It has provisions for Downloading the data, Creating the DataSet, Firing the training and the evaluation on the stored Data.
