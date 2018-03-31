

Kaggle Project : Image classification to identify largest digit in image.
Section number 551-002
Team Name:themeanteam







Team Members


Khalil Mohsin
khalil.mohsin@mail.mcgill.ca
260631318
Brian Hyung
brian.hyung@mail.mcgill.ca
260622837
Jongwoo Shim
Jongwoo.shim@mail.mcgill.ca
260670012






Introduction
Over the last decade or so the incredible breakthroughs in machine learning have given rise to numerous uses of its algorithms, that were previously either not possible or too unreliable. One of the most commonly proposed problem that the machine learning algorithms have been able to solve is the idea of recognizing images by extracting information out of it and being able to classify them to a certain label. For example, given a set of labels Y1,Y2 . . . . Yi Y labelling different types of animals, can we make a machine learning model that could take input images X1 X2 . . . . Xjand give and output Y1,Y2. . . Yj Ysuch that YY. 
In our defined problem we are given a set of grayscale images with different handwritten numbers and we want to make a model that could pick the biggest digit and be able to classify it accurately. The input x is hence a 3d matrix with 50,000, 64 x 64 pixels, and Y is the set of digits between 0 to 9. We approached this problem by first extracting the largest digit within each image as part of the preprocessing pipeline. Another important stage of preprocessing was to reduce the images to 28 x 28 pixels in order reduce the chances of overfitting and improve runtimes of our models. Additional stages involved data augmentation to increase the training data set. Our preprocessing pipeline produced a 3d matrix of ~80000 examples, each with dimensions of  28 x 28 pixels, which we used to train three different classifiers: a linear support vector machine (SVM), a dense feed-forward neural network, and a convolutional neural network (CNN). 
The levels of accuracy performance varied across our models. The linear SVM, after hyperparameter tuning, achieved 75.42% on the validation dataset and 67% on the test dataset. The dense feed-forward network, with five layers with 28 nodes each achieved 98% accuracy on the validation dataset. The CNN performed the best, with a validation dataset accuracy of 99.6% and 94.57% on the test dataset. 



Feature Design
Image preprocessing was divided into four main steps: feature reduction, largest digit isolation, standardization, and augmentations. The skimage library was used for most aspects of the preprocessing pipeline.

Feature reduction. An initial observation of the training dataset revealed that the digits in each image were encoded by a value of 255. The background, however, varied significantly. Therefore, to reduce the complexity of our dataset, we set all pixels with a value of 255 to 1, and all pixels with any other value to 0. This allowed an easy extraction of the digits from the complex backgrounds of each image.
Digit isolation. The largest digit in each image was isolated by calculating the smallest rectangle to fit each digit of each image. The digit with the rectangle with the longest side length was considered to be the largest digit of each image and was extracted for further processing. This method was a variation from the one suggested by Prof. Ryan Lowe. We also attempted isolating digits by calculating the longest length along the major axis starting from the centroid of each digit. However, this did not yield as accurate results.
Standardization. The largest digit of each image was extracted and rescaled to fit a standard shape of 28x28 pixels. This shape corresponds with MNIST handwritten number images. This allowed us to reduce the number features that must be taken into account by our models, reducing the amount of time to run each process. Our models were also tested with a standardized shape of 80 X 80 pixels, however, their performance differences were negligible. Thus, we concluded to use 28 X 28 to save time.
Augmentation. To increase the size of our training dataset, we employed random augmentations to our original data. These included rotation (±15°) and flipping digits 0, 1, 8 by 180° along the central vertical axis. Our dataset was increased to roughly 80000 examples. Furthermore, as part of the CNN model, we employed further alterations to our data without increasing the total size of our dataset. This included random shifts horizontally/vertically, rotations, image shearing, and zooming. This was done to reduce the chances of overfitting occurring on the training set.
Algorithms
1. Linear SVM.The first model we used was a linear SVM model as a supervised machine learning algorithm for classification. Linear SVM identifies the best hyperplane(s) that would be able to distinguish two classes as clearly as possible usually on data that is linearly separable. Although linear SVMs are great for data with large number of features (28 x 28), they are slow to train for large data sets such as the ones given and will result in lower accuracy if the data isn’t linearly separable. This model was expected to give a fairly worse result due to the fact that the problem needed multiple classifications, which would make it much more difficult to make them linearly separable.

2. Dense feed-forward neural network.A dense feed-forward neural network takes each of the pixels of an image as an input node, and links each of them to all of the nodes in the hidden layer, whos node each are in turn connected to each of the output node. Each of the nodes are passed through the hidden layer, the function h(x) = g(a) is applied; where a = wx + b and h is the relu activation function.  Which is then processed through the activation function - SoftMax- in order for the neural network to be able to perform multi-class classifications. Once fed, the whole network is  run backwards, where the results are used to update the weights and biases through the usage of gradient descent, with response to the loss function - Cross Entropy.

3. Convolutional neural network.The last model we chose, was to improve on the idea of employing a neural network. The convolutional Neural network (CNN) though similar to ordinary neural networks, have a key assumption, that all the inputs are going to be pixels from an image. Each image goes through a convolution step that applies a matrix operation (filters) on certain square pixels of the image, similar to a sliding window moving across the image. This process generates a new matrix which represents the convoled features. These filters can be used to recognize certain features from the images such as edges and curves. The CNN we employed also utilized pooling which downsamples our images to extract the biggest number in its window. By allowing for multiple layers of filters and pooling with the previously described elements of a neural network, the CNN is able to learn more detailed aspects of the images compared to other methods. 
Methodology
1. Linear SVM. For the linear SVM model, we used a a 80-20 training, validation split in our data. We reasoned that our training dataset of ~80000 examples was large enough to employ this cross validation technique over k-fold, or hold-one-out cross validation methods. In addition, it allowed us to run our model within a reasonable amount of time. Cross validation was used to tune the hyperparameter, C or the penalty parameter of the error term. The values that were considered were 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, and 0.9.  Out of these values, the optimal C was determined by the best accuracy performance in the validation set. No other optimization or tuning methods were used. 

2. Dense feed-forward neural network.For this neural network, we used a 80-20 training, validation split as well due to the large size of the original training dataset. We employed cross validation to choose the optimal number of  iterations, and learning rates. The learning rates between the ranges of 0.1 - .0001 were tested while the max epoch was set to 60,000 iterations. From the results of the validation set, we were able to determine the optimal learning rates and epochs to be used for training a neural net. L2 Regularization was applied upon the inputs in order to reduce overfitting the neural network to the training set. While the ReLU function was applied in order to reduce the likelihood of vanishing gradient occuring.

3. Convolutional neural network.For our CNN, we remained with a 80-20 training, validation split due the same reasons as above. We decided to employ four convolutional layers, two with  32 filters each and two with 64 filters each. All convolutional layers had a window size of 3x3. The output matrix was normalized after each convolutional layer to retain similar scales. We  had four activation layers, each coming after a convolutional layer. All activation layers used a ReLU function, due to a reduced likelihood of vanishing gradient. There were two pooling layers, each after two convolutional and activation layers. This was used to downsample and reduce overfitting. The outputs of these layers were used as input to a fully connected layer with 512 nodes. The second last layer contained 10 nodes, each corresponding to digits 0-9. Lastly, the final layer was a softmax activation layer, which converts the output into a probability. For the loss function, we utilized the cross-entropy loss function, and the Adam optimizer with 0.001 learning rate due to faster convergence. Lastly, we employed a dropout rate of 0.2 to reduce overfitting.
Results 
The linear SVM model was able to produce an accuracy of 75.42% after tuning the C hyperparameter. As shown in the left figure, the optimal C value was determined to be 0.1 through a process of hyperparameter tuning. After 0.1, the accuracy on the validation set began to decline. One reason for the poor performance of the linear SVM may be due to the fact that the dataset was not linearly separable. Furthermore, results may have improved if we employed multiple models, one for each digit, rather than using one model to approach this multiclass problem. 
The dense feed-forward neural network performed well with a validation accuracy of ~98%. As shown on the right, as the neural network was trained for longer number of iterations, the loss it experienced rapidly decreased until it hit an asymptote, which was ~%1 for the training set, not seen on the graph,  and ~%2 for the validation set. In addition the learning rate of 0.01 was selected as it seemed to avoid the issue of oscillating that larger learning rates had; while converging to relatively the same point that lower learning rates had, at a faster rate. We tested how changing the architecture would affect the results, by changing how many hidden layers that that the neural network had. Increasing it to 5 layers yielded the best results, but applying any more layers led to the occurrence of vanishing gradients, causing the weight values to be unchanging.

The CNN performed the best with a validation accuracy of 99.6% and 94.57% on the test dataset. The figure to the left shows the increase in performance with the number of epochs. This was expected as the CNN have the assumption that the inputs taken are image pixels. With more epochs and bigger training sets, the performance of the CNN also increases but only to a certain degree. The average of accuracy remains in the range between 97 to 99 % on the validation set. We played around with the architecture with 5 x 5 convolutional layer giving us a lower performance (94% on validation) than 3 x 3 convolutional layer that we chose for our model. From our research (3), we decided on running with an Adam optimizer with a learning rate of 0.01 and then 0.0001 with default values of beta1,2 and epsilon from the keras library, with 0.0001 showing only a 0.05% increase in the test values. Initial weights were randomly chosen through a normal distribution and batch sizes used were only 64. The main hyperparameter we did tune were epochs, with increase in epochs, the results did increase but with sacrifice the time and cpu usage. Also with increase in epochs, the model also showed signs of overfitting, so for our test we chose the optimum epoch of 30. 

Overall, the CNN model performed the best, with the dense network close behind, and the linear SVM performing the worst. This disparity may be due to the fact that this classification problem was non-linear, preventing the linear model from producing as accurate results. The CNN likely performed better than the other neural network due to its ability to extract specific features, such as edges and curves, which provided improved feature detection. 
Discussion 
1. Linear SVM.
As we had an idea that data such as image pixels don’t tend to be linearly separable, it was bound to give a lower accuracy. A suggestion to improve on the usage of svm’s would have been to use a different kernel. From countless previous recreations we can infer that a rbf kernel would have resulted in a much higher accuracy but at the same time would have taken much longer as well.  
2. Dense Forward Feed Neural Network.
    The Dense Feed Forward Neural Network proved to be significantly more accurate than the Linear SVM, but lower accuracy than the CNN. The best way to improve the accuracy of the neural network would be to apply convolutions, and making assumptions that all inputs are pixels from an image; turning it into a Convolutional Neural Networks.  However this would be at the cost of increased runtime, and CPU/GPU usage, something that is already a limiting factor for computers with limited resources. Another possible way that the neural network could be improved from the current state, would be to apply a learning rate decay function, which would help guarantee convergence. However the rate at which it decays at would be another hyper-parameter that would be fine-tuned.
3. Convolution Neural Network.
As expected, the CNN model did result in the best possible accuracy from all the three models, but at the cost of a much higher runtime and cpu utilization. The same training and evaluation could have been speed up by GPU integration but that is not always an option. This was especially evident as we increased the number of epochs, although the accuracy did increase, the runtime sacrificed in comparison was unbalanced for a 0.1% improvement.Our neural network could in fact be improved upon by experimenting with adding more data or preprocessing the data better to pick biggest digit more efficiently as most of our loss came from misclassification during preprocessing. Furthermore, it would be worthwhile to investigate further into tuning hyperparameters, such as convolution window size, number of layers, etc. The general structure and hyperparameters chosen in our model were inspired by previous classifications on the MNIST data, therefore, tuning these to fit our problem would have led to better performance.

Lastly, our performance could have been further improved by implementing a cross-validation technique, such as k-fold validation or hold-one-out. These methods would have allowed our models to be training using the entire dataset, rather than strictly 80%.



Statement of Contributions 
Khalil Mohsin led the creation of our CNN model, both in understanding the problems and coding the solution. He also contributed to the report by adding to the introduction, algorithms, methodology, results, and discussion sections. 

Jongwoo Shim created and coded the dense feed-forward neural network. He also contributed to the report by adding to the algorithms, methodology, results, and discussion sections.

Brian Hyung implemented the image preprocessing as well as the linear SVM model. He also contributed to the creation of the CNN and report by adding to the introduction, feature design, algorithms, methodology, results, and discussion sections.

All team members worked together to identify specific issues and provide insight into better optimization approaches. All members also agreed upon the methodologies, from image processing to optimizations, used in this assignment.

We hereby state that all the work presented in this report is that of the authors.


















References
1. Convolutional Neural-Network for MNIST (2017, January 10). In Github. Retrieved March 22, 2018, from https://github.com/hwalsuklee/tensorflow-mnist-cnn

2. Katariya, Y. (2017, April 15). Github. In Applying Convolutional Neural Network on the MNIST dataset. Retrieved March 22, 2018, from https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/


3. Diederik P. Kingma (2014),ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION University of Amsterdam, OpenAI, https://arxiv.org/pdf/1412.6980


4. Lau, S. (2017, July 10). A Walkthrough of Convolutional Neural Network — Hyperparameter Tuning. In Medium. Retrieved March 22, 2018, from https://towardsdatascience.com/a-walkthrough-of-convolutional-neural-network-7f474f91d7bd


5. Tensorflow Tutorial 2: image classifier using convolutional neural network (n.d.). Retrieved March 22, 2018, from Tensorflow Tutorial 2: image classifier using convolutional neural network


6. CS231n Convolutional Neural Networks for Visual Recognition (n.d.). Retrieved March 22, 2018, from http://cs231n.github.io/convolutional-networks/


7. Support Vector Machines (n.d.). In Scikit Learn. Retrieved March 22, 2018, from http://scikit-learn.org/stable/modules/svm.html

8. Gidudu Anthony, Hulley Greg and Marwala Tshilidzi, Classification of Images Using Support Vector Machines https://arxiv.org/ftp/arxiv/papers/0709/0709.3967.pdf

9. Label image regions (n.d.). In Scikit Image. Retrieved March 22, 2018, from http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html
