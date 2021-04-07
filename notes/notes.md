# CS 231n: Convolutional Neural Networks for Visual Recognition

# [Lecture 1](https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk)

## Computer Vision Overview


### Welcome & Obiquity of visual data
  * Many sensors in the world.
  * Enormous amount of visual data produced in the world.
  * Visual data is a bit like the *dark matter* of the internet.
  * Every second there are 5 hours of new videos uploaded onto Youtube (in 2017)
  * Computer Vision combines various fields.

### A brief historical overview
  * Biological:
   * 540 million years ago the first animals developed eyes.
   * The subsequent 10 million years saw an explosion of species.
   * Vision is hugely important especially for intelligent animals.
  * Machine vision history
    * Camera Obscura
    * Hubel & Wiesel 1959 use electrophysiology to understand what visual processing is like in mammals.
	  * Electrodes in cat brain, simple cells.
    * Block world - Roberts 1963
    * Summer Vision project - Papert 1966
    * Marr 1970s: "Vision": deconstructing visual information into simple, 2.5d, then 3d
    * Moving beyond simple blocks in 1970s: "generalized cylinder" / "pictorial structure": reduces complex structure into simpler shapes.
    * Lowe 1987: combining lines and edges (this looks like cool art to me!)
    * Image Segmentation: graph algorithms, and then face recognition using machine learning, even real time (Viola Jones 2001).
    * Object Recognition using SIFT (Lowe 1999): some features remain invariant: *diagnostic features*.
    * *Spatial Pyramid Matching*: take features from different parts of the image and put together in support vector machine.
    * Similar: Histogram of Gradients, Deformable Part Model (first decade of 2000s).
    * Pascal Visual Object Challenge: benchmark challenge, 20 object categories.
    * Most machine learning algorithms are likely to overfit, because there is a high dimension input and little input.
    * Solution: image-net. 14 million images. Large Scale Visual Recognition Challenge.
    * 2012: first convolutional neural network. (Popular name: deep learning)
	
### Overview of this class
  * Image classification is the focus of this class.
  * May seem restrictive but can be applied all over the place.
  * Also: object detection, action classification, image captioning.
  * Since 2012, breakthrough of convolutional neural networks, this is what always has won the imageneht challenge.
  * CNN foundation: LeCun 1998 at Bell Labs, digit recognition. Similar structure to 2012 AlexNet.
  * Why did it only become so popular recently?
	* Moore's Law
	* GPU's
	* Data
  * Quest for visual intelligence goes far beyond object recognition! Open problems!
	* What does every pixel mean?
	* Activity recognition
	* Problems guided by augmented, virtual reality
  * Example: images as "scene graphs" of large objects.
  * An image tells an entire story.
  * I really liked the **Obama image as an example** of how many layers the story behind a picture can have.

### Logistics
  * Optional text: Deep Learning by Goodfellow, Bengio, Courville, [Free online](https://www.deeplearningbook.org/).
  * Philosophy: understand the algorithms at a deep level. Thorough and detailed. Implementing it all in Python.
  * Also practical. State of the art tools.
  * Fun. Image captioning and artistic things. **I'm excited about this.**
  * Programming assignments in Python, being able to read C++. Calculus and Linear Algebra. Some knowledge about Machine Learning.

# [Lecture 2](https://www.youtube.com/watch?v=OoUX-nOEjG0&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=2)

[2020 slides](http://cs231n.stanford.edu/slides/2020/lecture_2.pdf)

### Introduction
* Can start on first assignment
* There is a Python Numpy tutorial
* Possible to use Google cloud

### Image Classification
* Given a set of discrete labels, like dog, cat, truck, ..., and a picture, give it the most appropriate label.
* A computer just sees an image as a big grid of numbers in [0,255], of dimension 800 x 600 x 3 (RGB channels).
  * "Semantic gap": how to convert this into a concept.
  * Possible challenges: deformation, illumination, camera angle, occlusion (this example is really cute), background clutter, intraclass variation.
  * Some of these things work very fast and well! This is pretty amazing.
* What would be the API? There is no obvious algorithm to hard-code.
  * Attempts have been made: find edges, find corners, ...
  * Even if successful, this is not scalable.
* Solution: Data-Driven Approach.
  * Collect dataset
  * Use Machine Learning to train a classifier
  * Evaluate the classifier on new images
  * API changes a little bit: one function called ```train```, and another called ```predict```.
  
### Nearest Neighbor
* The first example of a simple data-driven approach.
  * Training phase: memorize all,
  * Prediction phase: just predict the closest one.
* Example dataset: CIFAR-10: 10 classes, 50000 training images, 10000 testing images.
* Won't work well but nice to work through.
* But how do we compare?
* First possibility is just to use the l-1 (Manhattan)-distance: sum of absolute value of differences by pixel.
* Training is really fast O(1), predicting is pretty slow O(N).
  * Typically we want the reverse of this!
* Nice visualization shows weird islands and fingers.

### k-nearest neighbor
* It's better to take a majority vote among the k nearest neighbors.
* Larger values of k give smoother regions. You get white regions when there's a tie.
* Pictures are just vectors in a high dimensional space. It is useful to switch back and forth the actual pictures and the vector point of view.
* One can change the distance function: Euclidean distance. This is rotation-invariant and may be more natural when the vectors themselves don't have any particular meaning.
* There is a demo that works in the browser.
* How to choose the right k and distance metric?
* These are examples of **hyperparameters**.

### Setting Hyperparameters
* Idea 1: Choose hyperparameters based on data. 
       + This is a really bad idea: k = 1 will always work perfectly on training date.
   * Idea 2: Split data into train and test. Then pick hyperparameters based on test data. 
       + Also a terrible idea: no idea how algorithm will perform on new data.
   * Idea 3: Three different sets: train, validation, test. It is very important to keep this separate in research.
   * Idea 4: (Especially for small data sets) Cross-Validation. Split data into various 'folds', average the results of trying each fold as validation set. Find robust hyperparameters. In deep learning not used too much in practice.
   * You use this to choose the hyperparameters.

### Why k nearest neighbor is not useful in practice
   * Distance does not correspond to our perception: **cool example** showing different pictures with same Euclidean distance to a sample picture.
   * Curse of dimensionality: in a high dimensional space, everything is far apart. You need a dense number of training samples. The number of training samples needed to densely cover the space grows exponentially with the dimension of the space.
   * But we will implement it in the first homework.

### Linear Classification
   * Simple but important for neural networks. Single building block in a neural network.
   * Parametric approach.
	 + Taking as input: data x and parameters W.
	 + Gives as output a vector of 10 class scores.
   * What to take as function f?
     + Simplest answer: multiply. f(x, W) = Wx + b.
	 + b is called the bias vector.
   * You learn the weight matrix W. You can visualize the rows of W to see what the various class "templates" look like. **This looks really cool**
   * Easy to also understand what it's doing geometrically: drawing hyperplanes around the data of one class. 
   * Hard cases for a linear classifier:
     + Parity problem (odd vs even)
	 + Multimodal data: Two isolated "islands" of data
   * So how do you get the right W? Loss functions, optimization, etc. Next time!

# [Lecture 3](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3)

[2020 slides](http://cs231n.stanford.edu/slides/2020/lecture_3.pdf)

## Introduction
   * You can use Google cloud.
   * Recall:
	 + Recognition is a hard problem
	 + Data-driven approach, k-nearest neighbors
	 + Train, validation and test sets
	 + Linear classifier with parameter matrix W
     + Each row in the matrix W is a template in the class
	 + You can also view it as learning decision boundaries for images in a high-dimensional space
   * To do:
     + Define loss function -> quantifying the badness of W,
	 + Optimization -> searching through the space of all possible W's.

## Loss functions
   * We have a training data set of pairs (x,y):
	 + x is the image
	 + y is the label
   * Loss is average of losses on individual examples.
   
### Multi-class support vector machine
   * For an answer to be marked *correct*, the score given to the true answer should be at least 1 higher than the score given to any other class.
     + (This choice of 1 as a cut-off is irrelevant because we only care about relative differences between scores.)
   * **Hinge loss** is the sum over max(0, s-correct - s-other + 1) where s-correct is the score given to the correct answer and s-other is the score given to another class, as other ranges over all incorrect classes.
   * The explanation and the examples of the hinge loss function here were not so easy to follow. I think a type of example that was missing is where, on a cat picture, the scores are something like (cat=3.2, car=5.1, frog=2.5). Then for the car you have a 2.9 loss because 5.1 - 3.2 + 1 = 2.9, and for the frog score you *also* get a little loss, namely 2.5 - 3.2 + 1 = 0.3.
   * A few observations about this loss function:
     + If performance is really good on a particular example then even if we jiggle the score of the correct answer a little, loss will stay 0.
	 + The minimum loss is 0, the maximum loss is infinte.
	 + The expected initial loss (with a random initialization of W) is the number of classes - 1 (approximately 1 loss for each class except the correct one).
	 + For debugging it is useful to check that the initial loss agrees with what we'd expect.
	 + Could also use the mean, or make the minimum loss 1 by including the training example, but again this does not change anything.
   * You could also look at **square hinge loss**, this changes the algorithm. This means we weigh the mistakes differently.
   * Easily coded in a vectorized way.

### Regularization
   * The loss function can be too well-fitted to the training data.
   * Ockham's Razor: add an explicit *regularization* penalty R. The hyperparameter *lambda* decides how strong this regularization is applied.
   * In a formula we add lambda * R(W) to the loss function.
   * Various regularizations are used: L2-norm of W, L1-norm of W, max-norm of W, a lot more fancy ones.
   * L2 regularization asks to "spread" values through the factor.
   * L2 regularization also corresponds to "MAP inference using a Gaussian prior on W".

### Multinomial Logistic Regression (Softmax)
  * A bit more common in deep learning.
  * We will give meaning to the scores using "softmax/cross-entropy function":
  * Raw scores are viewed as **probabilities**.
  * Suppose the scores are (s0, s1, s2, ..., sn) where the correct answer is 0. Then we interpret the probability that the correct answer is k as exp(sk) / S, where S is the sum over exp(sj) for all j.
  * The *true* probability distribution is (1, 0, 0, ..., 0) and this is what we want to get close to.
  * So the loss on this (generic) example is -log( exp(s0) / S) = log(S) - s0.
     + Minimum loss (but in theory only due to logs and exps) is 0 and maximum is unbounded (but in theory only).
     + Initially the loss should be about log(C) where C is the number of classes.
	 + Suppose we change the score of one datapoint slightly. Then softmax will keep trying to improve, closer and closer to the true probability distribution.

## Optimization
  * You're trying to find the bottom of a valley.
  * You can't really compute it analytically.
  * How would you go about finding the valley then?
  * Strategy 1: Random guesses for W.
  * Strategy 2: Follow the slope.
  * To do so, we use the **negative gradient vector** of the loss function, which is the direction of greatest descent.
  * How to compute it?
	+ First way: finite difference approximation of the (partial) derivatives. But this is super slow.
	+ Can do something analytic here! The loss is a function of W! **Calculus!!**
  * In practice: while we always implement the analytic gradient, we want to check that it is correct with a numerical gradient. **Gradient check**.

### Vanilla Gradient Descent
  * The heart of all our learning algorithms.
  * Hyperparameter: step size / learning rate.
  * We will use fancier update rules that work better in practice.

### A last wrinkle
  * Computing the average loss every time is very expensive.
  * Instead, we just do that for a mini-batch : **stochastic gradient descent**.
  * There's a [very nice demo](http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/) of linear classifiers.
  
## Image features
  * Different features of the image first, instead of the raw pixels.
  * Clever feature transform can sometimes help to make the problem separable by a linear classifier.
  * For example: a color histogram.
  * Histogram of oriented gradients.
  * Another example: "bag of words".
  * The idea of a convolutional neural network is that the feature extraction is not done ahead of time, but keeps getting inferred from the data during training. **I liked this as an explanation.**

# [Lecture 4](https://www.youtube.com/watch?v=h7iBpEHGVNc&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&index=3
[2020 slides](http://cs231n.stanford.edu/slides/2020/lecture_4.pdf)

## Recap
  * Scores function
  * Loss function
  * Data loss plus regularization
  * We want the gradient with respect to W, to find the right W.
  * To do gradient descent: derive analytic gradient, and then check with numerical gradient.

## Backpropagation
  * Represent function as computational graph
  * A simple example: f(x,y,z) = (x + y)z, multiply the gradients moving backwards. **I did not like this example much** Better would be something like (2x + 3y) * 4z, so that you would see the effect of multiplying by a number other than 1...
  * Upstream gradient times local gradient gives new upstream gradient (this is the chain rule).
  * Computation of gradient of another example.
  * We can look at the graph at any granularity, for example the sigmoid function can be regarded as one node.
  * When a function branches, we add the contributions of the gradients coming in when backpropagation, using multivariate chain rule.
  
## Implementation
  * A computational graph has a forward pass and a backward pass.
  * You can modularize this for example using OOP.
  
## Neural networks
  * Making a non-linear function by composing multiple linear functions with various non-linear functions in between.
  * What kind of non-linear functions to use..? We'll see this later.
  * We can stack multiple such layers on top of each other.
  * We say "3 layer neural net" or "2 hidden layer neural net", it means the same.
  
### Loose analogy with biology
  * Analogy with neurons, axons, activation function
  * The activation function in neurons is most like Relu.
