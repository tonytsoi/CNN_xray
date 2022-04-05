# Diagnosing Pneumonia from X-Ray Images Using Convolutional Neural Network
![chest](https://github.com/tonytsoi/CNN_xray/blob/main/charts/IM-0115-0001.jpeg)
## Introduction
Deep learning, also known as neural network, has gained traction in recent years, and is now being widely used across multiple fields, with one of its popular applications being image classification which uses the convolutional neural network (“CNN”). The potential of image classification is also being explored in the field of medical diagnosis. In this article, I discuss how we can build a convolutional neural network that would help us diagnose pneumonia from chest x-ray images.
## Neural Network
First, a neural network can be described as a series of algorithms that has an objective to recognise relationships or patterns in the data. A simple neural network consists of different types of layers, namely the input layer, the output layer and the hidden layers. Each layer contains a number of neurons, and each neuron is made up of algorithms that take the outputs from the previous layer as inputs to generate further outputs and pass them onto the next layer.

The below diagram shows an example architecture of a simple neural network. In this example, there are 4 variables in the input layer represented by the green circles. They are entered into each of the 4 neurons represented by the blue circles in the 1st hidden layer. These neurons would then generate further outputs to be passed onto the neurons in the 2nd hidden layer. Lastly, the outputs of the 2nd hidden layer would go into the neuron in the output layer represented by the orange circle which would generate the final output.

![simple neural network](https://github.com/tonytsoi/CNN_xray/blob/main/charts/simple%20neural%20network.JPG)

## Convolutional Neural Network (CNN)
Convolutional neural network is a class of neural network that are commonly applied to tasks relating to images. Generally, it consists of 3 different types of layers which are the convolutional layer, the pooling layer and the fully connected layer.

A fully-connected layer is basically the same as one of the hidden layers from the simple neural network example above. I discuss the convolutional and pooling layers in turn below.
### Convolutional layer
The convolutional layer, as the name might suggest, applies the convolution operation to the data. This is a mathematical operation that multiplies two arrays of numbers together to produce a third array of numbers.
For example in the illustration below, each value in the 3x3 filter (or kernel) is multiplied to a 3x3 sub-section of the 4x4 image, and the results from the multiplications are summed within each sub-section. The filter is then shifted across the image to the next sub-section and the calculation is repeated until the whole of output array is filled.
![convolution](https://github.com/tonytsoi/CNN_xray/blob/main/charts/Convolution_1.gif)

The filter in the example can be considered as an edge detector which would identify edges from the image. Other types of features can be identified with different filters. In a convolutional layer, we would often have multiple filters to detect different types of features.

However, we might not necessarily know which filters are the most appropriate to be applied in our task. Therefore, instead of setting our own filters, the neural network would determine the optimal filters that best detect the features required for our classification task.

Images are often expressed as RGB values which consists of 3 values ranging from 0 to 255 corresponding to the level of red, green and blue. Therefore, the convolution filters would often be 3 dimensional (e.g. 3x3x3).
### Pooling layer
A pooling layer reduces the dimensions of the data by simplifying a sub-section of the image into a single value. For example in the illustration below, applying max pooling would shrink the image by taking the largest value in each sub-section of the image as the output.
![pooling](https://github.com/tonytsoi/CNN_xray/blob/main/charts/max%20pooling.gif)
## Data Preparation and Augmentation
In this article, I use the dataset from Kaggle which includes 5,856 chest x-ray images labelled either “Pneumonia” or “Normal”. These chest x-ray images were selected from retrospective cohorts of paediatric patients of one to five years old from Guangzhou Women and Children’s Medical Centre. These x-ray images are split into the training, validation and test sets. There are 5,216 images in the training set, 624 images in the validation set and 16 images in the test set. I save the directories of the folders and name them accordingly.

Below I plot 10 of the x-ray images from the training set, with 5 of them being “Normal” and 5 of them being “Pneumonia”. In an ordinary chest x-ray image examination, the radiologist would look for white spots in the lungs to determine if a patient suffers from pneumonia. However, as non-medical professionals, we might not necessarily have the expertise or experience to tell the difference between a “Pneumonia” image and a “Normal” image.
![x-ray-images](https://github.com/tonytsoi/CNN_xray/blob/main/charts/x-ray-images.jpg)

Let’s see if a neural network could help us as non-medical professionals to differentiate between chest x-ray images with and without Pneumonia. First, I set out my steps on data preparation and augmentation.

I use the image pre-processing library in Tensorflow to extract and generate batches of images from the folders. As mentioned above, images are expressed in RGB values ranging from 0 to 255. In the convolutional neural network, each image should contribute only an equal amount of information. However, a greater weight might be placed to an image with high pixel values than an image with low pixel values. Therefore, I apply a rescaling factor of 1/255 to the images to normalise them to have a value between 0 and 1.

I also apply sample-wise centring and standardisation to centre and standardise the images. The image would have a mean of zero and a standard deviation of one, derived by subtracting the mean pixel value and dividing by the standard deviation of the pixel values in each image.

To reduce the impact that the model might overfit the training set and does not generalise well to unseen images, I apply data augmentation to generate more images that look slightly different then the originals to be included in the dataset. The additional images are randomly zoomed by a factor of 0.2 and their heights and widths are randomly shifted by a fraction of 0.1 compared with the total.

I apply data augmentation to both the training and validation sets to assess how well the model does across a similar distribution of images with similar data augmentations. Nevertheless, I do not apply data augmentation but only rescaling and standardisation to the test set, to assess how well the model does on images that the model would more likely encounter under normal circumstances which the image is less likely to be distorted.

The class mode is set to binary as the category is either “Pneumonia” or “Normal”. I shrink the images to a size of (160, 160) to reduce the number of inputs. The batch size is set to be 16 which means the internal parameters in the model would be updated every 16 samples it has run through. It can be thought of as that the model would take a small step towards the optimum every 16 samples it has seen.
## Building the Convolutional Neural Network
I set out the structure of my convolutional neural network below. There are 4 blocks of layers with a convolution layer followed by batch normalisation and a max pooling layer. For the convolution layers across all four blocks, the number of filters increases from 32 to 128 with a kernel size of 3x3.

After the convolution layer in each block, I apply batch normalisation to normalise the results. They are then entered into the activation function, which I set to be the ReLU function. I then apply the max pooling layer with a pooling size of 2x2.

After four blocks convolutional and pooling layers, I flatten out the layer and include a fully-connected layer with 512 neurons with ReLU as the activation function. Finally, I include an output layer with 1 neuron with the sigmoid function as the activation function for our binary classification task.

I use stochastic gradient descent as the optimisation method with a learning rate that decays exponentially from 0.01 and binary cross entropy as the loss function.

I then train the model with 20 epochs (or cycles), with the steps per epoch as the number of images in the training set divided by the batch size, which would allow the whole of the training set to be trained in each epoch. I set out the model summary below.
![model_summary](https://github.com/tonytsoi/CNN_xray/blob/main/charts/model_summary.JPG)

I plot the various metrics including accuracy, loss, precision and recall across epochs below.
![accuracy](https://github.com/tonytsoi/CNN_xray/blob/main/charts/accuracy.jpg)
![loss](https://github.com/tonytsoi/CNN_xray/blob/main/charts/loss.jpg)

The accuracy in the training set increases to above 95% that we might be tempted to jump to the conclusion that the model is doing a pretty good job. However, when we look at the accuracy rate in the validation set across epochs, it hovers at around 75% to 90% with no clear trend of increasing as we train the model through more epochs.

The loss in the training set is trending downwards, whereas that in the validation loss does not show a clear downward trend after the first few epochs.

![precision](https://github.com/tonytsoi/CNN_xray/blob/main/charts/precision.jpg)
![recall](https://github.com/tonytsoi/CNN_xray/blob/main/charts/recall.jpg)

Similar observations can be identified in precision and recall, in which there are no clear convergence between the training and validation sets.

We also see that the overall precision is lower than the accuracy in the validation set which suggests that a certain percentage of predictions are false positives; whereas the recall in the validation set is generally high, which suggests a low percentage of predictions are false negatives.

The above observations could imply that our model suffer from the problem of overfitting, which the model overfits the training set and does not generalize well when it is met with new images that it has never seen.
## Confusion matrix
Below I set out the confusion matrices of the model in predicting the training, validation and test sets.
![training confusion](https://github.com/tonytsoi/CNN_xray/blob/main/charts/Training%20set%20confusion.jpg)
In the training set, the model correctly predicts most of the “Pneumonia” images, and “Normal” images with at a slightly lower accuracy.
![validation confusion](https://github.com/tonytsoi/CNN_xray/blob/main/charts/Validation%20set%20confusion.jpg)
In the validation set, the model accurately predicts the “Pneumonia” images but not so well in the “Normal” images.
![test confusion](https://github.com/tonytsoi/CNN_xray/blob/main/charts/Test%20set%20confusion.jpg)
A similar pattern can be seen in the test set that the model could not effectively differentiate the “Normal” images from the “Pneumonia” images.

From the results in the validation and test sets, we can conclude that the model could correctly predict most chest x-ray images with pneumonia but not those that are normal.
## Next steps
The above result could be due to the imbalance in the dataset as there is larger proportion of “Pneumonia” images but relatively fewer “Normal” images. The neural network might not have seen enough “Normal” images to be able to correctly identify them.

Furthermore, overfitting appears to remain as a problem despite our expansion of the dataset with data augmentation. Therefore, collecting more chest x-ray images to be added to our dataset, in particular the normal images would likely give us a more accurate model.

We could also explore the use of transfer learning. Transfer learning is to adopt a pre-trained model that was trained on a very large set of images and re-train the top layers for more specific tasks. This has the potential to outperform our current model as the pre-trained model has already learnt about the basic features from the very large set of images, and that would ideally help the model to deal with more complex tasks.
## Conclusion
Convolutional neural network as a branch of neural network is often used in image related tasks and can be applied in different fields such as medical diagnosis. Nevertheless, deep learning is data hungry that in order to train a well-performed model, a large amount of data is required which could be an obstacle in the medical field.
