---
layout: post
title: Fruit Classification Using A Convolutional Neural Network
image: "/posts/cnn-fruit-classification-title-img.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python, TensorFlow/Keras]
---

In this project I build and optimize a Convolutional Neural Network (CNN) to classify images of fruits. There are many applications of CNNs, one example being to classify images of items or products in order to enhance sorting and delivery processes for a business.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth and Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. Data Pipeline](#data-pipeline)
- [03. Convolutional Neural Network Overview](#cnn-overview)
- [04. Baseline Network](#cnn-baseline)
- [05. Combatting Overfitting with Dropout](#cnn-dropout)
- [06. Image Augmentation](#cnn-augmentation)
- [07. Hyperparameter Tuning](#cnn-tuning)
- [08. Transfer Learning](#cnn-transfer-learning)
- [09. Results Discussion](#cnn-results)
- [10. Growth and Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

The goal of this project is to accurately label images of fruits. The ability to classify images of products, some of them looking quite similar to each other, could be used in a business on a large scale, for instance in conjunction with a camera and robotics to identify and sort items for delivery.

<br>

### Actions <a name="overview-actions"></a>

I use the **Keras** Deep Learning library from TensorFlow in Python for this task.

First, I create a pipeline for feeding training and validation images from a local directory into the network in batches. I analyze the performance on the validation set epoch-by-epoch as the network trains, and then also on a separate test set of images.

The baseline network is simple, but it serves as a starting point that can be further refined. The network contains **2 Convolutional layers**, each with **32 filters** and subsequent **Max Pooling layers**. After flattening, there is a **single Dense (Fully Connected) layer** with **32 neurons**, followed by the output layer. I apply the **`relu`** activation function on all layers, and use the **`adam`** optimizer.

The first custom network refines the baseline network by adding **Dropout** to tackle the issue of overfitting.

Next, I try a second custom network, adding **Image Augmentation** to the data pipeline to increase the variation in the input images for the network to learn from. This produces more generalizable, robust results as well as helping to prevent overfitting.

Then, I use **Keras Tuner** to optimize the network architecture and tune the hyperparameters. The best network architecture from this testing is one with **1 Convolutional layer**. The input layer has **192 filters**, and the convolutional layer has **128 filters**. The output of this layer is flattened and passed to **single Dense (Fully Connected) layer** with **64 neurons**. The Dense layer has **Dropout** applied with a dropout rate of 0.5. The output of this is passed to the output layer. The **`relu`** activation function is applied to all layers, and the **`RMSProp`** optimizer is used.

Finally, I use **Transfer Learning** to compare the previous networks' results against that of the pre-trained **VGG16** network.

<br>

### Results <a name="overview-results"></a>

The baseline network did not have very high accuracy due to overfitting, but the addition of both Dropout and Image Augmentation eliminated overfitting almost entirely.

The test set classication accuracy percentages for each network architecture were:

* Baseline Network: **80%**
* Baseline + Dropout: **92%**
* Baseline + Image Augmentation: **95%**
* Optimized (Tuned) with Dropout + Image Augmentation: **95%**
* Transfer Learning using VGG16: **93%**

Tuning the network's architecture with Keras Tuner improved the performance over the baseline, but was very time intensive. That time investment could be worth it if it resulted in improved accuracy, but that was not the case in this particular instance, as the baseline model with image augmentation added had the same accuracy on test set.

Using Transfer Learning with the VGG16 architecture was quite successful. In 15 epochs it achieved accuracy similar to the smaller, custom networks that all trained 50 epochs. From a business point of view, there is overhead in storing a very large VGG16 network file and in possible increased latency on inference, so there are tradeoffs to consider when choosing a CNN architecture.

In this case, since Transfer Learning is not more accurate, the Baseline network with image augmentation seems sufficient for this task (so far).

<br>

### Growth and Next Steps <a name="overview-growth"></a>

This project had very accurate predictions for images, although on a small number of classes. The best networks above could be tested on a larger number of image classes to gain more insight into the different networks' performance and to see whether they still perform well.

The Transfer Learning network performed a little worse than the Baseline + Image Augmentation or the Tuned with Dropout + Image Augmentation networks in terms of classification accuracy on the test set, but it was only trained for 15 epochs, which is small. It could be trained for more epochs, and it would also be interesting to try transfer learning using other available pre-trained networks.

___

# Data Overview  <a name="data-overview"></a>

The data is made up of images of six different types of fruit (apple, avocado, banana, kiwi, lemon, and orange). All images are 300 x 200 pixels. I randomly split the images for each fruit into training (60%), validation (30%) and test (10%) sets. Examples of four images of each fruit class can be seen in the image below:

![CNN Fruit Image Examples](/img/posts/cnn-image-examples.png "CNN Fruit Classification Samples")

<br>

The folder structure consists of separate training, validation, and test directories, and within each of those are directories for each of the six fruit classes.

___

# Data Pipeline  <a name="data-pipeline"></a>

Before building the network architecture, and subsequently training and testing it, a pipeline needs to be set up for the images to flow through, from the local hard drive where they are located, to the network, and through the network.

The code below does the following:

* Imports the required packages
* Sets up the parameters for a pipeline
* Sets up image generators to process the images as they come in
* Sets up the generator flow, specifying what to pass in for each iteration of training

```python

# Import the required Python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# Image generators - scale images to the same scale for faster learning
training_generator = ImageDataGenerator(rescale = 1./255)
validation_generator = ImageDataGenerator(rescale = 1./255)

# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')
validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                                      target_size = (img_width, img_height),
                                                                      batch_size = batch_size,
                                                                      class_mode = 'categorical')

```

<br>

The above code specifies that images will be resized down to 128 x 128 pixels, and 32 images at a time (batch size) will be passed in for training. The image generators rescale the raw pixel values (ranging between 0 and 255) to float values that exist between 0 and 1. This helps Gradient Descent find an optimal or near optimal solution more efficiently because the features that are learned in the depths of the network are of a similar magnitude, making the learning rate somewhat proportionally similar across all dimensions. (I.e., training time will be less than it would with no rescaling.)

With this pipeline in place, images will be extracted in batches of 32 from their location on the hard drive. Next they will be sent to the network for training.

___

# Convolutional Neural Network Overview <a name="cnn-overview"></a>

Convolutional Neural Networks (CNNs) are an adaptation of Artificial Neural Networks (ANNs) and are primarily used for image-based tasks.

To a computer, an image is simply made up of numbers --- the RGB color intensity values for each pixel. The intensity value for each color ranges from 0 to 255. These pixel values are the input for a CNN. The CNN needs to make sense of these values to make predictions about the image. The pixel values themselves don't hold much useful information on their own, so the network needs to turn them into *features*, much like humans do without knowing it when we look at a picture.

A big part of the CNN process is called **Convolution**. Each input image is scanned and compared to many different, smaller filters, to pare the image down into something more generalized. This process helps reduce the network's sensitivity to minor changes so that it can know that two images are of the same object, even though the images are not exactly the same.

A somewhat similar process called **Pooling** is also applied to carry this generalization further. A CNN can contain many of these Convolutional and Pooling layers, with deeper layers finding more abstract features.

Similar to ANNs, **Activation Functions** are applied to the data as it moves forward through the network, helping the network decide which neurons will fire, i.e., helping the network understand which neurons are more or less important for different features and ultimately for the different output classes.

As a CNN trains, it iteratively calculates how well it is predicting on the known classes that are passed to it. It does this using a **Loss Function**. Then in a process known as **Backpropagation** it updates the parameters within the network in a way that reduces the error, improving the match between predicted outputs and actual outputs. Over time, it learns to find a good mapping between the input data and the output classes.

There are many parameters that can be changed in the architecture of a CNN, which can affect the predictive accuracy. Many of these parameters will be experimented with below.

___

# Baseline Network <a name="cnn-baseline"></a>

#### Network Architecture

The baseline network is simple, but serves as a starting point that can be further refined. This network contains **2 Convolutional layers**, each with **32 filters** and subsequent **Max Pooling layers**. After flattening, there is a **single Dense (Fully Connected) layer** with **32 neurons**, followed by the output layer. The **`relu`** activation function is applied on all layers, with the **`softmax`** activation function for the output layer, and the **`adam`** optimizer is applied.

```python

# Network architecture
model = Sequential()

# Filters = # of feature maps being output from the layer
# Size of filter = kernel size (3, 3) = 3x3 pixel sections
# Padding = 'same' means padding will be added in a way that ensures we can use all pixels
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# View network architecture
model.summary()

```

<br>

The below summary output from the code above shows the baseline network architecture:

```

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 32)      896       
_________________________________________________________________
activation (Activation)      (None, 128, 128, 32)      0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 32)        9248      
_________________________________________________________________
activation_1 (Activation)    (None, 64, 64, 32)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 32768)             0         
_________________________________________________________________
dense (Dense)                (None, 32)                1048608   
_________________________________________________________________
activation_2 (Activation)    (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 6)                 198       
_________________________________________________________________
activation_3 (Activation)    (None, 6)                 0         
=================================================================
Total params: 1,058,950
Trainable params: 1,058,950
Non-trainable params: 0
_________________________________________________________________

```

<br>

#### Training the Network

With the pipeline and architecture in place, the baseline network is ready to be trained.

The below code:

* Specifies the number of epochs for training
* Sets a location for the trained network to be saved
* Sets a **`ModelCheckPoint`** callback to save the best network at any point during training, based on validation accuracy
* Trains the network and saves the results to an object called **`history`**

```python

# Training parameters
num_epochs = 50
model_filename = 'models/fruits_cnn_v01.h5'

# Callbacks
save_best_model = ModelCheckpoint(filepath = model_filename,
                                  monitor = 'val_accuracy',
                                  mode = 'max',
                                  verbose = 1,
                                  save_best_only = True)

# Train the network
history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size = batch_size,
                    epochs = num_epochs,
                    callbacks = [save_best_model])

```

<br>

The **`ModelCheckpoint`** callback saves the best network out of all 50 epochs, in terms of the CNNs performance on the validation set. At the end of each of the 50 epochs, Keras assesses the performance on predicting the validation set. If it is has not seen any improvement in performance it will do nothing. If there is an improvement it will update the network file that is saved on the hard drive.

<br>

#### Analysis of Training Results

As the training process was saved to the **`history`** object, the performance (Classification Accuracy and Loss) of the network can now be analyzed for each epoch.

```python

import matplotlib.pyplot as plt

# Plot the validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# Get the best epoch performance
max(history.history['val_accuracy'])

```

<br>

The two plots below in the image below show the **Loss** and the **Classification Accuracy** for both the training set (blue) and the validation set (orange).

![CNN Baseline Accuracy Plot](/img/posts/cnn-baseline-accuracy-plot.png "CNN Baseline Accuracy Plot")

<br>

These plots show that with the baseline architecture and parameters set for training, the best performance is reached at around 10-20 epochs, after which not much improvement or change is seen. There is also a significant gap between orange and blue lines on the plot, which means there is a difference in the CNN's performance on the validation versus training images, with validation accuracy being lower than the training accuracy. This gap means there is overfitting --- the CNN is good at classifying the training images, but this doesn't generalize well to the validation images.

Focusing on the Classification Accuracy plot above, it looks like the network is learning the features of the training data so well that after about 20 epochs it is perfectly predicting those images, but on the validation data it never passes about **79% Classification Accuracy**.

Sections later on will address the overfitting to get more accurate image classifications using this network.

<br>

#### Performance on the Test Set

The CNN's performance was assessed above on both the training set and the validation set, both of which were being passed in during training. Now predictions will be made on the test set images in order to assess how well the network performs when classifiying images that were not any part of the training process.

The below code does the following:

* Imports the required packages for importing and manipulating the test set images
* Sets up parameters for the predictions
* Loads in the saved network file from training
* Creates a function for preprocessing test set images in the same way that training and validation images were
* Creates a function for making predictions and returning predicted class label and predicted class probability
* Iterates through the test set images, preprocessing and passing each one to the network for prediction
* Creates a Pandas DataFrame to hold the prediction data

```python

# Import required packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# Parameters for prediction
model_filename = 'models/fruits_cnn_v01.h5'
img_width = 128
img_height = 128
labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# Load model
model = load_model(model_filename)

# Image preprocessing function
def preprocess_image(filepath):
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = image * (1./255)
    return image

# Image prediction function
def make_prediction(image):
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]
    return predicted_label, predicted_prob

# Loop through test data
source_dir = 'data/test/'
folder_names = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']
actual_labels = []
predicted_labels = []
predicted_probs = []
filenames = []

for folder in folder_names:
    images = listdir(source_dir + '/' + folder)
    for image in images:
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_prob = make_prediction(processed_image)
        
        actual_labels.append(folder)
        predicted_labels.append(predicted_label)
        predicted_probs.append(predicted_prob)
        filenames.append(image)
        
# create dataframe to analyze
predictions_df = pd.DataFrame({"actual": actual_labels,
                               "predicted": predicted_labels,
                               "prob": predicted_probs,
                               "filename": filenames})

predictions_df['correct'] = np.where(predictions_df['actual'] == predictions_df['predicted'], 1, 0)

```

<br>

After running the code above, the Pandas DataFrame **`predictions_df`** contains prediction data for each image in the test set. A random sample of this can be seen in the table below:

| **actual** | **predicted** | **prob** | **filename** | **correct** |
|---|---|---|---|---|
| apple | lemon | 0.999 | apple_0034.jpg | 0 |
| avocado | avocado | 1.000 | avocado_0054.jpg | 1 |
| banana | lemon | 0.597 | banana_0084.jpg | 0 |
| kiwi | kiwi | 0.657 | kiwi_0094.jpg | 1 |
| lemon | banana | 0.561 | lemon_0064.jpg | 0 |
| orange | orange | 0.999 | orange_0064.jpg | 1 |

<br>

This data shows:

* **`actual`**: The true label for the test set image
* **`predicted`**: The predicted label from the network for the image
* **`prob`**: The network's probability for the predicted label being correct
* **`filename`**: The image filename for reference
* **`correct`**: A flag indicating whether the predicted label matches the actual label

This DataFrame can be used to calculate the classification accuracy, to look at images the network got wrong (or got right but with low probability), and to attempt to assess why certain images were misclassified. This can help with making improvements to the network or to the quality of the input data.

<br>

#### Test Set Classification Accuracy

Using the DataFrame, the overall test set classification accuracy is calculated with the below code:

```python

# Overall test set accuracy
test_set_accuracy = predictions_df['correct'].sum() / len(predictions_df)
print(test_set_accuracy)

```

<br>

The baseline network achieves a **80% Classification Accuracy** on the test set. We can try to improve accuracy by adding to or refining the network.

<br>

#### Test Set Confusion Matrix

Overall Classification Accuracy is useful, but sometimes does not tell us all that much about what is really going on with the network's predictions. For instance, the Classification Accuracy for the whole test set was 80%, but it doesn't tell us which types of fruit the network is predicting well versus struggling to predict. It also doesn't tell us if two types of fruit in particular are getting confused with each other. A **Confusion Matrix** can give us these insights.

The below code creates a Confusion Matrix:

```python

# Confusion matrix (percentages)
confusion_matrix = pd.crosstab(predictions_df['predicted'], predictions_df['actual'], normalize = 'columns')
print(confusion_matrix)

```

<br>

This results in the following output:

```

actual     apple  avocado  banana  kiwi  lemon  orange
predicted                                             
apple        0.8      0.0     0.0   0.0    0.0     0.2
avocado      0.0      1.0     0.0   0.0    0.0     0.0
banana       0.0      0.0     0.6   0.1    0.2     0.0
kiwi         0.1      0.0     0.0   0.9    0.1     0.0
lemon        0.1      0.0     0.4   0.0    0.7     0.0
orange       0.0      0.0     0.0   0.0    0.0     0.8

```

<br>

The column labels are the *actual* classes, and the row labels are the *predicted* classes. We can look down each column to find the Classification Accuracy for each class and see where the network is misclassifying.

While the overall test set accuracy was 80%, the following are the accuracy percentages for each individual type of fruit:

* Apple: 80%
* Avocado: 100%
* Banana: 60%
* Kiwi: 90%
* Lemon: 70%
* Orange: 80%

This shows what is driving the overall Classification Accuracy. The network is not predicting bananas or lemons well, in particular, with 60% and 70% accuracy, respectively. Of the ones it gets wrong, it always thinks the bananas are lemons and often thinks the lemons are bananas.

___

# Combatting Overfitting with Dropout <a name="cnn-dropout"></a>

#### Dropout Overview

**Dropout** is a technique used in Deep Learning primarily to reduce the effects of overfitting, which occurs when the network learns the patterns of the training data so specifically that it does not generalize well and is unreliable in predicting on new data.

Dropout is the practice of ignoring a preset proportion of neurons in a hidden layer for each batch of observations that is sent forward through the network. Dropout can be applied to any number of the hidden layers. This means the ignored or deactivated neurons do not pass any information through the network.

Over time, with different combinations of neurons being ignored for each mini-batch of data, the network becomes more adept at generalizing and thus is less likely to overfit to the training data. Since no particular neuron can rely on the presence of other neurons and the features they represent, the network learns more robust features and is less susceptible to noise.

In a CNN, it is generally best practice to only apply Dropout to the Dense (Fully Connected) layer or layers, rather than to the Convolutional layers.

<br>

#### Updated Network Architecture

As with the baseline network, there is only one Dense layer for this updated network. The code below applies Dropout of 50% (a commonly used proportion) to that layer only. All network parameters remain the same as the baseline network above.

```python

model = Sequential()

model.add(Conv2D(filters = 32, 
                 kernel_size = (3, 3), 
                 padding = 'same', 
                 input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters = 32, 
                 kernel_size = (3, 3), 
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.5))

model.add(Dense(units = num_classes, activation = 'softmax'))

```

<br>

#### Training the Updated Network

The same code from the baseline network is run to train this updated network with 50 epochs, the only change being a modified filename for the saved network with Dropout to keep all the network files separate for comparison.

<br>

#### Analysis of Training Results

The below image shows the same two plots we analyzed for the updated network, the first showing **Loss** and the second showing the **Classification Accuracy** for both the training set (blue) and the validation set (orange) for each epoch.

![CNN Dropout Accuracy Plot](/img/posts/cnn-dropout-accuracy-plot.png "CNN Dropout Accuracy Plot")

<br>

There is a peak Classification Accuracy for the validation set of around **91%**, which is higher than the **79%** for the baseline network.

Also, the gap between the Classification Accuracy on the training set and the validation set has been reduced, although not entirely eliminated. The two lines are trending up at more or less the same rate across all epochs, and the accuracy for the training set never reaches 100% unlike before. This means it is *generalizing* better. The goal is to get a network that generalizes so well that no neuron or combination of neurons in the network is becoming overly tied to certain features found in the training images. There is still room for improvement.

<br>

#### Performance on the Test Set

As was done for the baseline network, to check performance on the test set the same code from above will be run. The only change is to load in the network file for the updated network rather than the baseline network.

<br>

#### Test Set Classification Accuracy

The baseline network achieved an **80% Classification Accuracy** on the test set. With the addition of Dropout there was a reduction in overfitting and an increased *validation set* accuracy. On the test set there is also an increase in accuracy over the baseline network, with a **92% Classification Accuracy**. 

<br>

#### Test Set Confusion Matrix

As mentioned above, overall Classification Accuracy is useful, but sometimes does not tell us all that much about what is really going on with the network's predictions. The Confusion Matrix will show if the same fruits are still getting confused with the same other fruits as in the baseline network.

Running the same code from the baseline section on results for the updated network gives the following Confusion Matrix:

```

actual     apple  avocado  banana  kiwi  lemon  orange
predicted                                             
apple        1.0      0.0     0.0   0.0    0.0     0.0
avocado      0.0      1.0     0.0   0.1    0.0     0.0
banana       0.0      0.0     0.8   0.0    0.0     0.0
kiwi         0.0      0.0     0.0   0.8    0.0     0.0
lemon        0.0      0.0     0.2   0.0    0.9     0.0
orange       0.0      0.0     0.0   0.1    0.1     1.0

```

<br>

The column labels are the *actual* classes, and the row labels are the *predicted* classes. We can look down each column to find the Classification Accuracy for each class and see where the network is misclassifying.

While the overall test set accuracy was 92%, the following are the accuracy percentages for each individual type of fruit:

* Apple: 100%
* Avocado: 100%
* Banana: 80%
* Kiwi: 80%
* Lemon: 90%
* Orange: 100%

All classes here are being predicted *at least* as well as with the baseline network. Some bananas are still being misclassified as lemons, but lemons are now being classified correctly 90% of the time, and the only one that was incorrect was classified as an orange. This is an improvement over the baseline network.

___

# Image Augmentation <a name="cnn-augmentation"></a>

#### Image Augmentation Overview

**Image Augmentation** is a Deep Learning method that can increase predictive performance and the robustness of the network. Instead of passing in each of the training set images exactly as-is, many transformed versions of each image are passed through the network. This results in increased variation within the training data without having to provide any new images.

Common transformation techniques are:

* Rotation
* Horizontal/Vertical Shift
* Shearing
* Zoom
* Horizontal/Vertical Flipping
* Brightness Alteration

When applying Image Augmentation using Keras' **`ImageDataGenerator`** class, the network does not train on the *original* training set image but instead on several *transformed* versions of the image. For each epoch during training, each image will be randomly transformed based on the specified parameters. This variation will allow the network to generalize better for many different scenarios.

<br>

#### Implementing Image Augmentation

Image augmentation is applied directly in the **`ImageDataGenerator`** class that already existed in the baseline data pipeline. This is only done for the training images, not for the validation or test sets. The validation and test data should be static, which allows measuring performance over time. If the images in these sets kept changing we could not tell whether the network was actually improving or if it was a random set of validation set transformations that made it perform better.

In the code below, the Image Augmentation parameters are added in so that as images flow into the network for training, the transformations will be applied. The parameters limit the magnitudes that can be applied for each type of transformation:

```python

# Image generators
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)

```

<br>

The **`rotation_range`** of 20 is the maximum degrees of rotation that can be applied. A rotation value will be randomly selected for each image, each epoch, between negative and positive 20 degrees.

The **`width_shift_range`** and a **`height_shift_range`** of 0.2 are the fraction of the total width and height that the image can shift horizontally and vertically.

The **`zoom_range`** of 0.1 allows a maximum of 10% inward or outward zoom.

Because **`horizontal_flip`** is True, each image has a 50/50 chance of being flipped.

The **`brightness_range`** between 0.5 and 1.5 means images can become brighter or darker.

Finally, **`fill_mode`** set to **`'nearest'`** means that when images are shifted and/or rotated, the nearest pixel will be used to fill in any missing pixels.

<br>

#### Updated Network Architecture

The network setup will be the same as the baseline network. Dropout will not be applied in this network in order to see the impact of Image Augmentation.

<br>

#### Training the Updated Network

The same code from the baseline network is run to train this updated network with 50 epochs, the only change being a modified filename for the saved network with Image Augmentation to keep all the network files separate for comparison.

<br>

#### Analysis of Training Results

It will be interesting to see whether the addition of Image Augmentation helps with the problem of overfitting in the baseline model in the same way that Dropout did.

The below image shows the same two plots we analyzed for the updated network, the first showing **Loss** and the second showing the **Classification Accuracy** for both the training set (blue) and the validation set (orange) for each epoch.

![CNN Augmentation Accuracy Plot](/img/posts/cnn-augmentation-accuracy-plot.png "CNN Augmentation Accuracy Plot")

<br>

There is a peak Classification Accuracy for the validation set of around **97%**, which is higher than the approximately **79%** for the baseline network, and higher than the **91%** for the network with Dropout added.

Also, the gap between the Classification Accuracy on the training set and the validation set has been mostly eliminated. The two lines are trending up at more or less the same rate across all epochs, and the accuracy for the training set never reaches 100% unlike the baseline model. This means the network with Image Augmentation is *generalizing*. Since the network is getting a slightly different version of each image for epoch during training, the network cannot cling to a single version of those features when it is learning.

<br>

#### Performance on the Test Set

As was done for the baseline network, to check performance on the test set the same code from above will be run. The only change is to load in the network file for the updated network rather than the baseline network.

<br>

#### Test Set Classification Accuracy

The baseline network achieved an **80% Classification Accuracy** on the test set. With the addition of Image Augmentation there was a reduction in overfitting and an increased *validation set* accuracy. On the test set there is also an increase in accuracy over the baseline network, with a **95% Classification Accuracy**. 

<br>

#### Test Set Confusion Matrix

As mentioned above, overall Classification Accuracy is useful, but sometimes does not tell us all that much about what is really going on with the network's predictions. The Confusion Matrix will show if the same fruits are still getting confused with the same other fruits as in the baseline network.

Running the same code from the baseline section on results for the updated network gives the following Confusion Matrix:

```

actual     apple  avocado  banana  kiwi  lemon  orange
predicted                                             
apple        0.9      0.0     0.0   0.0    0.0     0.0
avocado      0.0      1.0     0.0   0.1    0.0     0.0
banana       0.0      0.0     1.0   0.0    0.0     0.0
kiwi         0.1      0.0     0.0   0.8    0.0     0.0
lemon        0.0      0.0     0.0   0.0    1.0     0.0
orange       0.0      0.0     0.0   0.1    0.0     1.0

```

<br>

The column labels are the *actual* classes, and the row labels are the *predicted* classes. We can look down each column to find the Classification Accuracy for each class and see where the network is misclassifying.

The overall test set accuracy was 95%, and the following are the accuracy percentages for each individual type of fruit:

* Apple: 90%
* Avocado: 100%
* Banana: 100%
* Kiwi: 80%
* Lemon: 100%
* Orange: 100%

All classes here are being predicted *at least* as well as with the baseline network, except for kiwis. Bananas and lemons were not misclassified at all. It seems that Image Augmentation really helps the model generalize the features related to these fruits. There is a big improvement over the baseline network.

___

# Hyperparameter Tuning <a name="cnn-tuning"></a>

#### Keras Tuner Overview

The addition of Dropout and Image Augmentation in separate models boosted accuracy. Different network *architecture* could also have a big impact on how well the network learns to find and make use of important features for classifying fruits.

So far, the networks have all used 2 convolutional layers, each with 32 filters, and a single Dense layer with 32 neurons. Maybe a different number of filters such as 64 would have an impact, or keeping the first convolutional layer at 32 filters but increasing the second to 64. The number of neurons in the hidden layer or using an optimizer other than Adam could also change the network's performance.

**Keras Tuner** makes finding the optimal network parameters much easier. Keras Tuner will test many different architecture and parameter options based on some specified limits. It will run tests and return summary statistics, showing what works best.

Once the highest performing architecture is found, it can be used to train the network and analyze the performance against the previous three networks.

The data pipeline will remain the same as it was when applying Image Augmentation. The code below shows this, along with extra packages needed for Keras Tuner.

```python

# Import the required Python libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
import os

# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 128
img_height = 128
num_channels = 3
num_classes = 6

# Image generators
training_generator = ImageDataGenerator(rescale = 1./255,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(rescale = 1./255)

# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')
validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

```

<br>

#### Application Of Keras Tuner

The code below specifies what Keras Tuner should test. The **`build_model`** function builds the network architecture based on randomized parameters. These parameters include:

* Convolutional Layer Count --- between 1 and 3
* Convolutional Layer Filter Count --- between 32 and 256 with a step size of 32 (can be different for each convolutional layer)
* Dense Layer Count --- between 1 and 4
* Dense Layer Neuron Count --- between 32 and 256 with a step size of 32 (can be different for each dense layer)
* Application Of Dropout --- yes or no (can be different for each dense layer)
* Optimizer --- Adam or RMSProp

```python

# Architecture
def build_model(hp):
    model = Sequential()
    
    # Filters = # of feature maps being output from the layer
    # Size of filter = kernel size (3, 3) = 3x3 pixel sections
    # Padding = 'same' means padding will be added in a way that ensures we can use all pixels
    model.add(Conv2D(filters = hp.Int('Input_Conv_Filters', min_value = 32, max_value = 256, step = 32), 
                     kernel_size = (3, 3), 
                     padding = 'same', 
                     input_shape = (img_width, img_height, num_channels)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    
    # Test 1-3 convolutional layers
    for i in range(hp.Int('n_Conv_Layers', min_value = 1, max_value = 3, step = 1)):
        model.add(Conv2D(filters = hp.Int(f'Conv_{i}_Filters', min_value = 32, max_value = 256, step = 32), 
                         kernel_size = (3, 3), 
                         padding = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D())
    
    model.add(Flatten())
    
    # Test 1-4 dense layers
    for i in range(hp.Int('n_Dense_Layers', min_value = 1, max_value = 4, step = 1)):
        model.add(Dense(units = hp.Int(f'Dense_{i}_Neurons', min_value = 32, max_value = 256, step = 32),
                        activation = 'relu'))
        
        if hp.Boolean(f'Dense_{i}_Dropout'):
            model.add(Dropout(rate = 0.5))
    
    model.add(Dense(units = num_classes, activation = 'softmax'))
    
    # Compile network
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = hp.Choice('Optimizer', values = ['adam', 'RMSProp']),
                  metrics = ['accuracy'])
    
    return model

```

<br>

The code below sets the following parameters for the tuner search:

* **`hypermodel`** --- function with the logic for building a network to test, in this case the **`build_model`** function defined above
* **`objective`** --- metric to optimize (in this case, accuracy is being optimized)
* **`max_trials`** --- maximum number of random network configurations to test
* **`executions_per_trial`** --- or the number of times to run each tested configuration (the results will be averaged)
* **`directory`**, **`project_name`**, and **`overwrite`** --- parameters related to the logging of the trial results

```python

# search parameters
tuner = RandomSearch(hypermodel = build_model,
                     objective = 'val_accuracy',
                     max_trials = 30,
                     executions_per_trial = 2,
                     directory = os.path.normpath('C:/'),
                     project_name = 'fruit-cnn',
                     overwrite = True)

```

<br>

With the tuner search parameters defined, the next lines of code below execute the search trials with the training and validation sets, with a specific number of **`epochs`** for each tested configuration and an in-epoch **`batch_size`**.

```python

# execute search
tuner.search(x = training_set,
             validation_data = validation_set,
             epochs = 30,
             batch_size = 32)

```

<br>

Depending on the number of configurations being tested, the number of epochs for each configuration, and processing speed, this can take a very long time. (Running the above on a laptop with no customizations took around 9-10 hours.) But it is almost certain to result in improved accuracy with more tests and epochs, up to a point of diminishing returns where the models are already so highly accurate that it is not worth the time.

<br>

#### Updated Network Architecture

The result of the above tuner search found that the best network architecture in terms of validation accuracy (97%) was one with **1 Convolutional layer**. The input layer has **192 filters**, and the convolutional layer has **128 filters**, along with a Max Pooling layer which was not tested with any changes. The network then has **1 Dense (Fully Connected) layer** after flattening with **64 neurons** with **Dropout applied**, followed by the output layer. The chosen optimizer was **`RMSProp`**.

The below code builds this network architecture.

```python

# Network architecture
model = Sequential()

# Input layer
model.add(Conv2D(filters = 192, kernel_size = (3, 3), padding = 'same', input_shape = (img_width, img_height, num_channels)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Convolutional layer
model.add(Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())

# Dense layer with dropout
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'RMSProp',
              metrics = ['accuracy'])

```

<br>

#### Training the Updated Network

The same code from the baseline network is run to train this updated network with 50 epochs, the only change being a modified filename for the tuned network to keep all the network files separate for comparison.

<br>

#### Analysis of Training Results

The below image shows the **Loss** and **Classification Accuracy** plots for the tuned network for both the training set (blue) and the validation set (orange) for each epoch.

![CNN Tuned Accuracy Plot](/img/posts/cnn-tuned-accuracy-plot.png "CNN Tuned Accuracy Plot")

<br>

There is a peak Classification Accuracy for the validation set of around **97%**, which is equal to the **97%** for the non-tuned network with Image Augmentation. The tuned network applies Dropout and Image Augmentation, and again overfitting is eliminated. Interestingly, the validation set seems to even achieve (mostly) a slightly higher accuracy than the training set after a certain number of epochs.

<br>

#### Performance on the Test Set

As was done for the baseline network, to check performance on the test set the same code from above will be run. The only change is to load in the network file for the updated network rather than the baseline network.

<br>

#### Test Set Classification Accuracy

The tuned network, with both Dropout and Image Augmentation in place, scored **95%** on the Test Set, again higher than the baseline network and the network with only Dropout applied, but approximately equal to the non-tuned network with Image Augmentation in place.

<br>

#### Test Set Confusion Matrix

As mentioned previously, overall Classification Accuracy is useful, but sometimes does not tell us all that much about what is really going on with the network's predictions.

95% accuracy on the test set probably means there isn't much to worry about here, but running the same code from the baseline section on results for the updated, tuned network gives the following Confusion Matrix:

```

actual     apple  avocado  banana  kiwi  lemon  orange
predicted                                             
apple        0.9      0.0     0.0   0.0    0.0     0.0
avocado      0.0      1.0     0.0   0.1    0.0     0.0
banana       0.0      0.0     1.0   0.0    0.0     0.0
kiwi         0.1      0.0     0.0   0.8    0.0     0.0
lemon        0.0      0.0     0.0   0.0    1.0     0.0
orange       0.0      0.0     0.0   0.1    0.0     1.0

```

<br>

The column labels are the *actual* classes, and the row labels are the *predicted* classes. We can look down each column to find the Classification Accuracy for each class and see where the network is misclassifying.

While the overall test set accuracy was 95%, the following are the accuracy percentages for each individual type of fruit:

* Apple: 90%
* Avocado: 100%
* Banana: 100%
* Kiwi: 80%
* Lemon: 100%
* Orange: 100%

This is the exact same breakdown as the non-tuned network with Image Augmentation. There wasn't much change or gain after tuning in this particular case and with the specific number of tests and epochs that were run to tune the network.

___

# Transfer Learning With VGG16 <a name="cnn-transfer-learning"></a>

#### Transfer Learning Overview

**Transfer Learning** is a powerful way to use pre-built, pre-trained networks and apply them to solve specific Deep Learning tasks. It involves leveraging features learned on one problem for a new, similar problem.

For image-based tasks this often means using all the the *pre-learned* features from a large network, including all convolutional filter values and feature maps, and then training just the last part for the new task at hand.

For a well-trained, well-established network, the features that have already been learned will be good enough to differentiate between new image classes, so using a pre-optimized network will save a lot of training time.

For this task I use a well-known network called **VGG16**. It was designed in 2014 and trained on over a million images across a wide variety of images (e.g, goldfish, bottles of wine, and scuba divers).

<br>

#### Updated Data Pipeline

The data pipeline will remain mostly the same as it was when applying the above custom built networks, with some small changes. The code below will import VGG16 and the custom preprocessing logic that it uses. The images will also be sent in with 224 x 224 pixels, as this is what VGG16 takes. The rest of the logic remains the same.

```python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# VGG preprocessing instead of 1./255. converts images from RGB to BGR

# Data flow parameters
training_data_dir = 'data/training'
validation_data_dir = 'data/validation'
batch_size = 32
img_width = 224
img_height = 224
num_channels = 3
num_classes = 6

# Image generators
training_generator = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 20,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        zoom_range = 0.1,
                                        horizontal_flip = True,
                                        brightness_range = (0.5, 1.5),
                                        fill_mode = 'nearest')

validation_generator = ImageDataGenerator(preprocessing_function = preprocess_input)

# Image flows
training_set = training_generator.flow_from_directory(directory = training_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')
validation_set = validation_generator.flow_from_directory(directory = validation_data_dir,
                                                      target_size = (img_width, img_height),
                                                      batch_size = batch_size,
                                                      class_mode = 'categorical')

```

<br>

#### Network Architecture

To build the Transfer Learning network in Keras, the code below will download the "bottom" of the VGG16 network (everything up to the Dense layers), then add the "top" of the model as it applies to this problem of fruit classification.

The code specifies not to retrain the imported layers from VGG16, as their parameter values should be frozen. Then two Dense layers with 128 neurons each are added in, followed by the output layer.

```python

# Architecture
vgg = VGG16(input_shape = (img_width, img_height, num_channels),
            include_top = False)                                    # This arg excludes the dense layers and output layer

# Freeze all layers (they won't be updated during training)
for layer in vgg.layers:
    layer.trainable = False

flatten = Flatten()(vgg.output)

dense1 = Dense(units = 128, activation = 'relu')(flatten)
dense2 = Dense(units = 128, activation = 'relu')(dense1)

output = Dense(num_classes, activation = 'softmax')(dense2)

model = Model(inputs = vgg.inputs, outputs = output)

# Compile network
model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

# View architecture
model.summary()

```

<br>

The final architecture is as shown below:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               3211392   
_________________________________________________________________
dense_5 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_6 (Dense)              (None, 6)                 774       
=================================================================
Total params: 17,943,366
Trainable params: 3,228,678
Non-trainable params: 14,714,688
_________________________________________________________________

```

<br>

The architecture has a total of 17.9 million parameters, which is much bigger than the previous networks above. Of the 17.9 million parameters, 14.7 million parameters are frozen, and 3.2 million parameters will be updated during each iteration of Backpropagation.

<br>

#### Training the Network

The same code is run to train this updated network as for the baseline network, with only 15 epochs to begin with, as it is far more computationally expensive to train due to the large number of trainable parameters. Once again, the filename is also modified for the saved Transfer Learning network to keep all the network files separate for comparison.

<br>

#### Analysis of Training Results


The below image shows the **Loss** and **Classification Accuracy** plots for the Transfer Learning network for both the training set (blue) and the validation set (orange) for each epoch.

![VGG16 Accuracy Plot](/img/posts/cnn-vgg16-accuracy-plot.png "VGG16 Accuracy Plot")

<br>

There is a peak Classification Accuracy for the validation set of approximately **99%**, which is the highest of all the networks considered in this project. It is especially impressive that this accuracy was achieved after training for only 15 epochs.

<br>

#### Performance on the Test Set

As was done for the baseline network, to check performance on the test set the same code from above will be run. The only change is to load in the network file for the Transfer Learning network rather than the baseline network.

<br>

#### Test Set Classification Accuracy

The VGG16 Transfer Learning network scored **93%** accuracy on the test set, higher than that of the best custom networks above.

<br>

#### Test Set Confusion Matrix

As mentioned previously, overall Classification Accuracy is useful, but sometimes does not tell us all that much about what is really going on with the network's predictions.

93% accuracy on the test set means there are very few errors to worry about, but to compare, after running the same code from the baseline section on results for the Transfer Learning VGG16 network, the following is the Confusion Matrix:

```

actual     apple  avocado  banana  kiwi  lemon  orange
predicted                                             
apple        1.0      0.0     0.0   0.0    0.0     0.0
avocado      0.0      1.0     0.0   0.0    0.0     0.0
banana       0.0      0.0     1.0   0.0    0.0     0.0
kiwi         0.0      0.0     0.0   0.9    0.0     0.0
lemon        0.0      0.0     0.0   0.0    0.8     0.1
orange       0.0      0.0     0.0   0.1    0.2     0.9

```

<br>

The column labels are the *actual* classes, and the row labels are the *predicted* classes. We can look down each column to find the Classification Accuracy for each class and see where the network is misclassifying.

The following are the accuracy percentages for each individual type of fruit:

* Apple: 100%
* Avocado: 100%
* Banana: 100%
* Kiwi: 90%
* Lemon: 80%
* Orange: 90%

The above classes are being predicted about as well as the best custom networks previously built above, with a network that has been trained for only 15 epochs.

___

# Results Discussion <a name="cnn-results"></a>

The baseline network had accuracy problems due to overfitting, but the addition of both Dropout and Image Augmentation eliminated overfitting almost entirely.

The test set classication accuracy percentages for each network architecture were:

* Baseline Network: **80%**
* Baseline + Dropout: **92%**
* Baseline + Image Augmentation: **95%**
* Optimized (Tuned) with Dropout + Image Augmentation: **95%**
* Transfer Learning using VGG16: **93%**

Tuning the network's architecture with Keras Tuner improved the performance over the baseline, but was very time intensive. That time investment could be worth it if it resulted in improved accuracy, but that was not the case in this particular instance, as the baseline model with image augmentation added had the same accuracy on test set.

Using Transfer Learning with the VGG16 architecture was quite successful. In 15 epochs it achieved accuracy similar to the smaller, custom networks that all trained 50 epochs. From a business point of view, there is overhead in storing a very large VGG16 network file and in possible increased latency on inference, so there are tradeoffs to consider when choosing a CNN architecture.

In this case, since Transfer Learning is not more accurate, the Baseline network with image augmentation seems sufficient for this task (so far).

___

# Growth and Next Steps <a name="growth-next-steps"></a>

This project had very accurate predictions for images, although on a small number of classes. The best networks above could be tested on a larger number of image classes to gain more insight into the different networks' performance and to see whether they still perform well.

The Transfer Learning network performed a little worse than the Baseline + Image Augmentation or the Tuned with Dropout + Image Augmentation networks in terms of classification accuracy on the test set, but it was only trained for 15 epochs, which is small. It could be trained for more epochs, and it would also be interesting to try transfer learning using other available pre-trained networks such as the ResNet, Inception, and DenseNet networks.
