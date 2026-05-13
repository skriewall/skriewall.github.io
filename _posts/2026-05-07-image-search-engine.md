---
layout: post
title: Creating An Image Search Engine Using Deep Learning
image: "/posts/dl-search-engine-title-img.png"
tags: [Deep Learning, CNN, Data Science, Computer Vision, Python, TensorFlow/Keras]
---

In this project I build an Image Search Engine built on Deep Learning that could have many applications, one being to help an online business' customers find similar products to ones they want.

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Discussion and Next Steps](#overview-growth)
- [01. Sample Data Overview](#sample-data-overview)
- [02. Transfer Learning Overview](#transfer-learning-overview)
- [03. Setting Up VGG16](#vgg16-setup)
- [04. Image Preprocessing and Featurization](#image-preprocessing)
- [05. Execute Search](#execute-search)
- [06. Discussion and Next Steps](#growth-next-steps)

___

# Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Suppose we had a client that has been getting feedback from their customers that it is difficult to find the products they are looking for on the website. The customers are often buying more expensive products, and then later finding out that the client actually stocks a very similar, lower-priced alternative. This project will show how deep learning can help.

<br>

### Actions <a name="overview-actions"></a>

Here I use the pre-trained VGG16 network. Instead of the final Max Pooling layer, I add in a **Global Average Pooling layer** at the end of the VGG16 architecture meaning the output of the network will be a single vector of numeric information rather than many arrays. I use "feature vectors" to compare image similarity.

I preprocess the 300 images in the base set and then pass them through the network to extract their feature vectors. I store the feature vectors in an object which will be used when a search image is passed in.

I pass in a search image, apply the same preprocessing steps, and extract the search image's feature vector. I use Cosine Similarity to compare the input feature vector with all feature vectors from the base set, and return the N smallest values. These represent the most similar images, i.e., the most similar products that would be shown to an online customer.

<br>

### Results <a name="overview-results"></a>

I test two different images and plot the search results along with the cosine similarity scores. These are shown in the dedicated section below.

<br>

### Discussion and Next Steps <a name="overview-growth"></a>

This project is a rough proof of concept. In reality the code would not all be in a single script, and the search functionality would be isolated from the rest of the code. Also, instead of fitting the Nearest Neighbors each time a search is submitted, that would ideally be stored in an object as well.

If something like this search engine were to go to production, additional functionality for adding or removing images from the stored images would be important, as products can be added and removed from a website, or some items could go out of stock and then come back in stock. Also, in production it would likely just return a list of the filepaths that could be pulled by the website (no plotting of images in matplotlib).

The search engine was tested only in one category (women's shoes), but it would be important to test on a broader array of categories and possibly have a separate saved network for each category to prevent irrelevant predictions.

Only Cosine Similarity was used to compare the input search image features to the features of the base set images. Other distance metrics could be used instead and could be compared to see if there are any differences in which images are identified as similar to the search image.

Currently the quality of the search results is not quantified in this project. We can see visually if the search results look similar to the search image, but this is somewhat subjective. A couple ways to quantify the quality could be from customer feedback on a survey or from click-through rates on the website.

It could be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks as well.

<br>

___

# Sample Data Overview  <a name="sample-data-overview"></a>

For this proof of concept I am working with images of women's shoes. Suppose these 300 shoes are currently available to purchase on a client's website. A random selection of 18 shoe images can be seen in the image below.

![Deep Learning Search Engine - Image Examples](/img/posts/search-engine-image-examples.png "Deep Learning Search Engine - Image Examples")

Using deep learning, I will extract the "features" of this base image set and compare them to the "features" in any given search image. The prdouct images with the closest match would be shown to the customer.

___

# Transfer Learning Overview  <a name="transfer-learning-overview"></a>

### Overview


**Transfer Learning** is a powerful way to use pre-built, pre-trained networks and apply them to solve specific Deep Learning tasks. It involves leveraging features learned on one problem for a new, similar problem.

For image-based tasks this often means using all the the *pre-learned* features from a large network, including all convolutional filter values and feature maps, and then training just the last part for the new task at hand.

For a well-trained, well-established network, the features that have already been learned will be good enough to differentiate between new image classes, so using a pre-optimized network will save a lot of training time.

For this task I use a well-known network called **VGG16**. It was designed in 2014 and trained on over a million images across a wide variety of images (e.g, goldfish, bottles of wine, and scuba divers).

![VGG16 Architecture](/img/posts/vgg16-architecture.png "VGG16 Architecture")

<br>

### Application

When using Transfer Learning for image classification tasks, it is common to import the architecture up to the final Max Pooling layer, prior to flattening, the Dense layers, and the Output layer. The frozen parameter values from the bottom of the network are used and instead of keeping the final Max Pooling layer, it is replaced with a Global Average Pooling layer.

With that approach, the final Max Pooling layer will be in the form of a number of pooled feature maps. For this task, however, I want a single set of numbers to represent these features. thus we add in a **Global Average Pooling layer** at the end of the VGG16 architecture meaning the output of the network will be a single array of numeric information rather than many arrays.

___

# Setting Up VGG16  <a name="vgg16-setup"></a>

Using **Keras** from the TensorFlow library in Python, I import the bottom of the VGG16 network (everything up to the Dense layers) and then add a parameter to ensure that the final layer is not a Max Pooling layer but a **Global Max Pooling layer**.

The code below does the following:

* Imports the required Python packages
* Sets up the image parameters required for VGG16
* Loads in VGG16 with Global Average Pooling
* Saves the network architecture and weights for use in the search engine

```python
# Import required packages
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle

# VGG16 image parameters
img_width = 224
img_height = 224
num_channels = 3

# Load in VGG16 network architecture using global pooling
vgg = VGG16(input_shape = (img_width, img_height, num_channels),
            include_top = False,
            pooling = 'avg')
model = Model(inputs = vgg.input, outputs = vgg.layers[-1].output)

# Save the model file
model.save('models/vgg16_search_engine.h5')
```

The architecture is summarized below:

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
global_average_pooling2d (Gl (None, 512)               0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
_________________________________________________________________
```

If the last parameter of **`pooling = avg`** was not there, then the final layer would have been the Max Pooling layer of shape 7 by 7 by 512. Instead, since the parameter was there, it ends up with a single array of size 512. Basically, all of the feature maps from the final Max Pooling layer are summarized into one vector of 512 numbers, and these numbers will represent each image's features. This feature vector is what will be used to compare the base set of images to any given search image to find similarities.

___

# Image Preprocessing and Featurization <a name="image-preprocessing"></a>

### Helper Functions

The following code creates two functions --- the first for preprocessing images prior to entering the network, and the second for "featurizing" the image, i.e., passing the image through the VGG16 network and outputting a single vector of 512 numeric values.

```python
# Image preprocessing function
def preprocess_image(filepath):
    image = load_img(filepath, target_size = (img_width, img_height))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    return image

# Featurize image function
def featurize_image(image):
    feature_vector = model.predict(image)
    return feature_vector
```

The **`preprocess_image`** function does the following:

* Receives the filepath of an image and loads the image
* Turns the image into an array
* Adds in the "batch" dimension for the array that Keras is expecting
* Applies the preprocessing logic for VGG16 that was imported from Keras
* Returns the image as an array

The **`featurize_image`** function does the following:

* Receives the image as an array
* Passes the array through the VGG16 architecture
* Returns the feature vector

<br>

### Setup

The code below:

* Specifies the directory **`source_dir`** of the base set of images
* Sets up an empty list **`filename_store`** to append the image filenames for easy lookup later on
* Sets up an empty array **`feature_vector_store`** to append the base set feature vectors

```python
# Source directory for images
source_dir = 'data/'

# Empty objects to append to
filename_store = []
feature_vector_store = np.empty((0, 512))
```

<br>

### Preprocess and Featurize Base Set Images

The following code preprocesses and featurizes all 300 images in the base set by executing a loop and applying the two functions created above. For each image, it appends the filename and the feature vector. It saves these stores which will be used when a search is executed.

```python
# Featurize the base image set
for image in listdir(source_dir):
    print(image)
    
    # Append image filename for future lookup
    filename_store.append(source_dir + image)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(source_dir + image)
    
    # Get the feature vector
    feature_vector = featurize_image(preprocessed_image)
    
    # Append feature vector to list for similarity calculations
    feature_vector_store = np.append(feature_vector_store, feature_vector, axis = 0)

# Save key objects for future use
pickle.dump(filename_store, open('models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store, open('models/feature_vector_store.p', 'wb'))
```
___

# Execute Search <a name="execute-search"></a>

### Setup

With the base set featurized, the code will now run a search on a new image. The code below does the following:

* Loads in the save network model
* Loads in the stored filenames and feature vectors
* Specifies a search image file
* Specifies the number of search results to find (these will be the n most similar results to the search image)

```python
# Load in required objects
model = load_model('models/vgg16_search_engine.h5', compile = False)
filename_store = pickle.load(open('models/filename_store.p', 'rb'))
feature_vector_store = pickle.load(open('models/feature_vector_store.p', 'rb'))

# Search parameters
search_results_n = 8
search_image = 'search_image_02.jpg'
```

The search image being tested is below:

![VGG16 Architecture](/img/posts/search-engine-search1.jpg "VGG16 Architecture")

<br>

### Preprocess and Featurize Search Image

Using the same helper functions from above, the following code preprocesses and featurizes the search image. The output is a vector containing 512 numeric values.

```python
# Preprocess and featurize search image
image = preprocess_image(search_image)
search_feature_vector = featurize_image(image)
```

<br>

### Locate Most Similar Images Using Cosine Similarity

Now that the search image exists as a 512 length feature vector, that feature vector needs to be compared to the feature vectors of all the base images. We need a way to understand which eight base image feature vectors are most like the feature vector of the search image. Therefore, I use the **`NearestNeighbors`** class from **scikit-learn** and apply the **Cosine Distance** metric to calculate the angle of difference between the feature vectors.

**Cosine Distance** essentially measures the angle between any two vectors and checks whether the two vectors are pointing in a similar direction. The more similar the direction the vectors are pointing, the smaller the angle between them in space; the more different the direction, the larger the angle between them in space. The angle gives the Cosine Distance score.

After calculating the score between the search image vector and each of the base image vectors, I return the images with the eight lowest cosine scores --- i.e., the eight most similar images in terms of the feature vector representation coming from the network.

The code below:

* Instantiates the Nearest Neighbors logic and specifies the distance metric as Cosine Similarity
* Applies this to the **`feature_vector_store`** object
* Passes the **`search_feature_vector`** object into the fitted Nearest Neighbors object. This will find the eight nearest base feature vectors, and for each it will return (a) the cosine distance, and (b) the index of that feature vector in the *feature_vector_store* object.
* Converts the outputs from arrays to lists (to make it easier to plot the results)
* Creates a list of filenames for the eight most similar images

```python
# Instantiate nearest neighbors logic
image_neighbors = NearestNeighbors(n_neighbors = search_results_n,
                                   metric = 'cosine')       # get angle between vectors

# Apply to the feature vector store
image_neighbors.fit(feature_vector_store)

# Return search results for search image (distances & indices)
image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)

# Convert the closest image indices and distances to lists
image_indices = list(image_indices[0])
image_distances = list(image_distances[0])

# Get a list of filenames for search results
search_result_files = [filename_store[i] for i in image_indices]
```

<br>

### Plot Search Results

Given the eight most similar images to the search image, the following code plots them in order from most to least similar and includes the cosine distance score for reference (smaller is more similar).

```python
# Plot the results
plt.figure(figsize=(20,15))
for counter, result_file in enumerate(search_result_files):    
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1)
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3), fontsize=28)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

The search image and search results are below:

**Search Image** <br>
![Search 1: Search Image](/img/posts/search-engine-search1.jpg "Search 1: Search Image")

**Search Results** <br>
![Search 1: Search Results](/img/posts/search-engine-search1-results.png "Search 1: Search Results")

Out of the 300 images in the base set, these are the eight that have been found to be *most similar*, and they indeed look quite similar to the search image.

Checking a second search image to see if it also looks good gives the following:

**Search Image** <br>
![Search 2: Search Image](/img/posts/search-engine-search2.jpg "Search 2: Search Image")

**Search Results** <br>
![Search 2: Search Results](/img/posts/search-engine-search2-results.png "Search 2: Search Results")

These have turned out well, so the features from VGG16 combined with Cosine Similarity have done well with this problem.

___

# Discussion and Next Steps <a name="growth-next-steps"></a>

This project is a rough proof of concept. In reality the code would not all be in a single script as above, and the search functionality would be isolated from the rest of the code. Also, instead of fitting the Nearest Neighbors each time a search is submitted, that would ideally be stored in an object as well.

If something like this search engine were to go to production, additional functionality for adding or removing images from the stored images would be important, as products can be added and removed from a website, or some items could go out of stock and then come back in stock. Also, in production it would likely just return a list of the filepaths that could be pulled by the website (no plotting of images in matplotlib).

The search engine was tested only in one category (women's shoes), but it would be important to test on a broader array of categories and possibly have a separate saved network for each category to prevent irrelevant predictions.

Only Cosine Similarity was used to compare the input search image features to the features of the base set images. Other distance metrics could be used instead and could be compared to see if there are any differences in which images are identified as similar to the search image.

Currently the quality of the search results is not quantified in this project. We can see visually if the search results look similar to the search image, but this is somewhat subjective. A couple ways to quantify the quality could be from customer feedback on a survey or from click-through rates on the website.

It could be worthwhile testing other available pre-trained networks such as ResNet, Inception, and the DenseNet networks as well.
