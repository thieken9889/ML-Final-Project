# ML-Final-Project

Task: 
Train CNNs to predict whether someone has pneumonia or not based on an X-ray of their chest.

Training & Testing Data: 
We will be using training and testing data given by the dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia, this dataset contains several x-rays of the thoracic cavity. There are three labels in the dataset, “healthy”, “bacterial pneumonia” and “viral pneumonia”. We will simplify this to simply “healthy” or “pneumonia”. The directory is already segmented into 5,216 training images and 624 test images.

Method: 
Our main focus will be to use transfer learning techniques to compare several well known pre-trained CNN models, including:
Inception v3
DenseNet
SqueezeNet
AlexNet
The main challenge will be to determine the best parameters to use for the fully connected layers that will be retrained on the dataset.

A secondary goal would be to design our own CNN and train on the data set and compare its accuracy to a pre-trained networks mentioned above. The main concern with this would be that there is not enough data to train a deeper network. One way around this might be to try PCA to reduce the dimensionality of the data so a shallower network is required.

Metrics: 
Accuracy - To measure how accurate our pneumonia diagnoses are 
False Negative Rate - When diagnosing we want less false negatives
False Positive Rate - When diagnosing we are okay with more false positives

References:
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6093039/
http://cs231n.github.io/transfer-learning/ 
https://medium.com/datadriveninvestor/detecting-pneumonia-with-deep-learning-a-soft-introduction-to-convolutional-neural-networks-b3c6b6c23a88
