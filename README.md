# Face_Mask Classification

#To run Entire project:
    
         ./Main.sh

It would execute 3 steps :  
1. Downloading dataset, converting files to .png
2. Do preprocessing, data loading, and training. It saves the model weights file.
For setting hyperparameters during training, you can set in the model_config.yml file
3. Using gradieo we do real-time inference. 
 
**End!
 
# Three IMPORTANT FILES ARE:
     1. Dataset.py - Downloading Data and converting to .png
     2. Traning.py - Training Binary Classifier
     3. Inference.py - Real time inference using Gradeio 
     

Approach:      
1. As we are interested in the classification of Face Mask Vs No Mask, color information is not that valuable for us, hence we convert all images into grayscale
2. Using Dataloaders we load data.

3. CNN Architecture:
   - A FEATURE EXTRACTOR- 
      1. INPUT LAYER: CNN( 32, filter_size = 3,activation='relu',INPUT_SHPAE)
               We are more interested in the face region hence 3x3 size would be much more enough for us to extract information.
      2. Without changing anything we add the same layer, just to extract more information.
      3. Now here we increase filters to 64 after acquiring sufficient high-level features, we aim for some medium-level features. So we do this operation twice.
      4. Now we are not much interested in getting low-level features, hence only one CNN(128) would be sufficient.
      5. Lastly, we add 512 filters with regularisation  l1 and l2 just to control loss getting exploited.
      6. Also to avoid much overfitting we apply a 30% DROPOUT
   - B. LEARNING LAYER:
     1. To pass this extracted information to Fully Connected layers to make this array flatten.
     2. We apply 128 nodes of DNN
     3. Finally we classify it as SIGMOID (AS we have binary classification)
   
4. To optimize this network we use "binary_crossentropy" as we have two classes and accuracy and loss to monitor during training.


In this way, we achieved 98% Accuracy on the validation set.
We make different real-time test set and calculate confusion matrix, precision, and other data on it, which helps a decision to deploy that model or not


THANK YOU!
