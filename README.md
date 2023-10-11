# The-Detection-of-Dermatological-Disorders-Through-Image-Analysis

**Name:** Vanessa-Ramona Marin  
**Institution:** University POLITEHNICA of Bucharest, Faculty of Automatic Control and Computer Science  
**Generation:** 2019-2023  

This repository contains all the code and data related to my BSc Thesis. Its primary goal is to develop a reliable system that can automatically identify and categorize dermatological disorders based on visual characteristics present in images.

**Structure**
-

1. Code structure

SkinDetector
------------------------|     
 │   ├── dataset.py      |           
 │   ├── procces_one_image.py    |            
 │   ├── skin_detector.py |
 │   ├── train.py          |       
 │   ├── skin-detector          |      
 
 └── README.md     

 * **skin-detector** -> folder contains the frontend part of the project. The final user interface of the project is presented at the end of this README.md file.


**Dataset**
-

[Get Dataset] (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

The dataset used for this project contains 7 dermatological disorders:


Dermatological disorders | Label | 
------------ | ------------- | 
Actinic keratoses and intraepithelial carcinoma / Bowen's disease | akiec | 
Basal cell carcinoma | bcc |
Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) | bkl | 
Dermatofibroma | df | 
Melanoma | mel | 
Melanocytic nevi | nv | 
Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)  | vasc | 

**Methodology**
-

I developed a new model for dermatological disorder detection using transfer learning with a ResNet50V2 model. I froze the starting layers and added more layers to create a finely-tuned, accurate solution.


 ![Architecture of the model](https://github.com/marinvanessa/The-Detection-of-Dermatological-Disorders-Through-Image-Analysis/assets/127364101/873c8caf-679b-4a17-8321-e9e37b52342b)

In the training loop, the code iterates through the training dataset, utilizing the Gelu activation function to introduce non-linearity to specific convolutional and dense layers. This enhances the model's ability to capture intricate patterns and features in the image data during both training and evaluation. The model's weights are updated using the RMSprop optimizer, and it keeps track of binary cross-entropy loss and accuracy. Additionally, every 100 training steps, the model's performance is assessed on a validation dataset, where validation loss and accuracy are computed and logged.

**Results**
-

Learning Rate | Batch Size | Loss| Accuracy
------------ | ------------- | ------------- | ------------- | 
0.005 | 32 | 1.3002 | 91.56%
0.005| 64 | 1.3448 | 91.34%
0.002 | 32 | 1.2830 | 91.70%
0.002 | 64 | 1.2506 | 91.92%
0.001 | 32 | 1.2308 | 91.96%
0.001 | 64 | 1.1538 | 92.48%


**User Interface**
-

The React-based interface features two main pages: one for image upload and accepting terms, and another for displaying the detected disorder result.

![User Interface](https://github.com/marinvanessa/The-Detection-of-Dermatological-Disorders-Through-Image-Analysis/assets/127364101/53614dd1-8c87-4e35-a3eb-b1c2b3834628)










