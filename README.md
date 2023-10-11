# The-Detection-of-Dermatological-Disorders-Through-Image-Analysis

This is my thesis project, and it aims to develop a robust system for automatically recognizing and categorizing dermatological disorders based on visual characteristics captured in images.

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

 * **skin-detector** folder contains the frontend part of the project.


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






