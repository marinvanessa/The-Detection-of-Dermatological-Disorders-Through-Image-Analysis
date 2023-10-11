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

[Get Dataset] (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

The dataset used for this project contains 7 dermatological disorders. 


Dermatological disorders | Label | 
------------ | ------------- | 
Actinic keratoses and intraepithelial carcinoma / Bowen's disease | akiec | 
Basal cell carcinoma | bcc |
Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) | bkl | 
dermatofibroma | df | 
melanoma | mel | 
melanocytic nevi | nv | 
vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)  | vasc | 

