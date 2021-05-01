# Automatic_raster_classifier

Semantic segmentation is one of the fundamental research in remote sensing image processing. Because of the complex environment, a fully unsupervised segmentation and classification is a challenging task.  Although the neural network has achieved excellent performance in semantic segmentation in the previous years, But due to extremely large size of geological image and complexity and interference in multiple bands it is quite difficult to perform deep learning based segmentation and classification because of the current hardware limitations. This project aims to solve that very problem by down sampling the image band space by K-means clustering to get pixel level classification 


## Requirements
* GDAL
* OpenCV
* QGIS


  
## Insatallation
* Install QGIS
* Open python console in QGIS and run `import pip` and `pip.main(['install', ' opencv-python'])` to install all the prerequisites.
* Search `Automatic Classifier` in Plugin manager and install.
* To execute the program fo to Raster and Select `auto_class`.

![image](https://user-images.githubusercontent.com/25346465/116771364-7e774b00-aa68-11eb-9243-eafe20073c29.png)
![image](https://user-images.githubusercontent.com/25346465/116771369-846d2c00-aa68-11eb-8b78-6d3e300358c5.png)
