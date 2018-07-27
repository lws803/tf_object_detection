## Steps

1. Follow installation here https://github.com/tensorflow/models/blob/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/g3doc/installation.md to get object_detection ready. If you are facing issues with conflicting python versions, please set up a virtualenv for python2.7 

2. Download your images and store them in a folder 

3. Seperate your images into training, eval, test categories

4. Use labelimg tool to label the images and save them in an annotations folder

5. Use xml_to_csv.py to generate the csv file required for generating tf record 
```shell
python xml_to_csv.py -in TRAINING_ANNOTATION_FOLDER_PATH -out train.csv
```
7. Edit generate_tfrecord.py to generate the tf record see line ~25
8. Use it to generate the tfrecord 
```shell
python generate_tfrecord.py --input_csv=PATH_TO_CSV  --output_tfrecord=train.record
```
9. Repeat steps 3 to 9 for eval and test categories of your dataset

## Misc 
To download images with the included script:  
```shell
python google_images_download.py  -k OBJECT_KEYWORD -l NUMBER_OF_IMAGES -f jpg -s medium -o OUTPUT_FOLDER
```
-f : file extension  
-k : keyword for the image  
-l : number of images  

## Moving forward
To run the training locally: https://github.com/tensorflow/models/blob/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/g3doc/running_locally.md  

models can be found in tensorflow model zoo: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md  

model pipeline configs can be found in:  https://github.com/tensorflow/models/tree/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/samples/configs


## Some helpful resources

1. Understanding different kinds of models for detection: https://medium.com/@jonathan_hui/what-do-we-learn-from-region-based-object-detectors-faster-r-cnn-r-fcn-fpn-7e354377a7c9
2. Types of optimizers: http://ruder.io/optimizing-gradient-descent/
3. SSD(Single shot detectors) paper: https://arxiv.org/abs/1512.02325
4. Understanding mAPs (mean average precision): https://stackoverflow.com/questions/46094282/why-we-use-map-score-for-evaluate-object-detectors-in-deep-learning