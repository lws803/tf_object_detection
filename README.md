#Steps

1. Follow installation here https://github.com/tensorflow/models/blob/4f32535fe7040bb1e429ad0e3c948a492a89482d/research/object_detection/g3doc/installation.md to get object_detection ready. If you are facing issues with conflicting python versions, please set up a virtualenv for python2.7 

2. Download your images and store them in a folder 

3. Seperate your images into training, eval, test categories

4. Use labelimg tool to label the images and save them in an annotations folder

5. Use xml_to_csv.py to generate the csv file required for generating tf record 
```shell
function () { return "python xml_to_csv.py -in TRAINING_ANNOTATION_FOLDER_PATH -out train.csv"}
```
7. Edit generate_tfrecord.py to generate the tf record see line ~25
8. Use it to generate the tfrecord 
```shell
function () { return "python generate_tfrecord.py --input_csv=PATH_TO_CSV  --output_tfrecord=train.record"}
```
9. Repeat steps 3 to 9 for eval and test categories of your dataset

