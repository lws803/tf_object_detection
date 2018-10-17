# Tensorflow inference with OpenCV

## Contents
1. Installation
2. Running
3. Changing models and labels
4. Editing ```Pipeline``` class
5. Debugging ```object_detection``` undefined

### 1. Installation
1. Clone the repo
2. Install python dependencies
```bash
# From /inference
$ pip install -r requirements.txt
```
3. Install tf object detection library
```bash
# From /inference/src
$ protoc object_detection/protos/*.proto --python_out=.
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### 2. Running
```bash
# In /inference/scripts
$ python inference_realtime.py # For realtime detection via webcam
$ python inference_single.py PATH/TO/IMAGE # For single image detection
```

### 3. Changing/ Adding your model and labels
1. Head to /inference/data/models
2. Head to https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and download a model of your choice
3. Extract the model of your choice and leave it as ```/inference/data/model/ssdlite_mobilenet_v2_coco_2018_05_09/...``` *for example*
4. Edit MODEL_NAME in ```/inference/scripts/inference_realtime.py``` and change label map *(if needed)* for LABEL_NAME. Default model is ```ssdlite_mobilenet_v2_coco_2018_05_09``` with label = ```mscoco_label_map.pbtxt```


### 4. Editing Pipeline class
```python
def preprocess(self):
    # # Chaining the preprocessors
    # self.img = cv2.GaussianBlur(self.img,(5,5),0)
    # self.img = norm_illum_color(self.img, 0.8)
    pass
```
Use this method to add your pre-processors eg. GaussianBlur

```python
def process (self):
    img = self.image
    # Your code goes here
    
    return img
```
This is the main processor which you can extract the processed image and detected objects from

```python
def visualisation(self):
    img = self.process()
    return img
```
This method controls what gets output in the main window. By default it calls self.process() to visualise and initiate the process.


### 5. Debugging object_detection not found
```bash
# From /inference/src
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Perform the command above again before running inference
