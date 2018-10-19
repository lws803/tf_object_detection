import cv2
import numpy as np
import math
import io
import sys
import os
try:
    import tensorflow as tf
except ImportError:
    print("unable to import TensorFlow. Is it installed?")
    print("  sudo apt install python-pip")
    print("  sudo pip install tensorflow")
    sys.exit(1)

# Object detection module imports
import object_detection
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# SET FRACTION OF GPU YOU WANT TO USE HERE
GPU_FRACTION = 0.4

######### Set model here ############
MODEL_NAME =  'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
# By default models are stored in data/models/
MODEL_PATH = os.path.join(os.path.dirname(sys.path[0]),'data','models' , MODEL_NAME)
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_PATH + '/frozen_inference_graph.pb'
######### Set the label map file here ###########
LABEL_NAME = 'mscoco_label_map.pbtxt'
# By default label maps are stored in data/labels/
PATH_TO_LABELS = os.path.join(os.path.dirname(sys.path[0]),'data','labels', LABEL_NAME)
######### Set the number of classes here #########
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`,
# we know that this corresponds to `airplane`.  Here we use internal utility functions,
# but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Setting the GPU options to use fraction of gpu that has been set
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION




class Pipeline:
    def __init__ (self, img):
        self.image = img
        self.sess = tf.Session(graph=detection_graph,config=config)



    def preprocess(self):
        # # Chaining the preprocessors
        # self.img = cv2.GaussianBlur(self.img,(5,5),0)
        # self.img = norm_illum_color(self.img, 0.8)
        pass

    def process (self):
        objArray = []
        # objArray = Detection2DArray()

        image=cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = np.asarray(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = self.sess.run([boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        objects=vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=1)

        object_count=1
        
        print (objects)

        for i in range(len(objects)):
            object_count+=1

        img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        return img


    def visualisation(self):

        img = self.process()
        return img


# main
if __name__ == '__main__':
    
    print (sys.argv[1])
    img = cv2.imread(sys.argv[1])

    while(True):
        frame = img

        # initialize
        frame_size = frame.shape
        frame_width  = frame_size[1]
        frame_height = frame_size[0]

        frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3) # Scale resizing

        my_pipeline = Pipeline(frame)
        visualisation = my_pipeline.visualisation()

        numpy_horizontal_concat = np.concatenate((frame, visualisation), 1)

        cv2.imshow('image', numpy_horizontal_concat)

        cv2.waitKey()
        # exit if the key "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
