#coding:utf-8

import os
import tensorflow as tf
import numpy as np
from PIL import Image , ImageFont , ImageDraw
import colorsys

from model import yolo_eval
from utils import letterbox_image
import random
import cv2

class YOLO(object):
    def __init__(self):
        self.model_path = 'yolo.pb'
        self.anchors_path = 'model_data/tiny_yolo_anchors.txt'
        self.classes_path = 'model_data/voc.names'

        self.score = 0.3
        self.iou = 0.45
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.model_image_size = (416 , 416)
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.output_graph_def = tf.GraphDef()
            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            with open(self.model_path , 'rb') as f:
                self.output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(self.output_graph_def , name="")
            self.boxes , self.scores , self.classes = self.generate()



    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors == 6  # default setting

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        # self.input_image_shape = K.placeholder(shape=(2, ))
        self.input_image_shape = tf.placeholder(dtype=tf.float32, shape=(2,))


        output = [self.graph.get_tensor_by_name("conv2d_10/BiasAdd:0"),
                  self.graph.get_tensor_by_name("conv2d_13/BiasAdd:0")]
        boxes, scores, classes = yolo_eval(output, self.anchors, len(self.class_names), self.input_image_shape,
                                           score_threshold=self.score, iou_threshold=self.iou)

        return boxes, scores, classes

    def detect_image(self, image):
        print('get detect_image--------------------')
        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        
        model_input = self.graph.get_tensor_by_name("input_1:0")
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                model_input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]]
            })
        
        results = []
        for i , c in reversed(list(enumerate(out_classes))):
            prdicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            result = []
            result.append(prdicted_class)
            result.append(score)
            
            box[1] = max(0 , np.floor(box[1] + 0.5).astype('int32'))
            box[0] = max(0 , np.floor(box[0] + 0.5).astype('int32'))
            box[3] = min(image.size[0] , np.floor(box[3] + 0.5).astype('int32'))
            box[2] = min(image.size[1] , np.floor(box[2] + 0.5).astype('int32'))
            
            result.append(box)

            results.append(result)
        return results

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def draw_rect(img, rst):
    if len(rst) > 0:
        cls , conf , x0 , y0 , x1 , y1 = rst[0] , rst[1] , int(rst[2][1]) , int(rst[2][0]) , int(rst[2][3]) , int(rst[2][2])
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, str(int(cls)), (int(x0 + (x1 - x0) * 0.5 - 10 ), int(y0 + (y1 - y0) / 2)), cv2.FONT_HERSHEY_SIMPLEX,
		            0.6, (0, 0, 255), 2)
        cv2.putText(img , str(conf) , (x0 , y0 + 20) , cv2.FONT_HERSHEY_SIMPLEX , 0.5 , (0 , 255 , 0) , 1)


def detect_img(yolo):
    
    img_path = "/home/han/tensorflow_read_pb/1.png"
    
    img = Image.open(img_path)
    results = yolo.detect_image(img)
    img_cv = cv2.cvtColor(np.asarray(img) , cv2.COLOR_RGB2BGR)
    for result in results:
        print('result:' , result)
        draw_rect(img_cv , result)

    cv2.imshow("img" , img_cv)
    cv2.waitKey(0)

    yolo.close_session()


if __name__ == '__main__':
    detect_img(YOLO())


