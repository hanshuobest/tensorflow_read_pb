#coding:utf-8
# author:hanshuo

import tensorflow as tf
import numpy as np

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    # anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    anchors_tensor = tf.reshape(tf.constant(anchors , tf.float32), [1, 1, 1, num_anchors, 2])

    # grid_shape = K.shape(feats)[1:3] # height, width
    grid_shape = tf.shape(feats)[1:3]

    # grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),[1, grid_shape[1], 1, 1])
    # grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),[grid_shape[0], 1, 1, 1])

    grid_y = tf.tile(tf.reshape(tf.range(0 , limit=grid_shape[0]) , [-1 , 1 , 1 , 1]) , [1, grid_shape[1], 1, 1])
    grid_x = tf.tile(tf.reshape(tf.range(0 , limit=grid_shape[1]) , [1 , -1 , 1 , 1]) , [grid_shape[0], 1, 1, 1])

    # grid = K.concatenate([grid_x, grid_y])
    # grid = K.cast(grid, K.dtype(feats))

    grid = tf.concat([grid_x , grid_y] , -1)
    grid = tf.cast(grid , feats.dtype.base_dtype.name)


    # feats = K.reshape(
        # feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    feats = tf.reshape(feats , [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    # box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #
    # box_confidence = K.sigmoid(feats[..., 4:5])
    # box_class_probs = K.sigmoid(feats[..., 5:])

    box_xy = (tf.sigmoid(feats[...,:2]) + grid) / tf.cast(grid_shape[::-1] , feats.dtype.base_dtype.name)
    box_wh = tf.exp(feats[...,2:4]) * anchors_tensor / tf.cast(input_shape[::-1] , feats.dtype.base_dtype.name)
    box_confidence = tf.sigmoid(feats[...,4:5])
    box_class_probs = tf.sigmoid(feats[...,5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]

    input_shape = tf.cast(input_shape , box_yx.dtype.base_dtype.name)
    image_shape = tf.cast(image_shape , box_yx.dtype.base_dtype.name)
    new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes = tf.concat([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ] , -1)

    # Scale boxes back to original image shape.
    boxes *= tf.concat([image_shape , image_shape] , -1)
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,anchors, num_classes, input_shape)

    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = tf.reshape(boxes , [-1 , 4])
    box_scores = box_confidence * box_class_probs
    box_scores = tf.reshape(box_scores , [-1 , num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""


    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = tf.shape(yolo_outputs[0])[1:3] * 32

    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # boxes = K.concatenate(boxes, axis=0)
    # box_scores = K.concatenate(box_scores, axis=0)

    boxes = tf.concat(boxes , axis=0)
    box_scores = tf.concat(box_scores , axis=0)

    mask = box_scores >= score_threshold
    # max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    max_boxes_tensor = tf.constant(max_boxes , dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []

    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # class_boxes = K.gather(class_boxes, nms_index)
        # class_box_scores = K.gather(class_box_scores, nms_index)
        # classes = K.ones_like(class_box_scores, 'int32') * c

        class_boxes = tf.gather(class_boxes , nms_index)
        class_box_scores = tf.gather(class_box_scores, nms_index)
        classes = tf.ones_like(class_box_scores, 'int32') * c

        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
        # boxes_ = K.concatenate(boxes_, axis=0)
        # scores_ = K.concatenate(scores_, axis=0)
        # classes_ = K.concatenate(classes_, axis=0)

    boxes_ = tf.concat(boxes_, axis=0)
    scores_ = tf.concat(scores_, axis=0)
    classes_ = tf.concat(classes_, axis=0)

    return boxes_, scores_, classes_


