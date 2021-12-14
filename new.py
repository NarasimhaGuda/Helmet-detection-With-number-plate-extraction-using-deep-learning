
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
from imutils.video import FPS
import imutils
import time
from decimal import Decimal, ROUND_HALF_UP

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'inference_graph'
VIDEO_NAME = 'accurate.mp4'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'labelmap.pbtxt')
PATH_TO_VIDEO = os.path.join(CWD_PATH, VIDEO_NAME)
NUM_CLASSES = 4
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.compat.v1.Session(graph=detection_graph)
fps = FPS().start()
video = cv2.VideoCapture(PATH_TO_VIDEO)
import json
from collections import OrderedDict
from glob import glob
import cv2
import requests


def platemain():
    regions = ['in']
    result = []
    path = 'frame.jpg'
    with open(path, 'rb') as fp:
        response = requests.post(
            'https://api.platerecognizer.com/v1/plate-reader/',
            files=dict(upload=fp),
            data=dict(regions=regions),
            headers={'Authorization': 'Token ' + 'fe801a314498e5fd43a6069099e65b7bc5ff9c3d'})
        print("respond.status_code", response.status_code)
        result.append(response.json(object_pairs_hook=OrderedDict))
    print(result)
    time.sleep(1)
    im = cv2.imread(path)
    resp_dict = json.loads(json.dumps(result, indent=2))
    if resp_dict[0]['results']:
        num = resp_dict[0]['results'][0]['plate']
        boxs = resp_dict[0]['results'][0]['box']
        xmins, ymins, ymaxs, xmaxs = boxs['xmin'], boxs['ymin'], boxs['ymax'], boxs['xmax']
        #    cv2.imshow("image",im)
        #    cv2.waitKey(0)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #    cv2.imshow("Gray Image",img)
        #    cv2.waitKey(0)
        edges = cv2.Canny(img, 100, 200)
        #    cv2.imshow("Edge Image",edges)
        #    cv2.waitKey(0)
        cv2.rectangle(im, (xmins, ymins), (xmaxs, ymaxs), (255, 0, 0), 2)
        cv2.rectangle(edges, (xmins, ymins), (xmaxs, ymaxs), (255, 0, 0), 2)
        #    cv2.imshow("Box Edges",edges)
        #    cv2.waitKey(0)
        #    cv2.imshow("Box On Original",im)
        #    cv2.waitKey(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, num, (xmins, ymins - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        #    cv2.imshow("Number",im)
        #    cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("the bike number is {}".format(str(num).upper()))
        return str(num).upper()


def resize(w, h, w_box, h_box, pil_image):
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height))


j = 0
s = []
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        nmp = []
        st = []
        while True:
            start_time = time.time()
            j += 1
            ret, image_np = video.read()
            if ret == True:
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                #        scores=(scores[0]*10).astype(int)
                cl = (classes[0].astype(int))[:7]
                sc = (scores[0][:7]) * 100
                sc = list(map(int, sc))
                # print(cl,sc)
                d = {1: 0, 2: 0, 3: 0, 4: 0}
                for i in range(len(cl)):
                    if (sc[i] > d[cl[i]]):
                        d[cl[i]] = sc[i]
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=4,
                    min_score_thresh=0.85)
                data = {}
                if (d[3] > 90):
                    if (d[4] < 90):
                        print("Without Helmet: No Number plate")

                if (d[3] > 90) and (d[4]>80):
                    if (d[4] > 80):
                        name = 'frame.jpg'
                        print('Creating...' + name)
                        cv2.imwrite(name, image_np)
                        number = platemain()
                        # if (d[4] > 80):
                        # name = 'frame.jpg'
                        # print('Creating...' + name)
                        # cv2.imwrite(name, image_np)
                        # number = platemain()
                    print("Without Helmet Number plate: ",number)
                elif (d[2] > 90) and (d[4]>80):
                    if (d[4] > 80):
                        name = 'frame.jpg'
                        print('Creating...' + name)
                        cv2.imwrite(name, image_np)
                        number = platemain()
                        # if (d[4] > 80):
                        # name = 'frame.jpg'
                        # print('Creating...' + name)
                        # cv2.imwrite(name, image_np)
                        # number = platemain()
                    print("With Helmet Number plate:",number)
                elif (d[1] < 90):
                    print("Bike rider detected")
                # print(nmp, st)
                print('Iteration %d: %.3f sec' % (j, time.time() - start_time))
                cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
                # cv2.waitKey()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        for m in range(len(nmp)):
            data = {'Status': st[m],
                    'Number_Plate': nmp[m]
                    }
            print(data)

            m = json.dumps(data)
            with open("D:\helmet detection\data1.json", "w") as f:
                f.write(m)
cv2.destroyAllWindows()
