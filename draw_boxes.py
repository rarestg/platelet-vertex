import cv2
import os
import numpy as np
import json
import glob
import matplotlib.pyplot as plt


HEIGHT = 2400
WIDTH = 3200
NUM_SPLITS = 2
cwd = os.getcwd()


def bound_box(filename, im_name):

    file = open(filename)
    data = json.load(file)

    im = cv2.imread(im_name)
    image = im.copy()

    for i, box in enumerate(data['boundingBoxAnnotations']):
        xMin = int(float(box['xMin']) * (WIDTH/NUM_SPLITS))
        xMax = int(float(box['xMax']) * (WIDTH/NUM_SPLITS))
        yMin = int(float(box['yMin']) * (HEIGHT/NUM_SPLITS))
        yMax = int(float(box['yMax']) * (HEIGHT/NUM_SPLITS))
        
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 0, 255), 1)

    cv2.imwrite(cwd + '/test-images/old-model/old_' + im_name.split('/')[-1], image)
    # cv2.imwrite(cwd + '/comparison_images/old_model_boxed/old_boxed_' + im_name.split('/')[-1], image)
    # plt.figure(figsize=(8,8))
    # plt.imshow(image)
    # print('image saved')
    return (len(data['boundingBoxAnnotations']))

def bound_box_output(vertex_predictions, im_name):

    # file = open(filename)
    # data = json.load(file)

    data = vertex_predictions

    im = cv2.imread(im_name)
    image = im.copy()

    for i, box in enumerate(data['bboxes']):
        xMin = int(float(box[0]) * (WIDTH/NUM_SPLITS))
        xMax = int(float(box[1]) * (WIDTH/NUM_SPLITS))
        yMin = int(float(box[2]) * (HEIGHT/NUM_SPLITS))
        yMax = int(float(box[3]) * (HEIGHT/NUM_SPLITS))
        
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 255, 0), 1)

    cv2.imwrite(cwd + '/test-images/full-trained-model/vertex_boxed_' + im_name.split('/')[-1], image)
    return(len(data['bboxes']))

def half_bound_box(vertex_predictions, im_name):

    # file = open(filename)
    # data = json.load(file)

    data = vertex_predictions

    im = cv2.imread(im_name)
    image = im.copy()

    for i, box in enumerate(data['bboxes']):
        xMin = int(float(box[0]) * (WIDTH/NUM_SPLITS))
        xMax = int(float(box[1]) * (WIDTH/NUM_SPLITS))
        yMin = int(float(box[2]) * (HEIGHT/NUM_SPLITS))
        yMax = int(float(box[3]) * (HEIGHT/NUM_SPLITS))
        
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 255, 0), 1)

    cv2.imwrite(cwd + '/test-images/half-trained-model/half_vertex_boxed_' + im_name.split('/')[-1], image)
    return(len(data['bboxes']))


if __name__ == '__main__':
    # print(cwd)
    # bound_box_output('/single_image/output.json', '/split_images/199067_112_376_(1, 0).jpg')
    # print('done')

    # for i in os.listdir(cwd + '/split_images'):
    #     i_name = i.split(".")[0]
    #     if len(i_name) == 0:
    #         continue
    #     for j in os.listdir(cwd + '/json_files'):
    #         j_name = j.split(".")[0]
    #         if j_name == i_name:
    #             bound_box(j_name + '.json', i_name + '.jpg')
    bound_box(cwd + '/training-data/json-files/bright-files/bright_185027_108_218_(0, 0).json', cwd + '/training-data/training-images/bright-images/bright_185027_108_218_(0, 0).jpg')