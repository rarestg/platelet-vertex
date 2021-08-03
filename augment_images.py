import os
from imgaug.augmenters.blend import BoundingBoxesMaskGen
import numpy as np
import random
import json
import cv2
from imgaug import augmenters as iaa

import matplotlib.pyplot as plt

cwd = os.getcwd()

def lr_flip(image_name, json_name, large_json=None, save=True):

    image = cv2.imread(image_name)
    image_h = image.copy()
    
    horiz_flip = iaa.Fliplr(1.0)
    horiz_flip_image = horiz_flip(image = image_h)
    horiz_flip_file = horiz_flip_json(json_name)

    if save:
        cv2.imwrite(cwd + '/training-data/training-images/horiz-images/horiz_' + image_name.split('/')[-1], horiz_flip_image)
        with open(cwd + '/training-data/json-files/horiz-files/horiz_' + json_name.split('/')[-1], 'w') as f:
            json.dump(horiz_flip_file, f)


    return horiz_flip_image, horiz_flip_file

def ud_flip(image_name, json_name, large_json=None, save=True):

    image = cv2.imread(image_name)
    image_v = image.copy()

    vert_flip = iaa.Flipud(1.0)

    vert_flip_image = vert_flip(image = image_v)
    vert_flip_file = vert_flip_json(json_name)

    if save:
        cv2.imwrite(cwd + '/training-data/training-images/vert-images/vert_' + image_name.split('/')[-1], vert_flip_image)
        with open(cwd + '/training-data/json-files/vert-files/vert_' + json_name.split('/')[-1], 'w') as f:
            json.dump(vert_flip_file, f)
        

    return vert_flip_image, vert_flip_file

def random_stitch(image_name, json_name, large_json=None, save=True, split=2):
    """
    Can only do even number splits right now
    """

    full_image = cv2.imread(image_name)
    height, width = full_image.shape[0], full_image.shape[1]
    new_image = np.zeros(full_image.shape)
    json_prefix = json_name.split('/')[-1][:-5]
    file = open(json_name)
    data = json.load(file)

    map_keys = [str(x) for x in range(1, split*split + 1)]
    mappings = {}
    for m in range(split):
        for n in range(split):
            key = map_keys.pop(0)
            mappings[key] = (m, n)
    new_mappings = {}

    count = 0
    #ij is content, row, col is locatio
    for row in range(split):
        for col in range(split):
            key = random.choice(list(mappings))
            i, j = mappings[key]
            new_mappings[(i, j)] = (row, col)

            h_low = int(i * height // split)
            h_high = int((i + 1) * height // split)
            w_low = int(j * width // split)
            w_high = int((j + 1) * width // split)

            r_low = int(row * height // split)
            r_high = int((row + 1) * height // split)
            c_low = int(col * width // split)
            c_high = int((col + 1) * width // split)

            sub_region = full_image[h_low:h_high, w_low:w_high]
            new_image[r_low:r_high, c_low:c_high] = sub_region

            old_key = mappings.pop(key, None)

    unit = 1.0/split

    new_box_list = []
    new_json_dict = {}
    
    for p, box in enumerate(data['boundingBoxAnnotations']):
        new_box = {}
        new_box['displayName'] = box['displayName']
        xMin, xMax, yMin, yMax = float(box['xMin']), float(box['xMax']), float(box['yMin']), float(box['yMax'])
        
        
        center = ((xMax + xMin)/2.0, (yMax + yMin)/2.0)
        
        current_coord = (int(center[1]/unit), int(center[0]/unit))
        
        new_coord = new_mappings[current_coord]
        
        translation = ((new_coord[1] - current_coord[1])*unit, (new_coord[0] - current_coord[0])*unit)
        

        new_box['xMin'] = str(xMin + translation[0])
        new_box['xMax'] = str(xMax + translation[0])
        new_box['yMin'] = str(yMin + translation[1])
        new_box['yMax'] = str(yMax + translation[1])

        new_box["xMin"] = 0 if float(new_box["xMin"]) < 0 else new_box["xMin"]
        new_box["xMax"] = 1 if float(new_box["xMax"]) > 1 else new_box["xMax"]
        new_box["yMin"] = 0 if float(new_box["yMin"]) < 0 else new_box["yMin"]
        new_box["yMax"] = 1 if float(new_box["yMax"]) > 1 else new_box["yMax"]

        new_box_list.append(new_box)
        
        
    new_json_dict["imageGcsUri"] = 'gs://platelet-vertex-training/full-platelet-training/training-images/stitch-images/restitch_{}.jpg'.format(json_prefix)
    new_json_dict["boundingBoxAnnotations"] = new_box_list

    
    if save:
        cv2.imwrite(cwd + '/training-data/training-images/stitch-images/restitch_' + image_name.split('/')[-1], new_image)
        with open(cwd + '/training-data/json-files/stitch-files/restitch_' + json_name.split('/')[-1], 'w') as f:
            json.dump(new_json_dict, f)


    return new_image, new_json_dict
            

def change_brightness(image_name, json_name, large_json=None, save=True):
    
    img = cv2.imread(image_name)
    image = img.copy()

    file = open(json_name)
    data = json.load(file)
    new_json = data.copy()


    change = iaa.Sequential([
        iaa.AddToBrightness((-50,50)),
        iaa.Flipud(1.0),
        iaa.Fliplr(1.0)
    ])
    
    new_image = change(image = image)
    new_json = horiz_flip_json(new_json, json_prefix=json_name.split('/')[-1][:-5])
    new_json = vert_flip_json(new_json, json_prefix=json_name.split('/')[-1][:-5])

    new_json["imageGcsUri"] = "gs://platelet-vertex-training/full-platelet-training/training-images/bright-images/bright_{}".format(image_name.split('/')[-1])
    
    if save:
        cv2.imwrite(cwd + '/training-data/training-images/bright-images/bright_' + image_name.split('/')[-1], new_image)
        with open(cwd + '/training-data/json-files/bright-files/bright_' + json_name.split('/')[-1], 'w') as f:
            json.dump(new_json, f)
                
    return new_image, new_json
    

    

def horiz_flip_json(json_name, json_prefix=None):
    
    if type(json_name) == dict:
        data = json_name
    else:
        file = open(json_name)
        data = json.load(file)
        json_prefix = json_name.split('/')[-1][:-5]
    new_bounding_boxes = []
    new_json_dict = {}

    for i, box in enumerate(data['boundingBoxAnnotations']):
        new_box = {}
        
        new_box["displayName"] = box["displayName"]
        new_box["xMax"] = str(1.0 - float(box["xMin"]))
        new_box["xMin"] = str(1.0 - float(box["xMax"]))
        new_box["yMin"] = box["yMin"]
        new_box["yMax"] = box["yMax"]
        
        new_box["xMin"] = 0 if float(new_box["xMin"]) < 0 else new_box["xMin"]
        new_box["xMax"] = 1 if float(new_box["xMax"]) > 1 else new_box["xMax"]


        new_bounding_boxes.append(new_box)

    new_json_dict["imageGcsUri"] = 'gs://platelet-vertex-training/full-platelet-training/training-images/horiz-images/horiz_{}.jpg'.format(json_prefix)
    new_json_dict["boundingBoxAnnotations"] = new_bounding_boxes
    return new_json_dict                                    


def vert_flip_json(json_name, json_prefix=None):

    if type(json_name) == dict:
        data = json_name
    else:
        file = open(json_name)
        data = json.load(file)
        json_prefix = json_name.split('/')[-1][:-5]
    new_bounding_boxes = []
    new_json_dict = {}
    
    for i, box in enumerate(data['boundingBoxAnnotations']):

        new_box = {}

        new_box["displayName"] = box["displayName"]
        new_box["xMin"] = box["xMin"]
        new_box["xMax"] = box["xMax"]
        new_box["yMin"] = str(1.0 - float(box["yMax"]))
        new_box["yMax"] = str(1.0 - float(box["yMin"]))

        new_box["yMin"] = 0 if float(new_box["yMin"]) < 0 else new_box["yMin"]
        new_box["yMax"] = 1 if float(new_box["yMax"]) > 1 else new_box["yMax"]
        
        new_bounding_boxes.append(new_box)
        
    
    new_json_dict["imageGcsUri"] = 'gs://platelet-vertex-training/full-platelet-training/training-images/vert-images/vert_{}.jpg'.format(json_prefix)
    new_json_dict["boundingBoxAnnotations"] = new_bounding_boxes
    return new_json_dict


HEIGHT = 1200
WIDTH = 1600
NUM_SPLITS = 1
cwd = os.getcwd()


def bound_box(filename, im_name):

    data = filename
    im = im_name
    image = np.array(im.copy())
    image = np.ascontiguousarray(image, dtype=np.uint8)

    for i, box in enumerate(data['boundingBoxAnnotations']):
        xMin = int(float(box['xMin']) * (WIDTH/NUM_SPLITS))
        xMax = int(float(box['xMax']) * (WIDTH/NUM_SPLITS))
        yMin = int(float(box['yMin']) * (HEIGHT/NUM_SPLITS))
        yMax = int(float(box['yMax']) * (HEIGHT/NUM_SPLITS))
        
        xMin = 0 if xMin < 0 else xMin
        xMax = int(WIDTH/NUM_SPLITS - 1) if xMax > WIDTH/NUM_SPLITS - 1 else xMax
        yMin = 0 if yMin < 0 else yMin
        yMax = int(WIDTH/NUM_SPLITS - 1) if yMax > HEIGHT/NUM_SPLITS - 1 else yMax
        
        if yMax - yMin > 200:
            print('CHECK HERE', xMin, xMax, yMin, yMax)
#         print(image.shape)
        print(xMin, xMax, yMin, yMax)
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 0, 255), 1)

#     cv2.imwrite(cwd + '/comparison_images/old_model_boxed/old_boxed_' + im_name.split('/')[-1], image)
    # print('image saved')
    plt.figure(figsize=(16,16))
    plt.imshow(image)
    return (len(data['boundingBoxAnnotations']))

def bound_box_output(vertex_predictions, im_name):

    data = vertex_predictions

    # print(type(data))
    # print(im_name)
    im = cv2.imread(im_name)
    image = im.copy()

    for i, box in enumerate(data['bboxes']):
        xMin = int(float(box[0]) * (WIDTH/NUM_SPLITS))
        xMax = int(float(box[1]) * (WIDTH/NUM_SPLITS))
        yMin = int(float(box[2]) * (HEIGHT/NUM_SPLITS))
        yMax = int(float(box[3]) * (HEIGHT/NUM_SPLITS))
        
        cv2.rectangle(image, (xMin, yMin), (xMax, yMax), (0, 255, 0), 1)

    cv2.imwrite(cwd + '/comparison_images/vertex_boxed/vertex_boxed_' + im_name.split('/')[-1], image)
    
    return(len(data['bboxes']))

def augment_all_images():

    large_json = cwd + '/training-data/all_platelet_classification.jsonl'
    all_json = []
    if os.path.exists(large_json):
        with open(large_json, "r") as file:
            json_list = list(file)
        for json_string in json_list:
            result = json.loads(json_string)
            all_json.append(result)

    
    for filename in os.listdir( './training-data/training-images/orig-images'):
        if filename[-4:] != '.jpg':
            continue

        image_name = cwd + '/training-data/training-images/orig-images/' + filename
        json_name = cwd + '/training-data/json-files/orig-files/' + filename[:-4] + '.json'

        horiz_image, horiz_json = lr_flip(image_name, json_name, large_json=large_json, save=True)
        vert_image, vert_json = ud_flip(image_name, json_name, large_json=large_json, save=True)
        bright_image, bright_json = change_brightness(image_name, json_name, large_json=large_json, save=True)
        stitch_image, stitch_json = random_stitch(image_name, json_name, large_json=large_json, save=True, split=4)

        all_json += [horiz_json, vert_json, bright_json, stitch_json]

    with open(large_json, 'w') as j:
        for json_file in all_json:
            json.dump(json_file, j)
            j.write('\n')

    print('COMPLETE')

# cwd + '/training-data/training-images/orig-images/' + 
#cwd + '/training-data/json-files/orig-files/' + 
if __name__ == "__main__":
    augment_all_images()