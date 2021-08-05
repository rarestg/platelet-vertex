import os
import requests

import numpy as np
import json
import cv2
cwd = os.getcwd()


# # BASE_URL = 'http://api.athelas.com'
# BASE_URL = 'http://staging-api.athelas.com'
# # TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2Mjg5ODQyMzQsImlhdCI6MTYyNjM5MjIzNCwic3ViIjoxODIxLCJpc19kZXZpY2UiOmZhbHNlLCJpc19hZG1pbiI6dHJ1ZSwiaW1wZXJzb25hdG9yX3VzZXJfaWQiOm51bGx9.9_YGgOyMMnY0hZAfY3VnR5q0BmLZas0IrY4wgzmvYko'
# TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MzA3MDM0NTIsImlhdCI6MTYyODExMTQ1Miwic3ViIjoyMzgsImlzX2RldmljZSI6ZmFsc2UsImlzX2FkbWluIjp0cnVlLCJpbXBlcnNvbmF0b3JfdXNlcl9pZCI6bnVsbH0.zTXFGXYXJMBLogQHEca3ED45pDi1lUZml-VyNcWeLfo'
HEIGHT = 2400
WIDTH = 3200
BUCKET = 'platelet-vertex-training/full-platelet-training/training-images/orig-images'
        
def split_and_save_images(path, job_id, image_number, NUM_SPLITS):
    
    full_image = cv2.imread(path)
    
    height, width, _ = np.shape(full_image)
    
    for i in range(NUM_SPLITS):
        for j in range(NUM_SPLITS):
            h_low = int(i * height // NUM_SPLITS)
            h_high = int((i + 1) * height // NUM_SPLITS)
            w_low = int(j * width // NUM_SPLITS)
            w_high = int((j + 1) * width // NUM_SPLITS)

            sub_region = full_image[h_low:h_high, w_low:w_high]
            cv2.imwrite(cwd + '/training-data/training-images/orig-images/{}_{}_({}, {}).jpg'.format(job_id, image_number, i, j), sub_region)
            

def create_translated_json_dict(keypoint, NUM_SPLITS):
    x, y = keypoint['x'], keypoint['y']
    pix_height, pix_width = keypoint['height'], keypoint['width']
    
    y_unit = HEIGHT//NUM_SPLITS
    x_unit = WIDTH//NUM_SPLITS
    
    y_translation = int(y/y_unit)
    new_y = y - (y_translation*y_unit)

    x_translation = int(x/x_unit)
    new_x = x - (x_translation*x_unit)
    
    coord = (y_translation, x_translation)
    
    xmin = (new_x - pix_width/2.0)/(WIDTH/NUM_SPLITS)
    xmax = (new_x + pix_width/2.0)/(WIDTH/NUM_SPLITS)
    ymin = (new_y - pix_height/2.0)/(HEIGHT/NUM_SPLITS)
    ymax = (new_y + pix_height/2.0)/(HEIGHT/NUM_SPLITS)

    xmin = 0 if xmin < 0 else xmin
    xmax = 1 if xmax > 1 else xmax
    ymin = 0 if ymin < 0 else ymin
    ymax = 1 if ymax > 1 else ymax
    
    json_dict = {'displayName': keypoint['note'], 'xMin': str(xmin), 'yMin': str(ymin), 'xMax': str(xmax), 'yMax': str(ymax)}
    return coord, json_dict

def load_images(job_id, NUM_SPLITS, true=True, save=True, database='prod'):
    if database == 'prod':
        BASE_URL = 'http://api.athelas.com'
        TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2Mjg5ODQyMzQsImlhdCI6MTYyNjM5MjIzNCwic3ViIjoxODIxLCJpc19kZXZpY2UiOmZhbHNlLCJpc19hZG1pbiI6dHJ1ZSwiaW1wZXJzb25hdG9yX3VzZXJfaWQiOm51bGx9.9_YGgOyMMnY0hZAfY3VnR5q0BmLZas0IrY4wgzmvYko'
    else:
        BASE_URL = 'http://staging-api.athelas.com'
        TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MzA3MDM0NTIsImlhdCI6MTYyODExMTQ1Miwic3ViIjoyMzgsImlzX2RldmljZSI6ZmFsc2UsImlzX2FkbWluIjp0cnVlLCJpbXBlcnNvbmF0b3JfdXNlcl9pZCI6bnVsbH0.zTXFGXYXJMBLogQHEca3ED45pDi1lUZml-VyNcWeLfo'
    endpoint = BASE_URL + '/jobs/' + str(job_id) + '/get_images'
    # print(endpoint)
    response = requests.get(endpoint, headers={ 'Authorization': 'Bearer {}'.format(TOKEN)})
    response = response.json()
    image_paths = response['images']
    all_json = []


    for image in image_paths:  
        coord_dict = {'({}, {})'.format(i, j): [] for i in range(NUM_SPLITS) for j in range(NUM_SPLITS)}
        image_url = image['raw']
        image_number = image_url.split('.jpg')[0].rsplit('/',1)[-1].split("-")[0]   
        
        actual_image = requests.get(image_url)
   
        if save:
            im_file = open(cwd + '/training-data/training-images/full-images/{}_{}.jpg'.format(job_id, image_number), 'wb')
            im_file.write(actual_image.content)
            im_file.close()

            split_and_save_images(cwd + '/training-data/training-images/full-images/{}_{}.jpg'.format(job_id, image_number), job_id, image_number, NUM_SPLITS)

        
        if true:
            for kp in image['keypoints']['true']:
                coord, json_dict = create_translated_json_dict(kp, NUM_SPLITS)
                coord_dict[str(coord)].append(json_dict)
        else:
            for kp in image['keypoints']['predicted']:
                coord, json_dict = create_translated_json_dict(kp, NUM_SPLITS)
                coord_dict[str(coord)].append(json_dict)

        for coord in coord_dict.keys():
            coord_json = {'imageGcsUri': 'gs://{}/{}_{}_{}.jpg'.format(BUCKET, job_id, image_number, coord), 'boundingBoxAnnotations': coord_dict[coord]}

            if save:
                with open(cwd + '/training-data/json-files/orig-files/{}_{}_{}.json'.format(job_id, image_number, coord), 'w') as f:
                    json.dump(coord_json, f)
            all_json.append(coord_json)
    
    # print(all_json)
    print('COMPLETE')
    return all_json

def load_jobs(jobs, NUM_SPLITS, save=True, true=True,database='prod'):
    master_json = {}
    for job_id in jobs:
        file = load_images(job_id, NUM_SPLITS, save=save, true=true, database=database)
        master_json[job_id] = file
    
    if save:
        with open(cwd + '/training-data/all_platelet_classification.jsonl', 'w') as f:
            for id in master_json.keys():
                json_file = master_json[id]
                for single_json in json_file:
                    json.dump(single_json, f)
                    f.write('\n')
    
    return master_json
                
if __name__ == '__main__':
    # job_list = ['200194', '200184', '200260', '200211', '199067', '201497', '201492', '199109', '200241', '200235', '200213', '200202']
    # job_list = ['200194', '200260', '199067', '201497', '200235']
    job_list = [30902]
    # job_list = [200235]
    job_list = [str(job) for job in job_list]

    load_jobs(job_list, 2)

        











            