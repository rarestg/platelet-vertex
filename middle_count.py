import os 
import numpy as np
import cv2
import requests
import request_platelet

cwd = os.getcwd()

HEIGHT = 2400
WIDTH = 3200
NUM_SPLITS = 3
y_unit = HEIGHT//NUM_SPLITS
x_unit = WIDTH//NUM_SPLITS

def pull_quads(desired_quads, image_path):
    print(desired_quads)
    
    full_image = cv2.imread(image_path)
    height, width, _ = np.shape(full_image)

    new_name = image_path.split('/')[-1].split('_')[0] + '_' + image_path.split('/')[-1].split('_')[1] + '_' + image_path.split('/')[-1].split('_')[2] 
    
    new_paths = []
    
    for i, j in desired_quads:
        h_low = int(i * height // NUM_SPLITS)
        h_high = int((i + 1) * height // NUM_SPLITS)
        w_low = int(j * width // NUM_SPLITS)
        w_high = int((j + 1) * width // NUM_SPLITS)

        sub_region = full_image[h_low:h_high, w_low:w_high]
        new_path = cwd + '/middle_count/quads/{}_({}, {}).jpg'.format(new_name, i, j)
        cv2.imwrite(new_path, sub_region)
        new_paths.append(new_path)
    return new_paths

def count_in_middle(job_id, desired_quads, database='prod'):
    if database == 'prod':
        BASE_URL = 'http://api.athelas.com'
        TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2Mjg5ODQyMzQsImlhdCI6MTYyNjM5MjIzNCwic3ViIjoxODIxLCJpc19kZXZpY2UiOmZhbHNlLCJpc19hZG1pbiI6dHJ1ZSwiaW1wZXJzb25hdG9yX3VzZXJfaWQiOm51bGx9.9_YGgOyMMnY0hZAfY3VnR5q0BmLZas0IrY4wgzmvYko'
    else:
        BASE_URL = 'http://staging-api.athelas.com'
        TOKEN = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MzA3MDM0NTIsImlhdCI6MTYyODExMTQ1Miwic3ViIjoyMzgsImlzX2RldmljZSI6ZmFsc2UsImlzX2FkbWluIjp0cnVlLCJpbXBlcnNvbmF0b3JfdXNlcl9pZCI6bnVsbH0.zTXFGXYXJMBLogQHEca3ED45pDi1lUZml-VyNcWeLfo'
    endpoint = BASE_URL + '/jobs/' + str(job_id) + '/get_images'
    response = requests.get(endpoint, headers={ 'Authorization': 'Bearer {}'.format(TOKEN)})
    response = response.json()
    image_paths = response['images']
    old_keypoints = []
    vertex_keypoints = []

    for image in image_paths:
        # print('IMAGE', image)
        image_url = image['raw']
        image_number = image_url.split('.jpg')[0].rsplit('/', 1)[-1].split("-")[0]
        actual_image = requests.get(image_url)
        image_path = cwd + '/middle_count/full_images/{}_{}_FULL.jpg'.format(job_id, image_number)

        im_file = open(image_path, 'wb')
        im_file.write(actual_image.content)
        im_file.close()
        
        new_paths = pull_quads(desired_quads, image_path)

        for kp in image['keypoints']['predicted']:

            coord = (int(kp['y']/ y_unit), int(kp['x']/x_unit))

            if coord in desired_quads:
                old_keypoints.append(kp)

        for quad_path in new_paths:
            vertex_predictions = request_platelet.predict_image_object_detection_sample(
                project="457995363627",
                endpoint_id="1451197018989920256",
                location="us-central1",
                filename=quad_path
            )
            
        vertex_keypoints += vertex_predictions['bboxes']
            # for box in vertex_predictions['bboxes']:
                
                # xMin = int(float(box[0]) * (WIDTH/NUM_SPLITS))
                # xMax = int(float(box[1]) * (WIDTH/NUM_SPLITS))
                # yMin = int(float(box[2]) * (HEIGHT/NUM_SPLITS))
                # yMax = int(float(box[3]) * (HEIGHT/NUM_SPLITS))

                # center = ((xMax + xMin)/2, (yMax - yMin)/2)
                # coord = (int(center[1]/y_unit), int(center[0]/x_unit))

                # if coord in desired_quads:
                #     vertex_keypoints.append(point)

    return old_keypoints, vertex_keypoints

if __name__ == '__main__':
    
    job_ids = [208744,
208752,
208762,
208765,
208769,
208773]
    desired_quads = [(1,1)]
    extrap_values = {}
    new_mod_extrap = {}
    for job_id in job_ids:

        old_keypoints, vertex_keypoints = count_in_middle(job_id, desired_quads, database='prod')
        extrap_val = len(old_keypoints)*9/10
        vertex_extrap = len(vertex_keypoints)*9/10

        print('{}: OLD COUNT: {}, NEW COUNT: {}'.format(job_id, len(old_keypoints), len(vertex_keypoints)))

        print('{}: OLD EXTRAPOLATED'.format(job_id), extrap_val)
        print('{}: VERTEX EXTRAPOLATED'.format(job_id), vertex_extrap)
        extrap_values[str(job_id)] = extrap_val
        new_mod_extrap[str(job_id)] = vertex_extrap
    print('ALL OLD EXTRAP VALS:', extrap_values)
    print('ALL VERTEX EXTRAP VALS:', new_mod_extrap)
    
        
        

                