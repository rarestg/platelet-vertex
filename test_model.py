import json
import requests
import os
import pickle

import request_platelet
import image_data
import draw_boxes

cwd = os.getcwd()
# job_list = ['201492',
# '200184',
# '200213',
# '200202']

job_list = [185026,
185032,
185035,
185038,
185047,
185055,
185066,
185074,
185084,
185104,
185113,
185120,
185127,
185138,
185148,
185160,
185166,
185176]


# job_list = ['185126', '185139', '185147', '185159', '185165', '185175']

# job_list = [199067, 200213, 200202, 200211, 200184]

job_list = [str(job_id) for job_id in job_list]

# do i awnt current model boxes as a json or as a dictionary. 
# lets do dictionary????
total_data = {}
FILENAME = './total_data.p'
if os.path.isfile(FILENAME):
    imported_data = pickle.load(open(FILENAME, 'rb'))
    if imported_data:
        total_data = imported_data

print('HERE')
master_json = image_data.load_jobs(job_list, 2, true=False,save=True)

for image in os.listdir(cwd + '/training-data/training-images/orig-images'):
    # print('image')
    prefix = image[:-4]
    job_id = prefix[:6]
    if job_id not in job_list:
        # print('job_id', job_id)
        continue
    print('THIS ONE WORKS!', job_id)
    image = cwd + '/training-data/training-images/orig-images/' + image
    key = image.split('/')[-1]
    if key in total_data:
        continue
    if image[-4:] != '.jpg':
        continue
    try:
        vertex_predictions = request_platelet.predict_image_object_detection_sample(
    project="457995363627",
    endpoint_id="1451197018989920256",
    location="us-central1",
    filename=image
)
        half_vertex_predictions = request_platelet.predict_image_object_detection_sample(
    project="457995363627",
    endpoint_id="8239810507297521664",
    location="us-central1",
    filename=image
)
    except Exception as e:
        print(e)
        continue
    
    filename = cwd + '/training-data/json-files/orig-files/' + prefix + '.json'
    old_model_count = draw_boxes.bound_box(filename, image)
    vertex_count = draw_boxes.bound_box_output(vertex_predictions, image)
    half_vertex_count = draw_boxes.half_bound_box(half_vertex_predictions, image)

    total_data[image.split('/')[-1]] = {'old_model_count': old_model_count, 'vertex_count': vertex_count, 'half_vertex_count': half_vertex_count}
    pickle.dump(total_data, open(FILENAME, 'wb'))
print('TOTAL', total_data)

vertex_dict = {}
old_model_dict = {}
half_vertex_dict = {}

for key in total_data.keys():
    print('{}: NewModel vs old: {}'.format(key, total_data[key]['vertex_count'] - total_data[key]['old_model_count']))
    if key[:7] not in vertex_dict.keys():
        vertex_dict[key[:7]] = 0
    vertex_dict[key[:7]] += total_data[key]['vertex_count']

    if key[:7] not in old_model_dict.keys():
        old_model_dict[key[:7]] = 0
    old_model_dict[key[:7]] += total_data[key]['old_model_count']

    if key[:7] not in half_vertex_dict.keys():
        half_vertex_dict[key[:7]] = 0
    half_vertex_dict[key[:7]] += total_data[key]['half_vertex_count']


for key in vertex_dict.keys():
    vertex_dict[key] = vertex_dict[key]/10.0
    old_model_dict[key] = old_model_dict[key]/10.0
    half_vertex_dict[key] = half_vertex_dict[key]/10.0

print('VERTEX_DICT', vertex_dict)
print('OLD MODEL', old_model_dict)
print('HALF MODEL', half_vertex_dict)


"""
WBC_CALIBRATION_FACTOR = 0.3423/6.65*12/11.5
PLATELET_DILUTION_RATE = 10
PLT_VOLUME_CALIBRATION = 0.78
PLT_CALIBRATION_FACTOR = PLATELET_DILUTION_RATE * \
    WBC_CALIBRATION_FACTOR * PLT_VOLUME_CALIBRATION
"""