'''

'''

import json
from pprint import pprint
import os.path as osp
import yaml
import io
import numpy as np
import data_utils.json_util as json_util

def clean_part_list(project, manifest_file, part_names=['nose tip','right ear','left ear','neck','right side body','left side body','tail base'], num_workers_max=5):
    data = []
    with open(osp.join(project, 'annotations', manifest_file)) as json_file:
        data = json.load(json_file)
    output_all = open(osp.join(project, 'annotations', 'cleaned_parts.manifest'), 'w')
    num_workers = num_workers_max
    for line in data:
        frame = dict(line)
        annotations = frame['annotatedResult']['annotationsFromAllWorkers']
        worker_num = 0
        for annot in annotations:
            keypts = eval(annot['annotationData']['content'])['annotatedResult']['keypoints']
            ind = [k for k in range(len(keypts)) if ' '.join(keypts[k]['label'].split(' ')[2:]) not in part_names] # assumes annotations are in "[color] mouse [keypoint name]" format
            ind.sort(reverse = True)
            for k in ind:
                keypts.pop(k)
            tmp = eval(annot['annotationData']['content'])
            tmp['annotatedResult']['keypoints'] = keypts
            annot['annotationData']['content'] = json.dumps(tmp)
            annot['workerID'] = str(worker_num)
            worker_num += 1
        annotations_copy = annotations[:]
        frame['annotatedResult']['annotationsFromAllWorkers'] = []
        if len(annotations_copy) < 5:
            print(len(annotations_copy))
            pprint(frame)
        for i in range(min(num_workers, len(annotations_copy))):
            frame['annotatedResult']['annotationsFromAllWorkers'].append(annotations_copy[i])
        output_all.write(json.dumps(frame))
        output_all.write('\n')
    output_all.close()

def get_id(path):
    return int(path.split('/')[-1].split('_')[-1].split('.')[0])

def convert_bbox_format(bbox):
    return [bbox[0], bbox[2], bbox[1] - bbox[0], bbox[3] - bbox[2]]

def convert_keypoint_format(coords):
    Y = coords[0]
    X = coords[1]
    keypoints = []
    assert len(X) == len(Y)
    for i in range(len(X)):
        keypoints.append(X[i])
        keypoints.append(Y[i])
        keypoints.append(2)
    assert len(keypoints) == 21
    return keypoints

def convert_to_coco_format(project, data, data_type='train', annot_id=0):
    with open(osp.join(project, "project_config.yaml"), 'r') as stream:
        config = yaml.safe_load(stream)
    name = config['species']
    # skeleton = [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]]
    skeleton = config['skeleton']
    view = config['view']
    part_names = data[0]['ann_label']
    coco = {}
    coco['info'] = {
            'description': data_type,
            'url': None,
            'version': '0.0', 'year': 2021,
            'contributor': 'First Last',
            'date_created': 'August 24, 2021'
        }
    coco['licenses'] = None
    coco['categories'] = [
        {
            'id': 1,
            'keypoints': part_names,
            'name': name,
            'skeleton': skeleton,
            'supercategory': 'mouse'
            }
        ]
    coco['images'] = []
    coco['annotations'] = []
    for frame in data:
        assert frame['ann_label'] == part_names
        coco['images'].append(
            {
                'coco_url': None,
                'date_captured': None,
                'file_name': frame['image'].split('/')[-1],
                'flickr_url': None,
                'height': frame['height'],
                'id': get_id(frame['image']),
                'license': None,
                'width': frame['width']
            }
        )
        for color in ['ann_black', 'ann_white']: # this is hard coded, whoops
            coco['annotations'].append(
                {
                    'area': frame[color]['area'],
                    'bbox': convert_bbox_format(frame[color]['bbox']),
                    'category_id': 1,
                    'id': annot_id,
                    'image_id': get_id(frame['image']),
                    'iscrowd': 0,
                    'keypoints': convert_keypoint_format(frame[color]['med']),
                    'num_keypoints': len(frame['ann_label']),
                    'segmentation': None
                }
            )
            annot_id += 1
    with open(osp.join(project, 'annotations', 'keypoints_' + view + '_' + data_type + '.json'), 'w') as f:
        json.dump(coco, f)
    return annot_id

def process_all_keypoints(project):
    with open(osp.join(project, "project_config.yaml"), 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    manifest_file = data_loaded['manifest_name']
    project = data_loaded['project_name']
    pixels_per_cm = data_loaded['pixels_per_cm']
    part_names = data_loaded['keypoints']
    data_split = data_loaded['train_val_test']
    view = data_loaded['view']
    clean_part_list(project, manifest_file=manifest_file, part_names=part_names)
    json_util.make_annot_dict(project)

    assert sum(data_split) == 1, 'Your train/val/test split fraction do not add up to 1'
    with open(osp.join(project, 'annotations', 'processed_keypoints.json'), 'r') as f:
        all_data = json.load(f)
    # can add data shuffle step here
    n = len(all_data)
    n_train = int(np.floor(n * data_split[0]))
    n_val = int(np.floor(n * data_split[1]))
    annot_id = convert_to_coco_format(project, all_data[:n_train], data_type='train', annot_id=0)
    annot_id = convert_to_coco_format(project, all_data[n_train:n_train+n_val], data_type='val', annot_id=annot_id)
    annot_id = convert_to_coco_format(project, all_data[n_train+n_val:], data_type='test', annot_id=annot_id)
    with open(osp.join(project, 'annotations', 'processed_keypoints_' + view + '_train.json'), 'w') as f:
        json.dump(all_data[:n_train], f)
    with open(osp.join(project, 'annotations', 'processed_keypoints_' + view + '_val.json'), 'w') as f:
        json.dump(all_data[n_train:n_train+n_val], f)
    with open(osp.join(project, 'annotations', 'processed_keypoints_' + view + '_test.json'), 'w') as f:
        json.dump(all_data[n_train+n_val:], f)
    


    
