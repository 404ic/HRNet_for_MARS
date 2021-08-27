import glob
import yaml
import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cm as cm
import math

def plot_frame(project, config_file, frame_num, save=True, markersize=8, figsize=[15, 10]):
    config_fid = config_file
    with open(config_fid) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    legend_flag=[False,False]

    image = glob.glob(os.path.join(project, cfg['DATASET']['ROOT'], 'images', 'MARS_' + cfg['DATASET']['VIEW'] + '_' + f'{frame_num:05d}' + '.jpg'))
    if not image:
        print("I couldn't find image " + str(frame_num))
        return

    matched_id = frame_num 
    config_file_name = config_file.split('/')[-1].split('.')[0]
    infile = os.path.join(project, cfg['OUTPUT_DIR'], cfg['DATASET']['DATASET'], cfg['MODEL']['NAME'], config_file_name, 'results', 'MARS_format.json')
    with open(infile) as jsonfile:
        cocodata = json.load(jsonfile)
    pred = [i for i in cocodata['pred_keypoints'] if i['category_id'] == 1 and int(i['image_id']) == matched_id]
    gt = [i for i in cocodata['gt_keypoints']['annotations'] if i['category_id'] == 1 and int(i['image_id']) == matched_id]

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

    im = mpimg.imread(image[0])
    plt.figure(figsize=figsize)
    plt.imshow(im, cmap='gray')

    for pt in gt:
        for i, [x, y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
            plt.plot(x, y, color=colors[i], marker='o',  markeredgecolor='k',
                        markeredgewidth=math.sqrt(markersize)/4, markersize=markersize, linestyle='None',
                        label='ground truth' if not legend_flag[0] else None)
            legend_flag[0]=True
    for pt in pred:
        for i,[x,y] in enumerate(zip(pt['keypoints'][::3], pt['keypoints'][1::3])):
            plt.plot(x, y, color=colors[i], marker='^', markeredgecolor='w',
                        markeredgewidth=math.sqrt(markersize)/2, markersize=markersize, linestyle='None',
                        label='predicted' if not legend_flag[1] else None)
            legend_flag[1] = True

    plt.legend(prop={'size': 14})
    plt.show()
    if save:
        save_dir = os.path.join(project, cfg['OUTPUT_DIR'], cfg['DATASET']['DATASET'], cfg['MODEL']['NAME'], config_file_name, 'results')
        plt.savefig(os.path.join(save_dir, 'frame_' + str(frame_num) + '.jpg'))