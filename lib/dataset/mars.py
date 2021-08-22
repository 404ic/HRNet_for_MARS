# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
from pprint import pprint
import math
import copy

from MARSeval.coco import COCO
from MARSeval.cocoeval import COCOeval
import json_tricks as json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms


logger = logging.getLogger(__name__)


class MARSDataset(JointsDataset):
    '''
    "keypoints": {
        0: "nose tip", 
        1: "right ear", 
        2: "left ear", 
        3: "neck", 
        4: "right side body", 
        5: "left side body", 
        6: "tail base"
    },
	"skeleton": [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [3, 4],
        [3, 5],
        [4, 6],
        [5, 6]
        ]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 7
        self.flip_pairs = [[1, 2], [4,5]]
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3)
        self.lower_body_ids = (4, 5, 6)

        self.joints_weight = np.array(
            [
                1., 1., 1., 1., 1., 1., 1.
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.image_set else 'person_keypoints'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                # print(ipt, obj['keypoints'])
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%06d.jpg' % index
        # if '2014' in self.image_set:
        #     file_name = 'COCO_%s_' % self.image_set + file_name

        # prefix = 'test2017' if 'test' in self.image_set else self.image_set

        prefix = self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def _pr_curve(self, res_folder, performance):

        rs_mat = performance.params.recThrs  # [0:.01:1] R=101 recall thresholds for evaluation
        ps_mat = performance.eval['precision']
        print('Statistics:')
        eval = performance
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        # pprint(rs_mat)
        # pprint(ps_mat)
        iou_mat = performance.params.iouThrs
        # pprint(iou_mat)
        jet = plt.get_cmap('jet')
        cNorm = colors.Normalize(vmin=0, vmax=len(iou_mat))
        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)

        show = [0.5, 0.75, 0.85, 0.9, 0.95]
        fig, ax = plt.subplots(1, figsize=[8, 6])
        for i in range(len(iou_mat)):
            if round(iou_mat[i]*1000.)/1000. in show:
                colorVal = scalarMap.to_rgba(i)
                # pprint(ps_mat.shape)
                # pprint(ps_mat[i, :, :, 0, :])
                ax.plot(rs_mat, ps_mat[i, :, :, 0, 0], c=colorVal, ls='-', lw=2, label='IoU >= %s' % np.round(iou_mat[i], 2))
        plt.grid()
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('PR curves for HRNet Keypoint Estimator')
        plt.legend(loc='best')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.savefig(os.path.join(res_folder,  'PR_curves.png'))
        plt.show()

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-10:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]
            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['keypoints'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )
            else:
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set or 1 == 1:
            info_str, performance = self._do_python_keypoint_eval(
                res_file, res_folder)
            self._pr_curve(res_folder, performance)
            self._format_to_MARS_Developer_dict_format(res_file, res_folder)
            perfs = self._coco_eval(res_folder)
            self._plot_model_PCK(res_folder, perfs, xlim=[-0.03, 1.03], combine_animals=True, print_PCK_values=True)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float
            )

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': list(key_points[k]),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(len(img_kpts))
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        parts = ["nose tip", "right ear", "left ear", "neck", "right side body", "left side body", "tail base"]
        coco_eval = COCOeval(self.coco, coco_dt, 'keypoints', sigmaType='MARS_top', useParts=parts)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats_names = ['AP', 'AR']
        # pprint(coco_eval.eval['precision'])
        # pprint(coco_eval.eval['recall'])
        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        return info_str, coco_eval

    def _compute_model_pck(self, cocoEval, lims=None, pixels_per_cm=None, pixel_units=False):

        bins = 10000
        pck = []
        partID = list(cocoEval.cocoGt.catToImgs.keys())[0]  # which body part are we looking at?
        for i in cocoEval.params.imgIds:
            # pprint(cocoEval._gts)
            # pprint(cocoEval.ious)
            # pprint(i)
            # pprint(partID)
            # pprint(cocoEval._gts[i, partID])
            pck.append(cocoEval.computePcks(int(i), partID)[0])
        pck = np.array(pck)

        if not lims:
            lims = [0, max(pck)]

        counts, binedges = np.histogram(pck, bins, range=lims)
        counts = counts / len(pck)
        binctrs = (binedges[:-1] + binedges[1:]) / 2
        if not pixel_units:
            binctrs = binctrs / pixels_per_cm

        return counts, binctrs
    
    

    def _compute_human_PCK(self, animal_names, xlim=None, pixel_units=False):
        pixels_per_cm = 37.7
        dictionary_file_path = os.path.join('/home/ericma/deep-high-resolution-net.pytorch/data/mars', 'test_processed_keypoints.json')
        if not os.path.exists(dictionary_file_path):
            assert 1 == 0, 'Must add the human annotation file to ' + dictionary_file_path
        with open(dictionary_file_path, 'r') as fp:
            D = json.load(fp)

        bins = 10000
        if xlim:
            binrange = [-1 / bins, xlim[1] * 2 + 1 / bins]
        else:
            binrange = [-1 / bins, max(D[0]['width'], D[0]['height']) + 1 / bins]
        nSamp = len(D)
        nKpts = len(D[0]['ann_label'])

        fields = ['min', 'max', 'mean', 'med']
        ptNames = D[0]['ann_label']
        ptNames = ['all'] + ptNames

        counts = {a: {p: {n: [] for n in fields} for p in ptNames} for a in animal_names}
        super_counts = {p: {n: np.zeros(bins) for n in fields} for p in ptNames}
        for cnum, animal in enumerate(animal_names):
            dMean = np.zeros((nKpts, nSamp))  # average worker-gt distance
            dMedian = np.zeros((nKpts, nSamp))  # median worker-gt distance
            dMin = np.zeros((nKpts, nSamp))  # performance of best worker on a given frame
            dMax = np.zeros((nKpts, nSamp))  # performance of worst worker on a given frame

            for fr, frame in enumerate(D):
                X = np.array(frame['ann_' + animal]['X']) * D[0]['width']
                Y = np.array(frame['ann_' + animal]['Y']) * D[0]['height']
                trial_dists = []
                for i, [pX, pY] in enumerate(zip(X, Y)):
                    mX = np.median(np.delete(X, i, axis=0), axis=0)
                    mY = np.median(np.delete(Y, i, axis=0), axis=0)
                    trial_dists.append(np.sqrt(np.square(mX - pX) + np.square(mY - pY)))
                trial_dists = np.array(trial_dists)

                dMean[:, fr] = np.mean(trial_dists, axis=0)
                dMedian[:, fr] = np.median(trial_dists, axis=0)
                dMin[:, fr] = np.min(trial_dists, axis=0)
                dMax[:, fr] = np.max(trial_dists, axis=0)

            for c, use in enumerate([dMin, dMax, dMean, dMedian]):
                for p, pt in enumerate(use):
                    counts[animal][ptNames[p+1]][fields[c]], usedbins = np.histogram(pt, bins, range=binrange)
                    counts[animal][ptNames[p+1]][fields[c]] = counts[animal][ptNames[p+1]][fields[c]]\
                                                            / sum(counts[animal][ptNames[p+1]][fields[c]])
                    super_counts[ptNames[p+1]][fields[c]] += counts[animal][ptNames[p+1]][fields[c]] / len(animal_names)

                counts[animal]['all'][fields[c]], _ = np.histogram(np.mean(use,axis=0), bins, range=binrange)
                counts[animal]['all'][fields[c]] = counts[animal]['all'][fields[c]] / sum(counts[animal]['all'][fields[c]])
                super_counts['all'][fields[c]] += counts[animal]['all'][fields[c]] / len(animal_names)

            if not pixel_units:
                usedbins = usedbins / pixels_per_cm
            binctrs = usedbins[1:]  # (usedbins[1:] + usedbins[:-1]) / 2.0

        return counts, super_counts, binctrs
    
    def _plot_model_PCK(self, res_folder, performance, pose_model_names=None, xlim=None, pixel_units=False,
                   combine_animals=False, print_PCK_values=False, custom_PCK_values=None):

        ptNames = ['all'] + ["nose tip", "right ear", "left ear", "neck", "right side body", "left side body", "tail base"]
        pix_per_cm = 37.7

        if not pix_per_cm:
            pixel_units=True
        elif xlim and not pixel_units:  # assume xlim was provided in cm; PCK computations are always done in pixels.
            xlim = [x * pix_per_cm for x in xlim]

        nKpts = len(ptNames)
        fig, ax = plt.subplots(math.ceil(nKpts / 4), 4, figsize=(15, 4 * math.ceil(nKpts / 4)))
        thr = 0.85
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
        # ax = ax.flatten()
        cnum = -1
        pose_model_names = ['top']
        for model in pose_model_names:
            animal_names = ['black', 'white']
            counts_hu, super_counts_hu, binctrs_hu = self._compute_human_PCK(animal_names=animal_names,
                                                                    xlim=xlim, pixel_units=pixel_units)
            if not xlim:
                delta = (binctrs_hu[1] - binctrs_hu[0]) / 2
                xlim_hu = [binctrs_hu[0] - delta, binctrs_hu[-1] + delta]
            else:
                xlim_hu = xlim

            counts_model = {n: [] for n in ptNames}
            for i, usePart in enumerate(ptNames):
                counts_model[usePart], binctrs_model = self._compute_model_pck(performance[model][usePart],
                                                                        lims=xlim_hu, pixels_per_cm=pix_per_cm,
                                                                        pixel_units=pixel_units)

            if not pixel_units and xlim:
                    xlim = [x / pix_per_cm for x in xlim]

            if combine_animals:
                cutoff = 0
                for p, pt in enumerate(ptNames):
                    objs = ax[int(p / 4), p % 4].stackplot(binctrs_hu, super_counts_hu[pt]['max'].cumsum(),
                                                        (super_counts_hu[pt]['min'].cumsum() -
                                                            super_counts_hu[pt]['max'].cumsum()),
                                                        color=colors[0], alpha=0.25)
                    objs[0].set_alpha(0)
                    ax[int(p / 4), p % 4].plot(binctrs_hu, super_counts_hu[pt]['med'].cumsum(),
                                            '--', color=colors[0], label='median (human)')

                    cutoff = max(cutoff, sum((super_counts_hu[pt]['med'].cumsum()) < thr))
            else:
                for animal in animal_names:
                    cnum+=1
                    cutoff = 0
                    for p, pt in enumerate(ptNames):
                        objs = ax[int(p / 4), p % 4].stackplot(binctrs_hu, counts_hu[animal][pt]['max'].cumsum(),
                                                            (counts_hu[animal][pt]['min'].cumsum() -
                                                                counts_hu[animal][pt]['max'].cumsum()),
                                                            color=colors[cnum], alpha=0.25)
                        objs[0].set_alpha(0)
                        ax[int(p / 4), p % 4].plot(binctrs_hu, counts_hu[animal][pt]['med'].cumsum(),
                                                '--', color=colors[cnum], label=animal + ' median (human)')

                        cutoff = max(cutoff, sum((counts_hu[animal][pt]['med'].cumsum()) < thr))
            for p, label in enumerate(ptNames):
                ax[int(p / 4), p % 4].plot(binctrs_model, counts_model[label].cumsum(),
                                        'k-', label='model')
                ax[int(p / 4), p % 4].set_title(label)
                xlim = xlim if xlim is not None else [0, binctrs_hu[cutoff]]
                ax[int(p / 4), p % 4].set_xlim(xlim)

                if print_PCK_values or custom_PCK_values is not None:
                    if custom_PCK_values is not None:
                        if not isinstance(custom_PCK_values,list):
                            custom_PCK_values  = [custom_PCK_values]
                        for val in custom_PCK_values:
                            if val > 1.0:
                                print('Please pass custom PCK thresholds as values between 0 and 1.')
                            med = binctrs_model[sum((counts_model[label].cumsum()) < val)]
                            print(label + ' ' + str(val*100.) + '%: ' + "{:.3f}".format(med))
                    else:
                        med = binctrs_model[sum((counts_model[label].cumsum()) < 0.5)]
                        print(label + ' 50%: ' + "{:.3f}".format(med))
                        med = binctrs_model[sum((counts_model[label].cumsum()) < 0.9)]
                        print(label + ' 90%: ' + "{:.3f}".format(med))
                    print('')

            ax[int(p / 4), p % 4].legend()

            fig.add_subplot(111, frameon=False)
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.grid(False)
            plt.ylabel('percent correct keypoints')
            if not pixel_units:
                plt.xlabel('error radius (cm)')
            else:
                plt.xlabel('error radius (pixels)')
            plt.show()
            fig.savefig(os.path.join(res_folder, 'PCK_curves.pdf'), bbox_inches='tight')

    def _coco_eval(self, res_folder, pose_model_names=None, view='top', fixedSigma=None):
        parts = ["nose tip", "right ear", "left ear", "neck", "right side body", "left side body", "tail base"]

        if not pose_model_names:
            pose_model_list = ['top']
            pose_model_names = ['top']

        # Parse things for COCO evaluation.
        savedEvals = {n: {} for n in pose_model_names}

        for model in pose_model_names:

            infile = os.path.join(res_folder,'MARS_format.json')
            with open(infile) as jsonfile:
                cocodata = json.load(jsonfile)
            gt_keypoints = cocodata['gt_keypoints']
            pred_keypoints = cocodata['pred_keypoints']
            # pprint(gt_keypoints)
            # pprint(pred_keypoints)

            for partNum in range(len(parts) + 1):

                MARS_gt = copy.deepcopy(gt_keypoints)
                MARS_gt['annotations'] = [d for d in MARS_gt['annotations'] if d['category_id'] == (partNum + 1)]
                MARS_pred = [d for d in pred_keypoints if d['category_id'] == (partNum + 1)]
                # print('partNum', partNum)
                # print(MARS_gt['annotations'][:5])
                # print(MARS_pred[:5])

                gt_coco = COCO()
                gt_coco.dataset = MARS_gt
                gt_coco.createIndex()
                pred_coco = gt_coco.loadRes(MARS_pred)
                # Actually perform the evaluation.
                part = [parts[partNum - 1]] if partNum else parts
                
                if 1 == 1 or view.lower() == 'top':
                    print('Top')
                    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_top', useParts=part)
                elif fixedSigma:
                    print('Fixed Sigma')
                    assert fixedSigma in ['narrow', 'moderate', 'wide', 'ultrawide']
                    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='fixed', useParts=[fixedSigma])
                elif not view:
                    print('Not view')
                    # print('warning: camera view not specified, defaulting to evaluation using fixedSigma=narrow')
                    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='fixed', useParts=['narrow'])
                elif view.lower() == 'front':
                    print('Front')
                    cocoEval = COCOeval(gt_coco, pred_coco, iouType='keypoints', sigmaType='MARS_front', useParts=part)
                else:
                    raise ValueError('Something went wrong.')

                cocoEval.evaluate()
                cocoEval.accumulate()
                partstr = part[0] if partNum else 'all'
                savedEvals[model][partstr] = cocoEval

        return savedEvals
    
    def _format_to_MARS_Developer_dict_format(self, res_file, res_folder):
        infile = self._get_ann_file_keypoint()
        data = {}
        data['partNames'] = ['nose tip', 'right ear', 'left ear', 'neck', 'right side body', 'left side body', 'tail base']
        with open(infile, 'r') as f:
            gt_data = json.load(f)
        with open(res_file, 'r') as f:
            pred_data = json.load(f)
        data['pred_keypoints'] = []
        for frame in pred_data:
            category_id = 2
            for i in range(0, len(frame['keypoints']), 3):
                annot = {
                    'image_id': int(frame['image_id']),
                    'keypoints': [frame['keypoints'][i], frame['keypoints'][i+1], 1],
                    'score': frame['keypoints'][i+2],
                    'category_id': category_id
                }
                data['pred_keypoints'].append(annot)
                category_id += 1
            annot = {
                'image_id': int(frame['image_id']),
                'keypoints': [1 if i % 3 == 2 else frame['keypoints'][i] for i in range(len(frame['keypoints']))],
                'score': frame['score'],
                'category_id': 1
            }
            data['pred_keypoints'].append(annot)
        data['gt_keypoints'] = {}
        data['gt_keypoints']['categories'] = [{'id': 1}, {'id': 2}, {'id': 3}, {'id': 4}, {'id': 5}, {'id': 6}, {'id': 7}, {'id': 8}]
        data['gt_keypoints']['annotations'] = []
        data['gt_keypoints']['images'] = []
        for frame in gt_data['annotations']:
            category_id = 2
            for i in range(0, len(frame['keypoints']), 3):
                data['gt_keypoints']['annotations'].append(
                    {
                        'area': frame['area'],
                        'bbox': frame['bbox'],
                        'image_id': int(frame['image_id']),
                        'category_id': category_id,
                        'id': frame['id'],
                        'iscrowd': 0,
                        'keypoints': [frame['keypoints'][i], frame['keypoints'][i+1], 1],
                        'num_keypoints': 1
                    }
                )
                category_id += 1
            data['gt_keypoints']['annotations'].append(
                {
                    'area': frame['area'],
                    'bbox': frame['bbox'],
                    'image_id': int(frame['image_id']),
                    'category_id': 1,
                    'id': frame['id'],
                    'iscrowd': 0,
                    'keypoints': [1 if i % 3 == 2 else frame['keypoints'][i] for i in range(len(frame['keypoints']))],
                    'num_keypoints': 7
                }
            )
            data['gt_keypoints']['images'].append({'id': int(frame['image_id'])})
        with open(os.path.join(res_folder, 'MARS_format.json'), 'w') as f:
            json.dump(data, f)
        return data
