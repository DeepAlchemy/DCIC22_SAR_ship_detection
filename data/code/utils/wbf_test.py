import pandas as pd
import os
from tqdm import tqdm
from ensemble_boxes import *
import csv
import numpy as np


iou_thr = 0.5
skip_box_thr = 0.01


def rescale(coord, size=None):
    """
    rescale xyxy to [0,1]
    :return: xyxy
    """
    if size is None:
        size = 256
    res = []
    for i in coord:
        i /= size
        res.append(i)

    return res


def wbf(testdata_path, models_path, out_path):
    img_ids = os.listdir(testdata_path)
    count7 = 0
    count8 = 0
    count9 = 0
    countlow = 0
    for img_id in tqdm(img_ids, total=len(img_ids)):

        boxes_list = []
        scores_list = []
        labels_list = []
        #weight = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        weights = []
        models = os.listdir(models_path)
        models.sort()
        mp = {}

        # for index, model in enumerate(models):
        #     mp[model] = weight[index]
        img_id = img_id.replace('jpg', 'txt')
        index = 1
        for model in models:
            box_list = []
            score_list = []
            label_list = []
            model_path = os.path.join(models_path, model)  # /Users/kevin/Downloads/wbf_models/0.9570
            txt_file = os.path.join(model_path, img_id)  # /Users/kevin/Downloads/wbf_models/0.9570/xxxxx.txt
            if os.path.exists(txt_file):
                try:
                    txt_df = pd.read_csv(txt_file, header=None, sep=' ').values
                except:  # 防止csv为空
                    continue
                for row in txt_df:
                    if row[0] == 0 and len(row) == 6:
                        box_list.append(rescale(row[1:5]))
                        label_list.append(int(row[0]))
                        score_list.append(row[-1])
                    elif row[0] == 1 and len(row) == 6:
                        continue
                    else:
                        box_list.append(rescale(row[0:4]))
                        score_list.append(row[-1])
                        label_list.append(0)
                boxes_list.append(box_list)
                scores_list.append(score_list)
                labels_list.append(label_list)
                #weights.append(float(model))  # model名必须是submission的得分
                weights.append(index)
                #weights.append(mp[model])
                index += 1
            else:
                continue

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')


        boxes_temp = []
        scores_temp = []
        for i in range(len(boxes)):
            if scores[i] >= 0.7:
                boxes_temp.append(boxes[i])
                scores_temp.append(scores[i])
        boxes = boxes_temp
        scores = scores_temp
        for score in scores:
            if score >= 0.9:
                count9 += 1
            elif score >= 0.8:
                count8 += 1
            elif score >= 0.7:
                count7 += 1
            else:
                countlow += 1
        # write result into new file
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        with open("{}/{}".format(out_path, img_id.replace('jpg', 'txt')), "w") as out:
            writer = csv.writer(out)
            writer.writerows(boxes)

if __name__ == '__main__':
    # cur_path = os.getcwd()
    # models_path = os.path.join(cur_path, "../../user_data/intermediate_results/output_models")
    # testdata_path = os.path.join(cur_path, "../../raw_data/test")
    # out_path = os.path.join(cur_path, "../../user_data/intermediate_results/wbf_output")
    models_path = '/data/user_data/intermediate_results/output_models'
    testdata_path = '/data/user_data/dcic_full_coco_dataset/test'
    out_path = '/data/user_data/intermediate_results/wbf_output'
    wbf(testdata_path, models_path, out_path)
