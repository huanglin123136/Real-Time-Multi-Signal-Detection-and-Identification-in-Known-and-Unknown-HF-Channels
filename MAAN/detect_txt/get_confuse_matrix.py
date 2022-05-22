import os
import numpy as np


def iou(gt, loc, th):
    loc_min = max(loc[0], gt[0])
    loc_max = min(loc[1], gt[1])
    inter = loc_max - loc_min
    iou_ = inter / (loc[1] - loc[0] + gt[1] - gt[0] - inter)
    return True if iou_ > th else False


def get_num(string):
    string_ = []
    for s in string:
        string_.append(s[7:-1])
    return string_


def read_txt(path):
    with open(path, 'r') as f:
        data = f.read()
    return data


def eval_result(res, th, th_conf, num_signal):
    blocks = res.split('Ground')[1:]
    matrix = np.zeros((num_signal + 1, num_signal + 1))     # confuse matrix
    for block in blocks:
        # Prediction exists
        if block.find('Prediction') != -1:
            gt_blocks = block.split('Prediction')[0]
            gt_lines = gt_blocks.split('\n')[1:-1]
            gt_lines = [x.strip() for x in gt_lines]
            gt = []
            for line in gt_lines:
                gt_label = {}
                line = line.split(' ')[1]
                label = line.split('||')
                gt_label['label'] = int(label[-1].split('.')[0])
                gt_label['loc'] = np.array(label[:-1], dtype=np.float)
                gt.append(gt_label)
            bbox_blocks = block.split('Prediction:')[1]
            bbox_lines = bbox_blocks.split('\n')[1:-2]

            scores = []
            locs = []
            classes = []

            for line in bbox_lines:

                if line.find('AM') != -1:
                    classes.append(0)
                elif line.find('SSB') != -1:
                    classes.append(1)
                elif line.find('PSK') != -1:
                    classes.append(2)
                elif line.find('2FSK') != -1:
                    classes.append(3)
                elif line.find('CW') != -1:
                    classes.append(4)
                elif line.find('Saopin') != -1:
                    classes.append(5)
                elif line.find('Interp') != -1:
                    classes.append(6)
                elif line.find('SingleSound') != -1:
                    classes.append(7)
                elif line.find('Amexpandlarge') != -1:
                    classes.append(8)
                elif line.find('Interp_flash') != -1:
                    classes.append(9)
                elif line.find('Unknow') != -1:
                    classes.append(10)
                elif line.find('Saolarge') != -1:
                    classes.append(11)
                elif line.find('Noise') != -1:
                    classes.append(12)
                elif line.find('Cfast') != -1:
                    classes.append(13)
                elif line.find('None1') != -1:
                    classes.append(14)
                elif line.find('None2') != -1:
                    classes.append(15)
                elif line.find('None3') != -1:
                    classes.append(16)
                elif line.find('None4') != -1:
                    classes.append(17)
                elif line.find('None5') != -1:
                    classes.append(18)
                elif line.find('None6') != -1:
                    classes.append(19)
                else:
                    classes.append(20)

                bbox_line = line.split(':')[-1].split(' ')
                score = float(bbox_line[-2])
                scores.append(score)
                loc = np.array(bbox_line[-1].split('||'), dtype=np.float)
                locs.append(loc)

    #         ground boxes match with bounding boxes
            det = [False] * len(gt)
            scores = np.array(scores)
            sort_index = np.argsort(-scores)
            sorted_scores = [scores[x] for x in sort_index]
            sorted_locs = [locs[x] for x in sort_index]
            sorted_classes = [classes[x] for x in sort_index]

            pivor = 0
            for ii in range(len(sort_index)):
                if sorted_scores[ii] < th_conf:
                    break
                else:
                    pivor += 1
            sort_index = sort_index[:pivor]
            sorted_locs = sorted_locs[:pivor]
            sorted_classes = sorted_classes[:pivor]
            matched_gt = 0

    #         matching
            len_bbox = len(sort_index)
            len_gt = len(gt)
            for i in range(len_bbox):
                pred_loc = sorted_locs[i]
                pred_classes = sorted_classes[i]

                match = 0

                flag = False

                for j in range(len_gt):
                    if iou(gt[j]['loc'], pred_loc, th):
                        matrix[pred_classes][gt[j]['label']] += 1
                        if pred_classes == gt[j]['label']:
                            det[j] = True
                        flag = True
                        break


            for i in range(len_gt):
                if not det[i]:
                    matrix[num_signal][gt[i]['label']] += 1
        else:
            # miss detect
            gt_blocks = block.split('label')[1:]
            for line in gt_blocks:
                gt_label = int(line.split('||')[-1].split('\n')[0].split('.')[0])
                matrix[num_signal][gt_label] += 1

    return matrix


def compute_p_r(matrix, nums):
    Num_class = len(nums)
    res = np.zeros((Num_class+1, 2))

    for i in range(Num_class):
        if nums[i]==0:
            nums[i]=nums[i]+1
        sum_pred = np.sum(matrix[i, :])
        res[i][0] = matrix[i][i] / nums[i]
        res[i][1] = matrix[i][i] / (sum_pred + 1e-6)
    return res


def main(path, th, signal_num):
    txt = read_txt(path)
    Num_class = len(signal_num)
    PR = np.zeros((9,Num_class+1, 2))
    # 置信度0.1-0.9
    for i in range(1, 10):
        th_conf = i / 10
        confuse_matrix = eval_result(txt, th, th_conf, len(signal_num))
        res = compute_p_r(np.array(confuse_matrix), signal_num)
        PR[i - 1] = res

    confuse_matrix = eval_result(txt, th, 0.3, len(signal_num))
    return PR,confuse_matrix


if __name__ == '__main__':
    path = './test_1124.txt'
    th = 0.5
    th_conf = 0.1
    txt = read_txt(path)
    confuse_matrix = eval_result(txt, th, th_conf)
    res = compute_p_r(np.array(confuse_matrix), [641, 115, 48])
