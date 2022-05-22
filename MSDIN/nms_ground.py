import torch
import time as time
from data import *
import numpy as np

def box_iou(boxes1):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1,  x2) format.
    Arguments:
        boxes1 (Tensor[B，N, 2])
    Returns:
        iou (Tensor[B,N, N]): the NxN matrix containing the pairwise
            IoU values for every element in boxes1 and boxes1
    """

    def box_area(box):
        # box = 2xn
        return torch.abs(box[:,:,1] - box[:,:,0])

    area1 = box_area(boxes1)

    lt = torch.max(boxes1[:, None, :,0], boxes1[:, :,None,0])  # [N,M]
    rb = torch.min(boxes1[:, None, :,1], boxes1[:, :,None,1])  # [N,M]

    inter = (rb - lt).clamp(min=0) # [N,M]
    return inter / (area1[:, :,None] + area1[:,None,:] - inter)  # iou = inter / (area1 + area2 - inter)

def decode(loc, priors, variance):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [B,num_priors,2]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [B,num_priors,2].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        (priors[:, 0] + loc[:,:, 0] * variance[0] * priors[:, 1]).unsqueeze(2),
        (priors[:, 1] * torch.exp(loc[:,:, 1] * variance[1])).unsqueeze(2)), 2)

    #point_form(boxes)
    boxes[:,:, 0] -= boxes[:,:, 1] / 2
    boxes[:,:, 1] += boxes[:,:,0]
    return boxes

def decode_result(out, NMS_thre = 0.3,score_thre = 0.3,variance = [0.1,0.2],top_kN = 100):

    pred_box_new , prior_score , priors = out
    Soft_max = torch.nn.Softmax(dim=-1)
    prior_score = Soft_max(prior_score)[:,:,1:]
    pred_box = decode(pred_box_new, priors, variance)
    pred_out = []
    Batch = pred_box_new.size(0)
    [max_prior_score, max_prior_class] = prior_score.max(dim=2)
    # 计算top_k的先验框

    [top_k_score, top_index] = max_prior_score.topk(k=top_kN, dim=1)

    # gather函数，计算top_k对应的box,可以扩展top_index一起得到box_start,box_end
    top_k_class = max_prior_class.gather(dim=1, index=top_index)
    box_start = pred_box[:, :, 0].gather(dim=1, index=top_index)
    box_end = pred_box[:, :, 1].gather(dim=1, index=top_index)
    boxes = torch.stack([box_start, box_end], dim=2)

    # 计算IoU
    Iou_out = box_iou(boxes)
    Iou_out.triu_(diagonal=1)
    # 得到IoU保留框,展开为1维的防止读取时候发生重叠
    IoU_keep = Iou_out.max(dim=1)[0] < NMS_thre
    thre_keep = top_k_score > score_thre
    IoU_keep = IoU_keep & thre_keep

    for bs in range(Batch):
        box_keep = boxes[bs,IoU_keep[bs,:]].view(-1,2)
        class_keep = top_k_class[bs,IoU_keep[bs,:]].float().view(-1,1)
        bs_pred = torch.cat((box_keep,class_keep),dim = 1).detach().cpu().numpy()
        pred_out.append(bs_pred)
    return pred_out

def decode_result_all_class(out, NMS_thre = 0.5,score_thre = 0.3,variance = [0.1,0.2],top_kN = 100):

    pred_box_new , prior_score , priors = out

    Soft_max = torch.nn.Softmax(dim=-1)
    prior_score = Soft_max(prior_score)[:,:,1:]
    pred_box = decode(pred_box_new, priors, variance)

    Batch = pred_box_new.size(0)
    pred_out = [list() for i in range(Batch)]
    for class_i in range(prior_score.size(2)):
        max_prior_score = prior_score [:,:,class_i]
        # 计算top_k的先验框
        [top_k_score, top_index] = max_prior_score.topk(k=top_kN, dim=1)

        # gather函数，计算top_k对应的box,可以扩展top_index一起得到box_start,box_end

        box_start = pred_box[:, :, 0].gather(dim=1, index=top_index)
        box_end = pred_box[:, :, 1].gather(dim=1, index=top_index)
        boxes = torch.stack([box_start, box_end], dim=2)

        # 计算IoU
        Iou_out = box_iou(boxes)
        Iou_out.triu_(diagonal=1)
        # 得到IoU保留框,展开为1维的防止读取时候发生重叠
        IoU_keep = Iou_out.max(dim=1)[0] < NMS_thre
        thre_keep = top_k_score > score_thre
        IoU_keep = IoU_keep & thre_keep
        for bs in range(Batch):
            box_keep = boxes[bs,IoU_keep[bs,:]].view(-1,2)
            if box_keep.size(0) != 0 :
                class_keep = class_i * torch.ones((box_keep.size(0),1))
                bs_pred = torch.cat((box_keep,class_keep),dim = 1).detach().cpu().numpy()
                if not len(pred_out[bs]):
                    pred_out[bs] = bs_pred
                else :
                    pred_out[bs] = np.concatenate((pred_out[bs],bs_pred),axis= 0)

    return pred_out

def iou(gt, loc, th):
    loc_min = max(loc[0], gt[0])
    loc_max = min(loc[1], gt[1])
    inter = loc_max - loc_min
    iou_ = inter / (loc[1] - loc[0] + gt[1] - gt[0] - inter)
    return True if iou_ > th else False

def match_result(pred,ground_truth ,settings,NMS_thre = 0.3,):
#         matching
    num_signal = cfg['num_classes'] - 1
    confu_matrix = np.zeros((num_signal+1, num_signal+1))
    Batch = len(pred)
    for bs in range(Batch):
        gt_loc = ground_truth[bs][:, :2].detach().cpu().numpy()
        gt_classes = ground_truth[bs][:, 2].detach().cpu().numpy()
        count_j = 0
        det = [False] * len(gt_classes)
        #         matching
        len_gt = len(gt_classes)

        # Prediction exists
        if len(pred[bs]):

            locs = pred[bs][:,:2]
            classes = pred[bs][:,2]
            count_j = 0

            len_bbox = len(locs)

    #         ground boxes match with bounding boxes
            for i in range(len_bbox):
                pred_loc = locs[i]
                pred_classes = int(classes[i])

                match = 0
                flag = False
                #匹配
                for j in range(len_gt):
                    if iou(gt_loc[j], pred_loc, NMS_thre):
                        confu_matrix[pred_classes][int(gt_classes[j])] += 1
                        flag = True
                        det[j] = True
                        break
                #误检，寻找不到匹配的gt
                if flag == False:
                    confu_matrix[pred_classes][num_signal] += 1
            #未被匹配的目标信号漏检
            for i in range(len_gt):
                if not det[i]:
                    confu_matrix[num_signal][int(gt_classes[i])] += 1
        else:
            # 漏检
            for i in range(len_gt):
                confu_matrix[num_signal][int(gt_classes[i])] += 1
    return confu_matrix

def compute_p_r(matrix,nums):

    Num_class = len(nums)-1
    det_p = np.zeros((Num_class, 1))
    acc_p = np.zeros((Num_class, 1))
    # nums = np.sum(matrix,axis=0)
    for i in range(Num_class):
        if nums[i]==0:
            nums[i]=nums[i]+1
        sum_pred = np.sum(matrix[i, :])
        sum_det = np.sum(matrix[:, i])
        # 假设共100个AM，检测结果为２００个，其中９０个是准确的，检测概率０.９，准确率　０.４５
        # 检测概率
        det_p[i] = matrix[i][i] / (sum_det +1e-6)
        # 检测准确概率
        acc_p[i] = matrix[i][i] / (sum_pred + 1e-6)
    return det_p,acc_p


if __name__ == '__main__':
    NMS_thre = 0.2
    score_thre = 0.3
    # 先验20类目标，去除了背景类，700个候选框，prior_score 先验框的score,pred_box偏移值
    Batch= 1024
    N_class = 20
    N_prio = 700
    #需要NMS的类别index_useful
    index_useful = torch.arange(0,20)
    #每个batch间隔batch_gap = 1（代表20KHz）
    batch_gap = torch.arange(0,Batch)
    #预测概率
    prior_score = torch.rand((Batch,N_class,N_prio))
    #预测位置
    pred_box_new = torch.randn((Batch,N_prio,2))
    #预选框坐标
    priors = torch.randn((N_prio,2))

    variance = [0.1,0.2]
    #提起计算好的先验框Iou
    IoU_maxtrix = torch.rand((N_prio*N_prio))
    if False:
        prior_score = prior_score.cuda()
        IoU_maxtrix = IoU_maxtrix.cuda()
        pred_box_new = pred_box_new.cuda()
        priors = priors.cuda()
        batch_gap = batch_gap.cuda()
        index_useful = index_useful.cuda()
    top_kN = 50

    # 直接读取80ms
    t1 = time.time()
    pred_box = decode(pred_box_new,priors,variance)
    #读取预选的矩阵
    for i in range(20):
        #保留概率最大类，也可以每一类单独做时间和类别成正比
        [max_prior_score,max_prior_class] = prior_score.max(dim=1)
        # max_prior_score =  prior_score[:,index_useful[i],:]
        # max_prior_class = index_useful.repeat([Batch,N_prio])
        # 计算top_k的先验框
        [top_k_score,top_index] = max_prior_score.topk(k=top_kN,dim=1)

        # gather函数，计算top_k对应的box,可以扩展top_index一起得到box_start,box_end
        top_k_class = max_prior_class.gather(dim=1, index=top_index)
        # top_k_class = max_prior_class.gather(dim=1, index=top_index).view(Batch * top_kN)
    #按照batch 计算相应坐标
        box_start = (pred_box[:, :, 0].view(Batch, N_prio).gather(dim=1, index=top_index) + batch_gap.unsqueeze(dim =1))
        box_end = (pred_box[:, :, 1].view(Batch, N_prio).gather(dim=1, index=top_index) + batch_gap.unsqueeze(dim =1))
    #展开为1维的方便读取
        # box_start = pred_box[:, :, 0].view(Batch, N_prio).gather(dim=1, index=top_index).view(Batch*top_kN)
        # box_end = pred_box[:, :, 1].view(Batch, N_prio).gather(dim=1, index=top_index).view(Batch*top_kN)
        #重新排列index,转换成一维的计算，计算top_index,meshgrid
        top_index_new = top_index.unsqueeze(2)+top_index.unsqueeze(1) *N_prio

        # 按照IoU_maxtrix读取相应数据
        Iou_out = IoU_maxtrix.take(index = top_index_new).reshape(Batch,top_kN,top_kN)
        Iou_out.triu_(diagonal=1)
        # 得到IoU保留框,展开为1维的防止读取时候发生重叠
        IoU_keep= Iou_out.max(dim=1)[0] < NMS_thre
        thre_keep = top_k_score > score_thre

        # IoU_keep= Iou_out.max(dim=1)[0].view(Batch*top_kN) < NMS_thre
        # thre_keep = top_k_score.view(Batch*top_kN) > score_thre

        IoU_keep = IoU_keep & thre_keep
        box_start_keep = box_start[IoU_keep]
        box_end_keep= box_end[IoU_keep]

        class_keep = top_k_class[IoU_keep]

    t2 = time.time()
    print(t2-t1)


    # 计算IOU,20次求最大类，80ms
    t1 = time.time()
    pred_box = decode(pred_box_new, priors, variance)
    for i in range(20):
        #计时器
        #保留概率最大类，也可以每一类单独做时间和类别成正比
        [max_prior_score,max_prior_class] = prior_score.max(dim=1)
        # 计算top_k的先验框
        [top_k_score,top_index] = max_prior_score.topk(k=top_kN,dim=1)

        # gather函数，计算top_k对应的box,可以扩展top_index一起得到box_start,box_end
        top_k_class = max_prior_class.gather(dim=1, index=top_index)
        box_start = pred_box[:, :, 0].view(Batch, N_prio).gather(dim=1, index=top_index)+ batch_gap.unsqueeze(dim =1)
        box_end = pred_box[:, :, 1].view(Batch, N_prio).gather(dim=1, index=top_index)+ batch_gap.unsqueeze(dim =1)
        boxes = torch.cat((box_start,box_end),dim=1).view(Batch,top_kN,2)

        # 计算IoU
        Iou_out = box_iou(boxes)
        Iou_out.triu_(diagonal=1)
        # 得到IoU保留框,展开为1维的防止读取时候发生重叠
        IoU_keep= Iou_out.max(dim=1)[0] <NMS_thre
        thre_keep = top_k_score > score_thre
        IoU_keep = IoU_keep & thre_keep

        box_start_keep = box_start[IoU_keep]
        box_end_keep= box_end[IoU_keep]
        class_keep = top_k_class[IoU_keep]

    t2 = time.time()
    print(t2-t1)

    # 计算IOU,按照每一类计算，总共70ms
    t1 = time.time()
    pred_box = decode(pred_box_new, priors, variance)
    for i in range(20):
        #计时器
        #可以每一类单独做时间和类别成正比
        max_prior_score =  prior_score[:,index_useful[i],:]
        # 计算top_k的先验框
        [top_k_score,top_index] = max_prior_score.topk(k=top_kN,dim=1)

        # gather函数，计算top_k对应的box,可以扩展top_index一起得到box_start,box_end
        box_start = pred_box[:, :, 0].gather(dim=1, index=top_index) + batch_gap.unsqueeze(dim =1)
        box_end = pred_box[:, :, 1].gather(dim=1, index=top_index) + batch_gap.unsqueeze(dim =1)
        boxes = torch.cat((box_start,box_end),dim=1).view(Batch,top_kN,2)

        # 计算IoU
        Iou_out = box_iou(boxes)
        Iou_out.triu_(diagonal=1)
        # 得到IoU保留框,展开为1维的防止读取时候发生重叠
        IoU_keep= Iou_out.max(dim=1)[0] <NMS_thre
        thre_keep = top_k_score > score_thre
        IoU_keep = IoU_keep & thre_keep

        box_start_keep = box_start[IoU_keep]
        box_end_keep= box_end[IoU_keep]
        class_keep = index_useful[i]

    t2 = time.time()
    print(t2-t1)
    # 计算IoU时间5ms