import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp
# from box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self, num_classes, overlap_threshold, prior_for_matchingm, bkg_label, neg_mining,
                 neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.num_calsses = num_classes
        self.threshold = overlap_threshold
        self.using_prior_for_matching = prior_for_matchingm
        self.background_label = bkg_label
        self.do_neg_ming = neg_mining
        self.neg_pos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.encode_target = encode_target
        self.use_gpu = use_gpu
        self.variance = [0.1, 0.2]
        # self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,2)
                priors shape: torch.size(num_priors,2)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,3] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        batch_size = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, num_priors, 2)
        conf_t = torch.Tensor(batch_size, num_priors)
        for idx in range(batch_size):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance,
                  labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,2]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_t = loc_t[pos_idx].view(-1, 2)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_calsses)
        # gath = torch.clamp(torch.round(conf_t.view(-1, 1)), min=0, max=2).long() #using for debug
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1).long())

        # Hard Negative Mining
        loss_c = loss_c.view(batch_size, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(num_pos*self.neg_pos_ratio, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_calsses)
        target_weighted = conf_t[(pos+neg).gt(0)].long()
        # target_weighted = torch.zeros(conf_p.shape).scatter_(1, target_weighted.unsqueeze(1), 1)   #for binary classifier -> one hot
        # loss_conf = F.binary_cross_entropy_with_logits(conf_p, target_weighted, size_average=False)
        loss_conf = F.cross_entropy(conf_p, target_weighted, size_average=False)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N  = num_pos.data.sum().double()
        loss_loc = loss_loc.double()
        loss_conf = loss_conf.double()
        return loss_loc / N, loss_conf / N


class MultiBoxLossv2(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """
    def __init__(self, num_classes, overlap_threshold, prior_for_matchingm, bkg_label, neg_mining,
                 neg_pos, neg_overlap, encode_target, use_gpu=True):
        super(MultiBoxLossv2, self).__init__()
        self.num_calsses = num_classes
        self.threshold = overlap_threshold
        self.using_prior_for_matching = prior_for_matchingm
        self.background_label = bkg_label
        self.do_neg_ming = neg_mining
        self.neg_pos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.encode_target = encode_target
        self.use_gpu = use_gpu
        self.variance = [0.1, 0.2]
        # self.variance = cfg['variance']

    def forward(self, predictions, targets , min):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,2)
                priors shape: torch.size(num_priors,2)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,3] (last idx is the label).
        """
        loc_data, conf_data, priors = predictions
        batch_size = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = priors.size(0)
        # if len(targets) != 0:
            # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(batch_size, num_priors, 2)
        conf_t = torch.Tensor(batch_size, num_priors)
        for idx in range(batch_size):
            target = targets[idx]
            if len(target) == 0:
                conf_t[idx] = torch.zeros([num_priors])
            else:
                truths = target[:, :-1].data
                labels = target[:, -1].data
                defaults = priors.data
                match(self.threshold, truths, defaults, self.variance,
                      labels, loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,2]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 2)
        loc_t = loc_t[pos_idx].view(-1, 2)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_calsses)
        # gath = torch.clamp(torch.round(conf_t.view(-1, 1)), min=0, max=2).long() #using for debug
        a = batch_conf.gather(1, conf_t.view(-1, 1).long())
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1).long())


        # Hard Negative Mining
        loss_c = loss_c.view(batch_size, -1)
        loss_c[pos] = 0

        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        # num_pos = torch.clamp(num_pos, min=min)
        num_neg = torch.clamp(num_pos*self.neg_pos_ratio, min=min , max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_calsses)
        target_weighted = conf_t[(pos+neg).gt(0)].long()
        # target_weighted = torch.zeros(conf_p.shape).scatter_(1, target_weighted.unsqueeze(1), 1)   #for binary classifier -> one hot
        # loss_conf = F.binary_cross_entropy_with_logits(conf_p, target_weighted, size_average=False)
        loss_conf = F.cross_entropy(conf_p, target_weighted, size_average=False)
        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum().float()
        # N = num_pos.data.sum().double()
        # loss_loc = loss_loc.double()
        # loss_conf = loss_conf.double()
        return loss_loc / N, loss_conf / N


if __name__ == '__main__':
    num_classes = 8
    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, False)
    num = 4
    num_priors = 42
    num_objs = 6
    loc_data = torch.rand(num, num_priors, 2)
    conf_data = torch.rand(num, num_priors, num_classes)
    priors = torch.rand(num_priors, 2)
    out = (loc_data, conf_data, priors)
    targets = torch.rand(num, num_objs, 3)
    loss_l