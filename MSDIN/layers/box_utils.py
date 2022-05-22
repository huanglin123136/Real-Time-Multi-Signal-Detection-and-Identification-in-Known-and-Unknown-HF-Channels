import torch
from torch.autograd import Variable
def piont_form(boxes):
    """ Convert prior_boxes to (xmin, xmax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, xmaxform of boxes.
    """

    return torch.cat((torch.unsqueeze(boxes[:, 0] - boxes[:, 1]/2, 1),
                      torch.unsqueeze(boxes[:, 0] + boxes[:, 1]/2, 1)), 1)

def center_form(boxes):
    """ Convert prior_boxes to (cx, s)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted cx, s form of boxes.
    """

    return torch.cat((torch.unsqueeze((boxes[:, 1] + boxes[:, 0])/2, 1),
                     torch.unsqueeze(boxes[:, 1] - boxes[:, 0], 1)), 1)

def intersection(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A] -> [A,1] -> [A,B]
    [B] -> [1,B] -> [A,B]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,2].
      box_b: (tensor) bounding boxes, Shape: [B,2].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """

    A = box_a.size(0)
    B = box_b.size(0)

    min_x = torch.max(
        box_a[:,0].unsqueeze(1).expand(A, B),
        box_b[:,0].unsqueeze(0).expand(A, B)
    )
    max_x = torch.min(
        box_a[:,1].unsqueeze(1).expand(A, B),
        box_b[:,1].unsqueeze(0).expand(A, B)
    )

    inter = torch.clamp(max_x - min_x, min=0)
    return inter

def Jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,2]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,2]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """

    inter = intersection(box_a, box_b)
    area_a = (box_a[:, 1] - box_a[:, 0]).unsqueeze(1).expand_as(inter)
    area_b = (box_b[:, 1] - box_b[:, 0]).unsqueeze(0).expand_as(inter)

    union = area_a + area_b - inter

    return inter / union

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when matching boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj,  2].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,2].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 2].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """

    overlap = Jaccard(truths, piont_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlap.max(1, keepdim=True)
    #[num_priors, 1] best gt box for each prior box
    best_gt_overlap, best_gt_idx = overlap.max(0, keepdim=True)

    best_prior_overlap.squeeze_(1)
    best_prior_idx.squeeze_(1)
    best_gt_overlap.squeeze_(0)
    best_gt_idx.squeeze_(0)
    # ensure best prior
    # set overlap between ground truth and its best matched prior box to 2, so it will be the best match of all time
    best_gt_overlap.index_fill_(0, best_prior_idx, 2)
    # ensure every gt matches with its prior of max overlap
    # accordingly, set idx of gt best matched
    for j in range(best_prior_idx.size(0)):
        best_gt_idx[best_prior_idx[j]] = j

    matches = truths[best_gt_idx]  #[num_priors, 2]
    conf = labels[best_gt_idx] + 1 #[num_priors]      #bck label = 0
    conf[best_gt_overlap < threshold] = 0

    loc = encode(matches, priors, variances)
    loc_t[idx] = loc                           #[num_priors, 2] encoded offsets for the net to learn
    conf_t[idx] = conf                         #[num_priors] gt label for each prior box

def encode(matches, priors, variance):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 2].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,2].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 2]
    """

    # dist b/t match center and prior's center
    g_x = (matches[:, 0] + matches[:, 1])/2 - priors[:, 0]
    # encode variance
    g_x /= ( variance[0] * priors[:, 1])
    # match s / prior s
    g_s = torch.log((matches[:, 1] - matches[:, 0]) / priors[:, 1])
    # encode variance
    g_s /= variance[1]
    # print(g_s.max())
    return torch.cat((g_x.unsqueeze(1), g_s.unsqueeze(1)), 1)  #[num_priors, 2] offsets of x and s

def decode(loc, priors, variance):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,2]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,2].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((
        (priors[:, 0] + loc[:, 0] * variance[0] * priors[:, 1]).unsqueeze(1),
        (priors[:, 1] * torch.exp(loc[:, 1] * variance[1])).unsqueeze(1)), 1)

    #point_form(boxes)
    boxes[:, 0] -= boxes[:, 1] / 2
    boxes[:, 1] += boxes[:, 0]
    return boxes

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    max_x = x.data.max()
    return torch.log(torch.sum(torch.exp(x - max_x), 1, keepdim=True)) + max_x

def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,2].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    x2 = boxes[:, 1]
    area = x2 - x1
    v, idx = scores.sort(0)

    idx = idx[-top_k:]
    xx1 = boxes.new()
    xx2 = boxes.new()
    s = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) ==1:
            break
        idx = idx[:-1]

        x1 = Variable(x1,requires_grad=False)
        x2 = Variable(x2, requires_grad=False)

        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(x2, 0, idx, out=xx2)
        xx1 = torch.clamp(xx1, min=x1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        s.resize_as_(xx1)
        s = xx2 - xx1
        s = torch.clamp(s, min=0)
        inter = s
        rem_area = torch.index_select(area, 0, idx)
        union = (rem_area - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count

if __name__ == '__main__':
    num = 4
    num_priors =42
    variance = [0.1, 0.2]
    targets = torch.rand(num, 6, 3)
    priors = torch.rand( num_priors, 2)
    loc_t = torch.Tensor(num, num_priors, 2)
    conf_t = torch.LongTensor(num, num_priors)
    for idx in range(num):
        truths = targets[idx][:, :-1].data
        labels = targets[idx][:, -1].data
        defaults = priors.data
        match(0.01, truths, defaults, variance, labels,
              loc_t, conf_t, idx)
    pos = conf_t > 0
    # Localization Loss (Smooth L1)
    # Shape: [batch,num_priors,2]
    pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_t)
    d1=pos+pos
    dd=d1.gt(0)

    tt





