import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

import torch
import torch.nn.functional as F
from utils.box_utils import decode, jaccard, index2d, crop, sanitize_coordinates 
from utils.config import cfg, mask_type
from utils.cython_nms import nms as cnms
from collections import defaultdict
import cv2


COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))
means = (123.68, 116.78, 103.94)
std = (58.40, 57.12, 57.38)
class_names = ["cp", "ALB", 'BET', 'BUM', 'DOL', 'LEC', 'MLS', 'OIL', 'SBT', 'SFA', 'SKJ', 'SSP', 'SWO', 'Shark', 'WAH', 'YFT']

class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.cross_class_nms = False
        self.use_fast_nms = False

    def __call__(self, predictions):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        batch_size = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

        for batch_idx in range(batch_size):
            decoded_boxes = decode(loc_data[batch_idx], prior_data)
            result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

            if result is not None and proto_data is not None:
                result['proto'] = proto_data[batch_idx]
            
            out.append(result)
        
        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if scores.size(1) == 0:
            return None
        
        if self.use_fast_nms:
            boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
        else:
            boxes, masks, classes, scores = self.traditional_nms(boxes, masks, scores, self.nms_thresh, self.conf_thresh)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}
    

    def coefficient_nms(self, coeffs, scores, cos_threshold=0.9, top_k=400):
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        coeffs_norm = F.normalize(coeffs[idx], dim=1)

        # Compute the pairwise cosine similarity between the coefficients
        cos_similarity = coeffs_norm @ coeffs_norm.t()
        
        # Zero out the lower triangle of the cosine similarity matrix and diagonal
        cos_similarity.triu_(diagonal=1)

        # Now that everything in the diagonal and below is zeroed out, if we take the max
        # of the cos similarity matrix along the columns, each column will represent the
        # maximum cosine similarity between this element and every element with a higher
        # score than this element.
        cos_max, _ = torch.max(cos_similarity, dim=0)

        # Now just filter out the ones higher than the threshold
        idx_out = idx[cos_max <= cos_threshold]


        # new_mask_norm = F.normalize(masks[idx_out], dim=1)
        # print(new_mask_norm[:5] @ new_mask_norm[:5].t())
        
        return idx_out, idx_out.size(0)

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def traditional_nms(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores

def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, score_threshold=0.3):
    """
    Note: If undo_transform=False then im_h and im_w are allowed to be None.
    """
    num_display = 1

    color_cache = defaultdict(lambda: {})
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy)
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    t = postprocess(dets_out, w, h, score_threshold = score_threshold)

    masks = t[3][:100]
    classes, scores, boxes = [x[:100].cpu().numpy() for x in t[:3]]

    vis_idx = classes != 0
    classes = classes[vis_idx]
    scores = scores[vis_idx]
    boxes = boxes[vis_idx]
    masks = masks[vis_idx]

    num_dets_to_consider = min(num_display, classes.shape[0])
    
    for j in range(num_dets_to_consider):
        if scores[j] < score_threshold:
            num_dets_to_consider = j
            break

    classes = classes[:num_display]
    scores = scores[:num_display]
    boxes = boxes[:num_display]
    masks = masks[:num_display]
    
    if num_dets_to_consider == 0:
        # No detections found so just output the original image
        return (img_gpu * 255).byte().cpu().numpy()

    # Quick and dirty lambda for selecting the color for a particular index
    # Also keeps track of a per-gpu color cache for maximum speed
    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                # The image might come in as RGB or BRG, depending
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    # First, draw the masks on the GPU where we can do it really fast
    # Beware: very fast but possibly unintelligible mask-drawing code ahead
    # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice

    # After this, mask is of size [num_dets, h, w, 1]
    masks = masks[:num_dets_to_consider, :, :, None]
    
    # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
    colors = torch.cat([(torch.Tensor(get_color(j)).float() / 255 ).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
    masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

    # This is 1 everywhere except for 1-mask_alpha where the mask is
    inv_alph_masks = masks * (-mask_alpha) + 1
    
    # I did the math for this on pen and paper. This whole block should be equivalent to:
    #    for j in range(num_dets_to_consider):
    #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
    masks_color_summand = masks_color[0]
    if num_dets_to_consider > 1:
        inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
        masks_color_cumul = masks_color[1:] * inv_alph_cumul
        masks_color_summand += masks_color_cumul.sum(dim=0)

    img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand
        
    # Then draw the stuff that needs to be done on the cpu
    # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
    img_numpy = (img_gpu * 255).byte().cpu().numpy()

    font_ratio = (h*w) / (1000*1000)
    
    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j)
        score = scores[j]

        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

        _class = class_names[classes[j]]
        text_str = '%s: %.2f' % (_class, score)

        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6 * font_ratio
        font_thickness = int(1 * font_ratio)

        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]

        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]

        cv2.rectangle(img_numpy, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_numpy, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_numpy, classes, scores

def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = (img_numpy * np.array(std) + np.array(means)) / 255.0
    img_numpy = np.clip(img_numpy, 0, 1)
    img_numpy = img_numpy[..., ::-1].copy()

    return cv2.resize(img_numpy, (w,h))

def postprocess(det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
                visualize_lincomb=False, crop_masks=True, score_threshold=0):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    
    dets = det_output[batch_idx]
    
    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    # At this points masks is only the coefficients
    proto_data = dets['proto']
    
    if visualize_lincomb:
        display_lincomb(proto_data, masks)

    masks = torch.matmul(proto_data, masks.t())
    masks = cfg.mask_proto_mask_activation(masks)

    # Crop masks before upsampling because you know why
    if crop_masks:
        masks = crop(masks, boxes)

    # Permute into the correct output shape [num_dets, proto_h, proto_w]
    masks = masks.permute(2, 0, 1).contiguous()
    
    masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

    # Binarize the masks
    masks.gt_(0.5)

    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.long()

    return classes, scores, boxes, masks