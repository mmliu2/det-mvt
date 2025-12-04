import math
import torch
import torch.nn as nn
from collections import OrderedDict
from ltr.models.meta import steepestdescent
import ltr.models.target_classifier.linear_filter as target_clf
import ltr.models.target_classifier.features as clf_features
import ltr.models.target_classifier.initializer as clf_initializer
import ltr.models.target_classifier.optimizer as clf_optimizer
import ltr.models.bbreg as bbmodels
import ltr.models.backbone as backbones
from ltr import model_constructor
import numpy as np

def crop_and_resize(img, bbox, out_size, context_amount=0.5):
    """
    img: HxWx3 BGR or RGB image (numpy)
    bbox: [x, y, w, h] in the image
    out_size: size of output crop (e.g., 127 or 255)
    context_amount: how much padding around bbox (typical: 0.5)
    """

    x, y, w, h = bbox
    cx = x + w/2
    cy = y + h/2

    # Add contextual padding around the box
    context = context_amount * (w + h)
    size = np.sqrt((w + context) * (h + context))

    # Make square
    size = max(size, 1)
    half = size / 2

    # Crop coordinates
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    # Padding if ROI goes outside image
    im_h, im_w = img.shape[:2]
    left  = max(0, -x1)
    top   = max(0, -y1)
    right = max(0, x2 - im_w)
    bottom= max(0, y2 - im_h)

    # Pad and crop
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    x1 += left
    y1 += top
    x2 += left
    y2 += top

    crop = img_pad[y1:y2, x1:x2]

    # Resize to network input size
    crop = cv2.resize(crop, (out_size, out_size))

    # Convert to tensor CHW
    crop = torch.from_numpy(crop).permute(2,0,1).float() / 255.0
    return crop


def get_template_and_search(image, bbox):
    """
    image: HxWx3 numpy image
    bbox: [x,y,w,h] of target in this frame
    """
    template = crop_and_resize(image, bbox, out_size=127)  # template crop
    search   = crop_and_resize(image, bbox, out_size=255)  # search crop
    return template, search


class MVT_DeT(nn.Module):
    """The MVT network.
    args:
        feature_extractor:  Backbone feature extractor network. Must return a dict of feature maps
        classifier:  Target classification module.
        bb_regressor:  Bounding box regression module.
        classification_layer:  Name of the backbone feature layer to use for classification.
        bb_regressor_layer:  Names of the backbone layers to use for bounding box regression."""

    def __init__(self, feature_extractor, feature_extractor_depth, classifier, bb_regressor, classification_layer, bb_regressor_layer,
                   merge_type='mean', W_rgb=0.6 ,W_depth=0.4):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.feature_extractor_depth = feature_extractor_depth
        self.classifier = classifier
        self.bb_regressor = bb_regressor
        self.classification_layer = [classification_layer] if isinstance(classification_layer, str) else classification_layer
        self.bb_regressor_layer = bb_regressor_layer
        self.output_layers = sorted(list(set(self.classification_layer + self.bb_regressor_layer)))

        self.merge_type = merge_type
        # if self.merge_type == 'conv':
        #     self.merge_layer2 = nn.Conv2d(1024, 512, (1,1))
        #     self.merge_layer3 = nn.Conv2d(2048, 1024, (1,1))

        # self.id = 1

    def forward(self, train_imgs, test_imgs, train_bb, test_proposals, *args, **kwargs):
        """Runs the MVT network the way it is applied during training.
        The forward function is ONLY used for training. Call the individual functions during tracking.
        args:
            train_imgs:  Train image samples (images, sequences, 3, H, W).
            test_imgs:  Test image samples (images, sequences, 3, H, W).
            train_bb:  Target boxes (x,y,w,h) for the train images. Dims (images, sequences, 4).
            test_proposals:  Proposal boxes to use for the IoUNet (bb_regressor) module.
            *args, **kwargs:  These are passed to the classifier module.
        returns:
            test_scores:  Classification scores on the test samples.
            iou_pred:  Predicted IoU scores for the test_proposals."""

        assert train_imgs.dim() == 5 and test_imgs.dim() == 5, 'Expect 5 dimensional inputs'

        # template = train_imgs.reshape(-1, *train_imgs.shape[-3:])
        # search = test_imgs.reshape(-1, *test_imgs.shape[-3:])
        # train_feat, test_feat = self.extract_backbone_features(template, search) 
        train_feat = self.extract_backbone_features(train_imgs.reshape(-1, *train_imgs.shape[-3:]))
        test_feat = self.extract_backbone_features(test_imgs.reshape(-1, *test_imgs.shape[-3:]))

        # Classification features
        train_feat_clf = self.get_backbone_clf_feat(train_feat)
        test_feat_clf = self.get_backbone_clf_feat(test_feat)

        # Run classifier module
        target_scores = self.classifier(train_feat_clf, test_feat_clf, train_bb, *args, **kwargs)

        # Get bb_regressor features
        train_feat_iou = self.get_backbone_bbreg_feat(train_feat)
        test_feat_iou = self.get_backbone_bbreg_feat(test_feat)

        # Run the IoUNet module
        iou_pred = self.bb_regressor(train_feat_iou, test_feat_iou, train_bb, test_proposals)

        return target_scores, iou_pred

    def get_backbone_clf_feat(self, backbone_feat):
        feat = OrderedDict({l: backbone_feat[l] for l in self.classification_layer})
        if len(self.classification_layer) == 1:
            return feat[self.classification_layer[0]]

    def get_backbone_bbreg_feat(self, backbone_feat):
        return [backbone_feat[l] for l in self.bb_regressor_layer]              # Song : layer2 and layer 3

    def extract_classification_feat(self, backbone_feat):
        return self.classifier.extract_classification_feat(self.get_backbone_clf_feat(backbone_feat))

    def merge(self, color_feat, depth_feat, layers=None):

        feat = {}
        assert self.merge_type == 'max'
        assert len(layers) == 2
        
        layerA, layerB = layers
        feat[layerA] = torch.max(color_feat[layerA], depth_feat[layerA])
        feat[layerB] = torch.max(color_feat[layerB], depth_feat[layerB])

        return feat

    def extract_backbone_features(self, im, layers=None):

        if layers is None:
            layers = self.output_layers

        dims = im.shape
        # dims = template.shape
        if dims[1] == 6:

            color_feat = self.feature_extractor(im[:, :3, :, :], layers)
            depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)

            merged_feat = self.merge(color_feat, depth_feat, layers)
            # self.id += 1
            return merged_feat

        else:
            return self.feature_extractor(im, layers)

    def extract_features(self, im, layers=None):
        dims = im.shape
        if layers is None:
            layers = self.bb_regressor_layer + ['classification']
        if 'classification' not in layers:
            if dims[1] == 6:
                color_feat = self.feature_extractor(im[:, :3, :, :], layers)
                depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
                return self.merge(color_feat, depth_feat)
            else:
                return self.feature_extractor(im, layers)
        backbone_layers = sorted(list(set([l for l in layers + self.classification_layer if l != 'classification'])))
        if dims[1] == 6:
            color_feat = self.feature_extractor(im[:, :3, :, :], layers)
            depth_feat = self.feature_extractor_depth(im[:, 3:, :, :], layers)
            all_feat = self.merge(color_feat, depth_feat)
        else:
            all_feat = self.feature_extractor(im, backbone_layers)
        all_feat['classification'] = self.extract_classification_feat(all_feat)
        return OrderedDict({l: all_feat[l] for l in layers})

@model_constructor
def mvt_DeT(filter_size=1, optim_iter=5, optim_init_step=1.0, optim_init_reg=0.01,
              classification_layer='layer4_A', feat_stride=16, backbone_pretrained='', clf_feat_blocks=0,
              clf_feat_norm=True, init_filter_norm=False, final_conv=True,
              out_feature_dim=512, init_gauss_sigma=1.0, num_dist_bins=5, bin_displacement=1.0,
              mask_init_factor=4.0, iou_input_dim=(256, 256), iou_inter_dim=(256, 256),
              score_act='relu', act_param=None, target_mask_act='sigmoid',
              detach_length=float('Inf'), frozen_backbone_layers=(),
              merge_type='max', W_rgb=0.6, W_depth=0.4):

    # Backbone

    if not backbone_pretrained: # lazy fix
        backbone_pretrained = '../outputs/pretrained_networks/mobilevit_s.pt'
        
    backbone_net = backbones.mobilevit(pretrained=backbone_pretrained) # , frozen_layers=frozen_backbone_layers)
    backbone_net_depth = backbones.mobilevit(pretrained=backbone_pretrained) #, frozen_layers=frozen_backbone_layers)
    
    # Feature normalization
    norm_scale = math.sqrt(1.0 / (out_feature_dim * filter_size * filter_size))

    # Classifier features
    if classification_layer == 'layer4_A':
        feature_dim = 32
    else:
        raise Exception


    clf_feature_extractor = clf_features.residual_bottleneck(feature_dim=feature_dim,
                                                             num_blocks=clf_feat_blocks, l2norm=clf_feat_norm,
                                                             final_conv=final_conv, norm_scale=norm_scale,
                                                             out_dim=out_feature_dim)

    # Initializer for the DiMP classifier
    initializer = clf_initializer.FilterInitializerLinear(filter_size=filter_size, filter_norm=init_filter_norm,
                                                          feature_dim=out_feature_dim)

    # Optimizer for the DiMP classifier
    optimizer = clf_optimizer.DiMPSteepestDescentGN(num_iter=optim_iter, feat_stride=feat_stride,
                                                    init_step_length=optim_init_step,
                                                    init_filter_reg=optim_init_reg, init_gauss_sigma=init_gauss_sigma,
                                                    num_dist_bins=num_dist_bins,
                                                    bin_displacement=bin_displacement,
                                                    mask_init_factor=mask_init_factor,
                                                    score_act=score_act, act_param=act_param, mask_act=target_mask_act,
                                                    detach_length=detach_length)

    # The classifier module
    classifier = target_clf.LinearFilter(filter_size=filter_size, filter_initializer=initializer,
                                         filter_optimizer=optimizer, feature_extractor=clf_feature_extractor)

    # Bounding box regressor
    bb_regressor = bbmodels.AtomIoUNet(input_dim=(128, 128), pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim) # layer3, layer4

    # DiMP network
    net = MVT_DeT(feature_extractor=backbone_net, feature_extractor_depth=backbone_net_depth, classifier=classifier, bb_regressor=bb_regressor,
                      classification_layer='layer4_A', bb_regressor_layer=['layer4_A', 'layer4_B'],
                      merge_type=merge_type, W_rgb=W_rgb, W_depth=W_depth)
    return net
