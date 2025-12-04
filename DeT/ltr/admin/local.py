import os

class EnvironmentSettings:
    def __init__(self):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.workspace_dir = os.path.join(file_dir, '../../../outputs/')    # Base directory for saving network checkpoints.
        self.tensorboard_dir = os.path.join(self.workspace_dir, 'tensorboard/')    # Directory for tensorboard files.
        self.pretrained_networks = os.path.join(self.workspace_dir, 'pretrained_networks/')
        self.lasot_dir = ''
        self.got10k_dir = ''
        self.trackingnet_dir = ''
        self.coco_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.cdtb_dir = ''
        self.depthtrack_dir = os.path.join(self.workspace_dir, '../../data/depthtrack/train')
        self.lasotdepth_dir = ''
        self.cocodepth_dir = ''
