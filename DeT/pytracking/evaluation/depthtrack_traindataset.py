import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList


class DepthTrack_Train_Dataset(BaseDataset):
    """
    CDTB, RGB dataset, Depth dataset, Colormap dataset, RGB+depth
    """
    def __init__(self, dtype='colormap'):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_train_path
        self.sequence_list = self._get_sequence_list()
        self.dtype = dtype

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        sequence_path = sequence_name
        nz = 8
        start_frame = 1

        if self.dtype == 'color':
            ext = 'jpg'
        elif self.dtype == 'rgbd':
            ext = ['jpg', 'png'] # Song not implemented yet
        else:
            ext = 'png'

        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)
        try:
            ground_truth_rect = np.loadtxt(str(anno_path), dtype=np.float64)
        except:
            ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64)

        end_frame = ground_truth_rect.shape[0]

        if self.dtype in ['colormap', 'normalized_depth', 'raw_depth', 'centered_colormap', 'centered_normalized_depth', 'centered_raw_depth']:
            group = 'depth'
        elif self.dtype == 'color':
            group = self.dtype
        else:
            group = self.dtype

        if self.dtype in ['rgbd', 'rgbcolormap']:
            depth_frames = ['{base_path}/{sequence_path}/depth/{frame:0{nz}}.png'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            color_frames = ['{base_path}/{sequence_path}/color/{frame:0{nz}}.jpg'.format(base_path=self.base_path,
                            sequence_path=sequence_path, frame=frame_num, nz=nz)
                            for frame_num in range(start_frame, end_frame+1)]
            # frames = {'color': color_frames, 'depth': depth_frames}
            frames = []
            for c_path, d_path in zip(color_frames, depth_frames):
                frames.append({'color': c_path, 'depth': d_path})

        else:
            frames = ['{base_path}/{sequence_path}/{group}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                      sequence_path=sequence_path, group=group, frame=frame_num, nz=nz, ext=ext)
                      for frame_num in range(start_frame, end_frame+1)]

        # Convert gt
        if ground_truth_rect.shape[1] > 4:
            gt_x_all = ground_truth_rect[:, [0, 2, 4, 6]]
            gt_y_all = ground_truth_rect[:, [1, 3, 5, 7]]

            x1 = np.amin(gt_x_all, 1).reshape(-1,1)
            y1 = np.amin(gt_y_all, 1).reshape(-1,1)
            x2 = np.amax(gt_x_all, 1).reshape(-1,1)
            y2 = np.amax(gt_y_all, 1).reshape(-1,1)

            ground_truth_rect = np.concatenate((x1, y1, x2-x1, y2-y1), 1)

        return Sequence(sequence_name, frames, 'depthtrack', ground_truth_rect, dtype=self.dtype)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        sequence_list= ['adapter02_indoor',       
                        'bag03_indoor',             
                        'bag04_indoor',             
                        'bag05_indoor',            
                        'ball02_indoor',           
                        'ball03_indoor',           
                        'ball04_indoor',           
                        'ball05_indoor',           
                        'ball07_indoor',           
                        'ball08_wild',             
                        'ball09_wild',             
                        'ball12_wild',              
                        'ball13_indoor',            
                        'ball14_wild',              
                        'ball16_indoor',            
                        'ball17_wild',              
                        'ball19_indoor',            
                        'ball21_indoor',            
                        'basket_indoor',            
                        'beautifullight01_indoor',  
                        'bike01_wild',              
                        'bike02_wild',              
                        'bike03_wild',              
                        'book01_indoor',            
                        'book02_indoor',            
                        'book04_indoor',            
                        'book05_indoor',            
                        'book06_indoor',            
                        'bottle01_indoor',          
                        'bottle02_indoor',          
                        'bottle03_indoor',          
                        'bottle05_indoor',     
                        'bottle06_indoor',     
                        'box_indoor',          
                        'candlecup_indoor',    
                        'car01_indoor',        
                        'car02_indoor',        
                        'cart_indoor',         
                        'cat02_indoor',        
                        'cat03_indoor',        
                        'cat04_indoor',        
                        'cat05_indoor',        
                        'chair01_indoor',      
                        'chair02_indoor',      
                        'clothes_indoor',      
                        'colacan01_indoor',    
                        'colacan02_indoor',    
                        'colacan04_indoor',    
                        'container01_indoor',  
                        'container02_indoor',  
                        'cube01_indoor',       
                        'cube04_indoor',       
                        'cube06_indoor',       
                        'cup03_indoor',        
                        'cup05_indoor',        
                        'cup06_indoor',        
                        'cup07_indoor',        
                        'cup08_indoor',        
                        'cup09_indoor',        
                        'cup10_indoor',        
                        'cup11_indoor',        
                        'cup13_indoor',        
                        'cup14_indoor',         
                        'duck01_wild',          
                        'duck02_wild',          
                        'duck04_wild',          
                        'duck05_wild',          
                        'duck06_wild',          
                        'dumbbells02_indoor',   
                        'earphone02_indoor',    
                        'egg_indoor',           
                        'file02_indoor',        
                        'flower01_indoor',      
                        'flower02_wild',        
                        'flowerbasket_indoor',  
                        'ghostmask_indoor',     
                        'glass02_indoor',       
                        'glass03_indoor',       
                        'glass04_indoor',       
                        'glass05_indoor',       
                        'guitarbag_indoor',     
                        'gymring_wild',         
                        'hand02_indoor',        
                        'hat01_indoor',         
                        'hat02_indoor_320',     
                        'hat03_indoor',         
                        'hat04_indoor',         
                        'human01_indoor',       
                        'human03_wild',         
                        'human04_wild',         
                        'human05_wild',         
                        'human06_indoor',       
                        'leaves01_wild',        
                        'leaves02_indoor',       
                        'leaves03_wild',         
                        'leaves04_indoor',       
                        'leaves05_indoor',       
                        'leaves06_wild',         
                        'lock01_wild',           
                        'mac_indoor',            
                        'milkbottle_indoor',     
                        'mirror_indoor',         
                        'mobilephone01_indoor',  
                        'mobilephone02_indoor',  
                        'mobilephone04_indoor',  
                        'mobilephone05_indoor',  
                        'mobilephone06_indoor',  
                        'mushroom01_indoor',     
                        'mushroom02_wild',       
                        'mushroom03_wild',       
                        'mushroom04_indoor',     
                        'mushroom05_indoor',     
                        'notebook02_indoor',     
                        'notebook03_indoor',     
                        'paintbottle_indoor',    
                        'painting_indoor_320',   
                        'parkingsign_wild',      
                        'pigeon03_wild',         
                        'pigeon05_wild',         
                        'pigeon06_wild',         
                        'pigeon07_wild',
                        'pine01_indoor',
                        'pine02_wild_320',
                        'shoes01_indoor',
                        'shoes03_indoor',
                        'skateboard01_indoor',
                        'skateboard02_indoor',
                        'speaker_indoor',
                        'stand_indoor',
                        'suitcase_indoor',
                        'swing01_wild',
                        'swing02_wild',
                        'teacup_indoor',
                        'thermos01_indoor',
                        'thermos02_indoor',
                        'toiletpaper02_indoor',
                        'toiletpaper03_indoor',
                        'toiletpaper04_indoor',
                        'toy01_indoor',
                        'toy03_indoor',
                        'toy04_indoor',
                        'toy05_indoor',
                        'toy06_indoor',
                        # 'toy07_indoor_320',
                        'toy08_indoor',
                        'toy10_indoor',
                        'toydog_indoor',
                        'trashbin_indoor',
                        'tree_wild',
                        'trophy_indoor',
                        'ukulele02_indoor',]

        return sequence_list
