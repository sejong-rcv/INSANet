import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

def find_tp(true_bndboxes, det_bndboxes):
    def compute_iou(outputs: np.array, labels: np.array):
        outputs = outputs.squeeze(1)
        
        intersection = (outputs & labels).sum((1, 2))
        union = (outputs | labels).sum((1, 2))
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
        
        return thresholded
        
    



# Load images
db_root = '../../data/kaist-rgbt/'
img_file = os.path.join(db_root, 'test-all-20.txt')
imgpath = os.path.join('%s', 'images', '%s', '%s', '%s', '%s.jpg')
annopath = os.path.join('%s', 'annotations_paired', '%s', '%s', '%s', '%s.txt')

width, height = 640, 512
save_path = os.path.join('./visualize')
os.makedirs(save_path, exist_ok=True)

ids = list()
for line in open(img_file):
    ids.append((db_root, line.strip().split('/')))

for idx in tqdm(range(len(ids))):
    frame_id = ids[idx]
    set_id, vid_id, img_id = frame_id[-1]
    vis_imgpath = imgpath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )
    lwir_imgpath = imgpath % ( *frame_id[:-1], set_id, vid_id, 'lwir', img_id )
    
    bndboxes_ = list()
    for line in open(annopath % ( *frame_id[:-1], set_id, vid_id, 'visible', img_id )) :
        bndboxes.append(line.strip().split(' '))
    bndboxes_ = bndboxes_[1:]
    
    bndboxes = list()
    if len(bndboxes_) >= 1:
        for i in range(len(bndboxes_)):
            bndbox = [int(i) for i in bndboxes_[i][1:5]]
            bndbox[2] = min( bndbox[2] + bndbox[0], width )
            bndbox[3] = min( bndbox[3] + bndbox[1], height )
            bndboxes.append(bndbox)
    
    