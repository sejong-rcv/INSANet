import os
from os.path import join as opj
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import argparse
import config
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import KAISTPed
from utils.evaluation_script import evaluate
from vis import visualize

from model import INSANet


def val_epoch(model: INSANet, dataloader: DataLoader, input_size: Tuple, min_score: float = 0.1, epoch: int = 0) -> Dict:
    """
    Validate the model during an epoch
    
    :param model: INSA network model for multispectral pedestrian detection defined by src/model.py
    :param dataloader: Dataloader instance to feed training data(images, labels, etc) for KAISTPed dataset
    :param input_size: A tuple of (height, width) for input image to restore bounding box from the raw prediction
    :param min_score: Detection score threshold, i.e. low-confidence detections(< min_score) will be discarded
    :return: A dict of numpy arrays (K x 5: xywh + score) for given image
    """

    model.eval()

    height, width = input_size
    xyxy_scaler_np = np.array([[width, height, width, height]], dtype=np.float32)

    device = next(model.parameters()).device
    results = dict()

    with torch.no_grad():
        for i, blob in enumerate(tqdm(dataloader, desc='Evaluating')):
            image_vis, image_lwir, boxes, labels, indices = blob

            image_vis = image_vis.to(device)
            image_lwir = image_lwir.to(device)

            # Forward prop.
            predicted_locs, predicted_scores = model(image_vis, image_lwir)
            
            # Detect objects in model output
            detections = model.module.detect_objects(predicted_locs, predicted_scores,
                                                     min_score=min_score, max_overlap=0.425, top_k=200)
            
            det_boxes_batch, det_labels_batch, det_scores_batch = detections[:3]
            
            for boxes_t, labels_t, scores_t, image_id in zip(det_boxes_batch, det_labels_batch, det_scores_batch, indices):
                boxes_np = boxes_t.cpu().numpy().reshape(-1, 4)
                scores_np = scores_t.cpu().numpy().mean(axis=1).reshape(-1, 1)

                xyxy_np = boxes_np * xyxy_scaler_np
                xywh_np = xyxy_np
                xywh_np[:, 2] -= xywh_np[:, 0]
                xywh_np[:, 3] -= xywh_np[:, 1]

                results[image_id.item() + 1] = np.hstack([xywh_np, scores_np])
           
    return results


def run_inference(model_path: str) -> Dict:
    """Load model and run inference

    Load pretrained model and run inference on KAIST dataset.

    :param model_path: Full path of pytorch model
    :return: A dict of numpy arrays (K x 5: xywh + score) for given image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path)['model']
    model = model.to(device)

    model = nn.DataParallel(model)

    input_size = config.test.input_size

    args = config.args
    
    batch_size = config.test.batch_size * torch.cuda.device_count()
    test_dataset = KAISTPed(args, condition='test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.dataset.workers,
                                              collate_fn=test_dataset.collate_fn,
                                              pin_memory=True)

    results = val_epoch(model, test_loader, input_size)

    return results


def save_results(results: Dict, result_filename: str):
    """Save detections

    Write a result file (.txt) for detection results.
    The results are saved in the order of image index.

    :param results: Detection results for each image_id: {image_id: bbox_xywh + score}
    :param file_name: Full path of result file name
    """

    if not result_filename.endswith('.txt'):
        result_filename += '.txt'

    with open(result_filename, 'w') as f:
        for image_id, detections in sorted(results.items(), key=lambda x: x[0]):
            for x, y, w, h, score in detections:
                f.write(f'{image_id},{x:.4f},{y:.4f},{w:.4f},{h:.4f},{score:.8f}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process with checkpoint')

    parser.add_argument('--model-path', required=True, type=str,
                        help='Pretrained model for evaluation.')
    parser.add_argument('--result-dir', type=str, default='../result',
                        help='Save result directory')
    parser.add_argument('--vis', default=False, action='store_true', 
                        help='Visualizing the results')
    arguments = parser.parse_args()

    print(arguments)

    model_path = Path(arguments.model_path).stem.replace('.', '_')

    # Run inference to get detection results
    os.makedirs(arguments.result_dir, exist_ok=True)
    result_filename = opj(arguments.result_dir,  f'{model_path}_TEST_det')

    # Run inference
    results = run_inference(arguments.model_path)

    # Save results
    save_results(results, result_filename)

    # Eval results
    phase = "Multispectral"
    evaluate(config.PATH.JSON_GT_FILE, result_filename + '.txt',  phase) 
    
    # Visualizing
    if arguments.vis:
        vis_dir = opj(arguments.result_dir, 'vis', model_path)
        os.makedirs(vis_dir, exist_ok=True)
        visualize(result_filename + '.txt', vis_dir)
