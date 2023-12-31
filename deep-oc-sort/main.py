import os
import shutil
import time

import torch
import cv2
import numpy as np

from loguru import logger
import dataset
import utils
from external.adaptors import detector
from trackers import integrated_ocsort_embedding as tracker_module

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_main_args():
    parser = tracker_module.args.make_parser()
    parser.add_argument("--dataset", type=str, default="soccernet")
    parser.add_argument("--ann_file_name", type=str, default=None)
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="exp1")
    parser.add_argument("--min_box_area", type=float, default=10, help="filter out tiny boxes")
    parser.add_argument(
        "--aspect_ratio_thresh",
        type=float,
        default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="run post-processing linear interpolation.",
    )
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    parser.add_argument(
        "--alpha_fixed_emb",
        type=float,
        default=0.95,
        help="Alpha fixed for EMA embedding",
    )
    parser.add_argument("--emb_off", action="store_true")
    parser.add_argument("--cmc_off", action="store_true")
    parser.add_argument("--aw_off", action="store_true")
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--new_kf_off", action="store_true")
    parser.add_argument("--grid_off", action="store_true")
    args = parser.parse_args()

    if args.dataset == "soccernet":
        args.result_folder = os.path.join(args.result_folder, "SoccerNet-val")
    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
        
    return args


def main():
    np.set_printoptions(suppress=True, precision=5)
    # Set dataset and detector
    args = get_main_args()
    if args.dataset == "soccernet":
        detector_path = "/scratch/hk3820/csgi2271_finalproject/YOLOX_outputs/yolox_x_10Ep_bytetrack_mot_17_run2/best_ckpt.pth.tar"
        size = (896, 1600)
    else:
        raise RuntimeError("Need to update paths for detector for extra datasets.")
    det = detector.Detector("yolox", detector_path, args.dataset)
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size, ann_file_name = args.ann_file_name)

    # Set up tracker
    oc_sort_args = dict(
        args=args,
        det_thresh=args.track_thresh,
        iou_threshold=args.iou_thresh,
        asso_func=args.asso,
        delta_t=args.deltat,
        inertia=args.inertia,
        w_association_emb=args.w_assoc_emb,
        alpha_fixed_emb=args.alpha_fixed_emb,
        embedding_off=args.emb_off,
        cmc_off=args.cmc_off,
        aw_off=args.aw_off,
        aw_param=args.aw_param,
        new_kf_off=args.new_kf_off,
        grid_off=args.grid_off,
    )
    tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
    results = {}
    frame_count = 0
    total_time = 0
    
    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)

    # See __getitem__ of dataset.MOTDataset
    prev_video_name = None
    for (img, np_img), label, info, idx in loader:
        # Frame info
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]
        if frame_id == 750:
            prev_video_name = video_name

        # Hacky way to skip SDP and DPM when testing
        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue
        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []
        img = img.cuda()

        # Initialize tracker on first frame of a new video
        if frame_id == 1: 
            logger.info(f"Initializing tracker for {video_name}")
            tracker = tracker_module.ocsort.OCSort(**oc_sort_args)
            video_start_time = time.time()
        
        start_time = time.time()
        # Nx5 of (x1, y1, x2, y2, conf), pass in tag for caching
        pred = det(img, tag)
        if pred is None:
            continue
        # Nx5 of (x1, y1, x2, y2, ID)
        targets = tracker.update(pred, img, np_img[0].numpy(), tag)
        tlwhs, ids = utils.filter_targets(targets, args.aspect_ratio_thresh, args.min_box_area)
        
        if frame_id % 50 == 0:
            logger.info(f"Processed {frame_id} frames. Time taken: {(time.time() - video_start_time):.2f}. fps: {(frame_id/(time.time() - video_start_time)):.2f}")
        total_time += time.time() - start_time
        frame_count += 1
        results[video_name].append((frame_id, tlwhs, ids))
        
        if frame_id == 750:
            if prev_video_name:
                logger.info(f"Finished, results saved to {folder}")
                result_filename = os.path.join(folder, f"{prev_video_name}.txt")
                utils.write_results_no_score(result_filename, results[prev_video_name])
                logger.info(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            tracker.dump_cache()

    logger.info(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    # Save detector results
    det.dump_cache()
    tracker.dump_cache()

    # Save post for all sequences
    if args.post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            logger.info(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        utils.dti(post_folder_data, post_folder_data)
        logger.info(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

if __name__ == "__main__":
    main()
