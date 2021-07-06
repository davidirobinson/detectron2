# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
from PIL import Image
import tqdm

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# From Vitis AI
import torch
import numpy as np
from pytorch_nndct.apis import torch_quantizer


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 10, kernel_size=3, stride=3),
            nn.BatchNorm2d(10),
            nn.Flatten()
            )
    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Setup model
    predictor = DefaultPredictor(cfg)

    # # Do inference
    # if args.input:
    #     if len(args.input) == 1:
    #         args.input = glob.glob(os.path.expanduser(args.input[0]))
    #         assert args.input, "The input path(s) was not found"
    #     for path in tqdm.tqdm(args.input, disable=not args.output):
    #         # use PIL, to be consistent with evaluation
    #         img = read_image(path, format="BGR")
    #         start_time = time.time()

    #         predictions = predictor(img)
    #         logger.info(
    #             "{}: {} in {:.2f}s".format(
    #                 path,
    #                 "detected {} instances".format(len(predictions["instances"]))
    #                 if "instances" in predictions
    #                 else "finished",
    #                 time.time() - start_time,
    #             )
    #         )

    # Quantize
    quant_mode = 'calib'
    quant_mode = 'test'
    channels = 3
    height = 320
    width = 480

    batched_inputs = torch.randn([channels, height, width])

    quantizer = torch_quantizer(
        quant_mode, predictor.model, batched_inputs, output_dir="quant_model/")
    quantized_model = quantizer.quant_model

    # Evaluate / Test
    def test(model, device, width, height):
        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"

            model.eval()
            with torch.no_grad():
                for path in tqdm.tqdm(args.input, disable=not args.output):
                    # use PIL, to be consistent with evaluation
                    image = read_image(path, width, height, format="BGR")

                    # Apply pre-processing to image.
                    if cfg.INPUT.FORMAT == "RGB":
                        # whether the model expects BGR inputs or RGB
                        image = image[:, :, ::-1]
                    height, width = image.shape[:2]

                    image = image.astype("float32").transpose(2, 0, 1)

                    # Normalize - Vitis AI Quantizer doesn't like doing this
                    image = image.reshape(3, height * width)
                    image -= np.array(cfg.MODEL.PIXEL_MEAN).reshape(3, 1)
                    image /= np.array(cfg.MODEL.PIXEL_STD).reshape(3, 1)
                    image = image.reshape(3, height, width)

                    # Convert to device
                    image = torch.as_tensor(image).to(device)

                    start_time = time.time()

                    og_result = predictor.model(image)
                    result = model(image)

                    print("in ", time.time() - start_time, "s")


    test(quantized_model, torch.device('cuda:0'), width, height)

    # export config
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if quant_mode == 'test':
        quantizer.export_xmodel(deploy_check=False, output_dir="quant_model/")
