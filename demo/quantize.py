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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    # Do inference
    if False:
        if args.input:
            if len(args.input) == 1:
                args.input = glob.glob(os.path.expanduser(args.input[0]))
                assert args.input, "The input path(s) was not found"
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()

                predictions = predictor(img)
                logger.info(
                    "{}: {} in {:.2f}s".format(
                        path,
                        "detected {} instances".format(len(predictions["instances"]))
                        if "instances" in predictions
                        else "finished",
                        time.time() - start_time,
                    )
                )

    def run_quantizer(cfg, quant_mode, device=torch.device('cuda:0')):
        # Setup model
        predictor = DefaultPredictor(cfg)

        # TODO(drobinson): Support larger image sizes. Note that currently the networks
        # must be divisible by the network's self.backbone.size_divisibility parameter
        channels = 3
        height = 320
        width = 480

        batched_inputs = torch.randn([channels, height, width])

        quantizer = torch_quantizer(
            quant_mode=quant_mode,
            module=predictor.model,
            input_args=batched_inputs,
            output_dir="quant_model/",
            bitwidth=8)
        quantized_model = quantizer.quant_model

        # Evaluate / Test
        def test(model, device, width, height):
            if args.input:
                if len(args.input) == 1:
                    args.input = glob.glob(os.path.expanduser(args.input[0]))
                    assert args.input, "The input path(s) was not found"

                model.eval()
                with torch.no_grad():
                    for path in args.input:
                        # Load image for inference
                        image = cv2.imread(path) # Load in BGR format
                        assert image.shape[-1] == 3
                        if cfg.INPUT.FORMAT == "RGB":
                            # whether the model expects BGR inputs or RGB
                            image = image[:, :, ::-1]

                        # Apply pre-processing to image.
                        image = cv2.resize(image, (width, height))
                        image = image.astype("float32").transpose(2, 0, 1)

                        # TODO(drobinson): Handle this in the model
                        image = image.reshape(3, height * width)
                        image -= np.array(cfg.MODEL.PIXEL_MEAN).reshape(3, 1)
                        image /= np.array(cfg.MODEL.PIXEL_STD).reshape(3, 1)
                        image = image.reshape(3, height, width)

                        # Convert to device
                        image = torch.as_tensor(image).to(device)

                        if False:
                            start_time = time.time()
                            og_result = predictor.model(image)
                            print("gpu elapsed:      ", time.time() - start_time, "s")

                        start_time = time.time()
                        result = model(image)
                        print("quantizer elapsed:", time.time() - start_time, "s")

        test(quantized_model, device, width, height)

        # export config
        if quant_mode == 'calib':
            quantizer.export_quant_config()
        if quant_mode == 'test':
            quantizer.export_xmodel(deploy_check=False, output_dir="quant_model/")

    run_quantizer(cfg, 'calib')
    run_quantizer(cfg, 'test')
