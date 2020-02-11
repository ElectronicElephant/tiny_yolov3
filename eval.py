"""Test Tiny-YOLOv3 with random shapes."""
import argparse
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.mscoco.detection import COCODetection
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from tqdm import tqdm
from gluoncv.data import batchify


def parse_args():
    parser = argparse.ArgumentParser(description='Tiny-YOLOv3 Evaluation')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... ")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size. Values higher than 1 may cause problems.')
    parser.add_argument('--save-prefix', type=str, default='./tiny_result/',
                        help='Saving parameter prefix')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Only COCO is supported.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int, default=1,
                        help='Number of data workers. Values higher than 1 may cause problems.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Eval with GPUs. We recommend using only 1 GPU to eval')
    parser.add_argument('--resume', type=str, default='',
                        help='The path of the saved params')
    parser.add_argument('--start-epoch', type=int, default=240,
                        help='The epoch of saved parameters')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    return parser.parse_args()


def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        val_dataset = COCODetection(root='./data/coco', splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(net, val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn, )
    return val_loader


def validate(net, val_data, ctx, eval_metric, size, args):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)

    with tqdm(total=size, ncols=0) as pbar:
        for ib, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []

            for x, y in zip(data, label):  # y stands for img_info
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])

    return eval_metric.get()


def demo_val(net, val_data, eval_metric, ctx, args):
    """Eval pipeline"""
    net.collect_params().reset_ctx(ctx)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_val.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    mx.nd.waitall()
    net.hybridize()

    map_bbox = validate(net, val_data, ctx, eval_metric, len(val_dataset), args)
    map_name, mean_ap = map_bbox
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Epoch {}] Validation: \n{}'.format(args.start_epoch, val_msg))


if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', 'tiny_darknet', args.dataset))
    args.save_prefix += net_name

    net = get_model(net_name, pretrained_base=True)
    async_net = net
    if args.resume.strip():
        print(f'Loading {args.resume}')
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        raise NotImplementedError("You haven't specify the prarms yet!")

    # val data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        async_net, val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    # Valing
    demo_val(net, val_data, eval_metric, ctx, args)
