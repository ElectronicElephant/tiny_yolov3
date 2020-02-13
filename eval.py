"""Test Tiny-YOLOv3 with random shapes."""
import argparse
import os
import logging
import mxnet as mx
from mxnet import gluon
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.mscoco.detection import COCODetection
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Tiny-YOLOv3 Evaluation')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... ")
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Training mini-batch size.')
    parser.add_argument('--save-prefix', type=str, default='./results/',
                        help='Saving parameter prefix')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Only COCO is supported.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int, default=2,
                        help='Number of data workers.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Eval with GPUs. We recommend using only 1 GPU to eval')
    parser.add_argument('--resume', type=str, default='',
                        help='The path of the saved params')
    parser.add_argument('--start-epoch', type=int, default=-1,
                        help='The epoch of saved parameters')
    parser.add_argument('--save-json', action='store_true',
                        help='To save the detection result to json files')
    parser.add_argument('--score-thresh', type=float, default=0.001,
                        help='Detections will be ignored if confidence scores < threshold.')
    return parser.parse_args()


def get_dataset(dataset, args):
    width, height = args.data_shape, args.data_shape
    if dataset.lower() == 'coco':
        val_dataset = COCODetection(root='./data/coco', splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=not args.save_json,
                                         data_shape=(height, width), score_thresh=args.score_thresh)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric


def get_dataloader(val_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers, batchify_fn=batchify_fn)
    return val_loader


def validate(net, val_data, ctx, eval_metric, size, args):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    eval_metric.reset()
    net.hybridize()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)

    mx.nd.waitall()
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

    map_bbox = validate(net, val_data, ctx, eval_metric, len(val_dataset), args)
    map_name, mean_ap = map_bbox
    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    logger.info('[Epoch {}] Validation: \n{}'.format(args.start_epoch, val_msg))


if __name__ == '__main__':
    args = parse_args()

    # evaluating contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', 'tiny_darknet', args.dataset))
    args.save_prefix += net_name

    net = get_model(net_name)
    if not args.resume.strip():
        if args.start_epoch == -1:
            raise ValueError("You have to either give the path of the saved model or specify the start epoch!")
        # Predict the path of the saved weights from the `start_epoch` parameter
        args.resume = '{:s}_{:04d}.params'.format(args.save_prefix, args.start_epoch)
    print(f'Loading weights from {args.resume}')
    net.load_parameters(args.resume.strip())

    # val data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(val_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    # Valing
    demo_val(net, val_data, eval_metric, ctx, args)
