"""Train Tiny-YOLOv3 with random shapes."""
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
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.utils import LRScheduler
from gluoncv.data import batchify


def parse_args():
    parser = argparse.ArgumentParser(description='Train Tiny-YOLOv3 networks with random input shape.')
    parser.add_argument('--data-shape', type=int, default=416,
                        help="Input data shape for evaluation, use 320, 416, 608... ")
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset. Currently support ONLY COCO.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. ')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,180',
                        help='epochs at which learning rate decays. default is 260,280.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='./results/',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                             'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='Control the optimizer (SGD / Adam)')

    args = parser.parse_args()
    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(root='./data/coco', splits='instances_train2017', use_crowd=False)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    if args.mixup:  # is broken now. do not try
        from gluoncv.data import MixupDetection
        train_dataset = MixupDetection(train_dataset)
    return train_dataset


def get_dataloader(net, train_dataset, data_shape, batch_size, num_workers, args):
    """Get dataloader."""
    width, height = data_shape, data_shape
    # stack image, all targets generated
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_dataset, batch_size=batch_size, interval=10, last_batch='rollover',
            shuffle=True, batchify_fn=batchify_fn, num_workers=num_workers)

    return train_loader


def save_params(net, epoch, save_interval, prefix):
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}.params'.format(prefix, epoch))


def train(net, train_data, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    if args.lr_decay_period > 0:
        lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    lr_scheduler = LRScheduler(mode=args.lr_mode,
                               baselr=args.lr,
                               niters=args.num_samples // args.batch_size,
                               nepochs=args.epochs,
                               step=lr_decay_epoch,
                               step_factor=args.lr_decay, power=2,
                               warmup_epochs=args.warmup_epochs)

    if args.optimizer.lower() == 'adam':
        opt_name = 'adam'
        opt_param = {'wd': args.wd, 'lr_scheduler': lr_scheduler}
    elif args.optimizer.lower() == 'sgd':
        opt_name = 'sgd'
        opt_param = {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler}
    else:
        raise NotImplementedError(f'The optimizer {args.optimizer.lower()} is not implemented.')

    trainer = gluon.Trainer(net.collect_params(), opt_name, opt_param, kvstore='local')

    # metrics
    obj_metrics = mx.metric.Loss('O')
    center_metrics = mx.metric.Loss('BC')
    scale_metrics = mx.metric.Loss('BS')
    cls_metrics = mx.metric.Loss('C')
    coef_metrics = mx.metric.Loss('Cf')
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    for epoch in range(args.start_epoch, args.epochs):
        if args.mixup:
            # TODO(threshold): more elegant way to control mixup during runtime
            try:
                train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
            except AttributeError:
                train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
            if epoch >= args.epochs - args.no_mixup_epochs:
                try:
                    train_data._dataset.set_mixup(None)
                except AttributeError:
                    train_data._dataset._data.set_mixup(None)

        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                      *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                logger.info('[E {}][B {}], LR: {:.2E}, {:.1f} S/s, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, batch_size / (time.time() - btic), name1, loss1, name2, loss2,
                    name3, loss3, name4, loss4))
            btic = time.time()

        name1, loss1 = obj_metrics.get()
        name2, loss2 = center_metrics.get()
        name3, loss3 = scale_metrics.get()
        name4, loss4 = cls_metrics.get()
        logger.info('[E {}] {:.1f} sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
        save_params(net, epoch, args.save_interval, args.save_prefix)


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
    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                        norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
    else:
        net = get_model(net_name, pretrained_base=True)
        async_net = net
    if args.resume.strip():
        print(f"Loading weights from {args.resume.strip()}")
        net.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            async_net.initialize()
    print("model loaded")
    # training data
    train_dataset = get_dataset(args.dataset, args)
    print("dataset loaded")
    train_data = get_dataloader(
        async_net, train_dataset, args.data_shape, args.batch_size, args.num_workers, args)
    print("dataloader done")
    # training
    train(net, train_data, ctx, args)
