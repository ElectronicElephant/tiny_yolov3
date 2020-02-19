Test implementation of Tiny-YOLO-v3. 

Based on MXNet and Gluon-cv.

This repo is in active development. Issues are welcomed.

---

### Features
- Morden-day tricks, including multi-scale training and mix-up
- Pretrained weights and logs provided
- EXTREMELY FAST (See below)

### Preparation

#### 0) Requirements
- python 3.7
- mxnet 1.5.1
- numpy < 1.18
- matplotlib
- tqdm
- opencv
- pycocotools

Note that `numpy 1.18` will cause problem for `pycocotools`.
[See more](https://github.com/xingyizhou/CenterNet/issues/547).

I'd suggest creating a new conda environment.

```
conda create -n tinyyolo python=3.7 numpy=1.17 matplotlib tqdm opencv Cython
conda activate tinyyolo
pip install mxnet-cu101mkl pycocotools
```

Other versions like `mxnet-cu92` and `mxnet-cu92mkl` are all acceptable.

#### 1) Code
```
git clone git@github.com:EletronicElephant/tiny_yolov3.git
cd tiny_yolov3/gluon-cv
python setup.py develop --user
```

#### 2) Data
Up to now, only MS COCO-formatted dataset is supported.
```
cd ./..  # return to the root
mkdir data
cd data
ln -s /disk1/data/coco
```

#### 3) weights
`weights/best.params`

Evaluation results on COCO `val2017` are listed below.

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.139
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.297
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.114
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.137
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.224
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.159
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.248
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.262
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.270
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.406
```

---

### Training
You can either edit the parameters by changing the default values in `trian.py` or specify it.

Personally, I would recommend creating a new file named `train.sh` and adds

```
python train.py \
--batch-size 64 \
--gpus 4,5  \
--num-workers 16 \
--warmup-epochs 2 \
--lr 0.001 \
--epochs 200 \
--lr-mode step \
--save-prefix ./results/1/ \
--save-interval 1 \
--log-interval 100 \
--start-epoch 0 \
--optimizer sgd \
--label-smooth 
```

Make sure you have 24GB gMemory for training with `batch-size=64` and `random-shape`.

In other words, training with `bs=64` and `data-shape=640` will use 24GB gMemory.

For a more commonly-used shape `416*416`, 12GB gMemory will be used for `bs=64` at `200 Samples/second` on a `Titan Xp`.

---

### Evaluation
I believe online-evaluating is stupid, for it can waste valuable training time.
Instead, I would suggest a `bash` trick.
```                                                                    
for epoch in {0000..0199..1}
do
    while [ ! -f ./results/yolo3_tiny_darknet_coco_${epoch}.params ]
    do
    echo -n "."
    sleep 60
    done

    python eval.py --data-shape 416 \
    --save-prefix ./results/ \
    --gpus 3 --batch-size 4 --num-workers 4 --start-epoch ${epoch} 
done           
```

### Demo
TODO

### Benchmarking the speed of network
`python eval.py --resume weights/best.params --benchmark`

Here is the test result on `Titan Xp`

| data-shape |   fps (bs=1) |fps (bs=8)
|:----------:|:------:|:------:|
| 320        |   218  |633|
|416|204|638|
|608|156|327|

---

### Credits
I got a lot of code from [gluon-cv](https://github.com/dmlc/gluon-cv.git). Thanks.

### Comments
If you encountered with high CPU-usage while training (especially on some machines that have more than 40 cores), you can set these environmental variables
```
export MKL_NUM_THREADS="1"
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
export OMP_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
```
[See more](http://www.diracprogram.org/doc/release-12/installation/mkl.html)

MXNet currently doesn't provide any high-performance image-data-argumentation method. The whole training speed is largely infected by the transformer.

### Known Issues

- [UNTESTED] Mixup will not work.
- [UNTESTED] `Adam` optimizer runs slowly.
