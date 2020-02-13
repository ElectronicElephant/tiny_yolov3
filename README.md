Test implementation of Tiny-YOLO-v3. 

Based on MXNet and Gluon-cv.

This repo is still in active development.

I'm currently training the network, which may cost about half a week.

# TODO
- [x] Delete redundant files
- [x] `train.py`
- [x] `eval.py`
- [ ] Train the network
- [ ] Test performance
- [ ] FINISH IN ONE WEEK
- [ ] Rewrite the transform part to save CPU load

---

### Features
TODO

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

I'd suggest create a new conda environment.

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
TODO

---

### Training
You can either edit the parameters by changing the default values in `trian.py` or specify it.

Personally, I would recommend create a new file named `train.sh` and adds

```
python train.py --syncbn \
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
TODO

### Demo
TODO

### Benchmarking the speed of network
TODO

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
