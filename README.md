Test implementation of Tiny-YOLOv3. 

Based on MXNet and Gluon-cv.

This repo is still in active development.

DO NOT RUN THE CODE IN THIS STAGE!

# TODO
- [ ] Wash all the info from previous project

- [ ] Delete redundant files

- [ ] Test Everything

- [ ] FINISH IN ONE WEEK

---

### Requirements
- python 3.7
- mxnet 1.5.1
- numpy < 1.18
- matplotlib
- tqdm
- opencv
- pycocotools

Note that `numpy 1.18` will cause problem for `pycocotools`.
[See more](https://github.com/xingyizhou/CenterNet/issues/547).

### Features
TODO

### Preparation
##### 1) Code
```
git clone git@github.com:EletronicElephant/tiny_yolov3.git
cd tiny_yolov3/gluon-cv
python setup.py develop --user
```

##### 2) Data
Up to now, only MS COCO-formatted dataset is supported.
TODO

##### 3) weights
TODO

### Training
TODO

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
If you encountered with high CPU-usage while training (especially on some machines that have more than 40 cores), you can set these environmental variavles
```
export MKL_NUM_THREADS="1"
export MKL_DOMAIN_NUM_THREADS="MKL_BLAS=1"
export OMP_NUM_THREADS="1"
export MKL_DYNAMIC="FALSE"
export OMP_DYNAMIC="FALSE"
```
[See more](http://www.diracprogram.org/doc/release-12/installation/mkl.html)

### Known Issues

- Mixup will not work
