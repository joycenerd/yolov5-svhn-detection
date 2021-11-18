# yolov5-svhn-detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### [Report](./REPORT.pdf)

by [Zhi-Yi Chin](https://joycenerd.github.io/)

This repository is implementation of homework2 for IOC5008 Selected Topics in Visual Recognition using Deep Learning course in 2021 fall semester at National Yang Ming Chiao Tung University.

In this homework, we participate in the SVHN detection competition hosted on [CodaLab](https://competitions.codalab.org/competitions/35888?secret_key=7e3231e6-358b-4f06-a528-0e3c8f9e328e). The [Street View House Numbers (SVHN) dataset](http://ufldl.stanford.edu/housenumbers/) contains 33,402 training images and 13,068 testing images. We are required to train not only an accurate but fast digit detector. The submission format should follow COCO results. To test the detection model's speed, we must benchmark the detection model in the Google Colab environment and screenshot the results.

## Getting the code

You can download a copy of all the files in this repository by cloning this repository:

```
git clone https://github.com/joycenerd/yolov5-svhn-detection.git
```

## Requirements

You need to have [Anaconda](https://www.anaconda.com/) or Miniconda already installed in your environment. To install requirements:
```
conda env create --name detect python=3
conda activate detect
cd yolov5
pip install -r requirements.txt
```

## Dataset

You can download the raw data after you have registered the challenge mention above. 

### Data pre-processing

#### 1. Turning the `.mat` label file into YOLO format annotations.

```
python mat2yolo.py --data-root <path_to_data_root_dir>
```
* input: your data root directory, inside this directory you should have `train/` which saves all the training images and `digitStruct.mat` which is the original label file. 
* output: `<path_to_data_root_dir>/labels/all_train/` -> inside this folder there will have text files with the name same as the training image name, they are YOLO format annotations.

#### 2. Train validation split -> Split the original training data into 80% training and 20% validation.
```
python train_valid_split.py --data-root <path_to_data_root_dir> --ratio 0.2
```
* input: same as last step plus the output of last step
* output: 
    * `<path_to_data_root_dir>/images/`: inside this folder will have two subfolder `train/` (training images) and `valid/` (validation images).
    * `<path_to_data_root_dir>/labels/train/`: text files that contain training labels 
    * `<path_to_data_root_dir>/labels/valid/`: text files that contain validation labels

#### 3. Data configuration

Got to `yolov5/data/custom-data.yml` and modified `path`, `train`, `val` and `test` path


## Training

You should have Graphics card to train the model. For your reference, we trained on 2 NVIDIA RTX 1080Ti for 14 hours. Before training, you should download `yolov5s.pt` from `https://github.com/ultralytics/yolov5/releases/tag/v6.0`.

Recommended training command:
```
cd yolov5
python train.py --weights <yolo5s.pt_file> --cfg models/yolov5s.yaml --data data/custom-data.yaml --epochs 150 --cache --device 0,1 --workers 4 --project <train_log_dir> --save-period 5
```
There are more setting arguments you can tune in `yolov5/train.py`, our recommendation is first stick with default setting.

The logging directory will be generated in the path you specified for `--project` and if this is your first experiment there will be a subdirectory name `exp/`, if second `exp2` and so on. Inside this logging directory you can find:
* `weights/`: All the training checkpoints will be saved inside here. Checkpoints is saved every 5 epochs and `best.pth` save the current best model and `last.pt` save the latest model.
* tensorboard
* Some miscellaneous information about the data and current hyperparameter

## Testing
You can test your training results by running this command:
```
python test.py [-h] [--data-root DATA_ROOT] [--ckpt CKPT] [--img-size IMG_SIZE]
               [--num-classes NUM_CLASSES] [--net NET] [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --data-root DATA_ROOT
                        data root dir
  --ckpt CKPT           checkpoint path
  --img-size IMG_SIZE   image size in model
  --num-classes NUM_CLASSES
                        number of classes
  --net NET             which model
  --gpu GPU             gpu id
```

## Submit the results
Run this command to `zip` your submission file:
```
zip answer.zip answer.txt
```
You can upload `answer.zip` to the challenge. Then you can get your testing score.

## Pre-trained models

Click into [Releases](https://github.com/joycenerd/bird-images-classification/releases). Under **EfficientNet-b4 model** download `efficientnet-b4_best_model.pth`. This pre-trained model get accuracy 72.53% on the test set.

Recommended testing command:
```
python test.py --data-root <path_to_data> --ckpt <path_to_checkpoint> --img-size 380 --net efficientnet-b4 --gpu 0
```

`answer.txt` will be generated in this directory. This file is the submission file.

## Inference
To reproduce our results, run this command:
```
python inference.py --data-root <path_to_data> --ckpt <pre-trained_model_path> --img-size 380 --net efficientnet-b4 --gpu 0
```

## Reproducing Submission

To reproduce our submission without retraining, do the following steps

1. [Getting the code](#getting-the-code)
2. [Install the dependencies](#requirements)
2. [Download the data](#dataset)
4. [Download pre-trained models](#pre-trained-models)
3. [Inference](#inference)
4. [Submit the results](#submit-the-results)

## Results

Our model achieves the following performance:

|     | EfficientNet-b4 w/o sched | EfficientNet-b4 with sched |
|-----|---------------------------|----------------------------|
| acc | 55.29%                    | 72.53%                     |

## Citation
If you find our work useful in your project, please cite:

```bibtex
@misc{
    title = {bird_image_classification},
    author = {Zhi-Yi Chin},
    url = {https://github.com/joycenerd/bird-images-classification},
    year = {2021}
}
```

## Contributing

If you'd like to contribute, or have any suggestions, you can contact us at [joycenerd.cs09@nycu.edu.tw](mailto:joycenerd.cs09@nycu.edu.tw) or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.