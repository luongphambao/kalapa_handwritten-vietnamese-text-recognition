# Solution for Deadline team in Kalapa handwritten-vietnamese text recognition


## 1. Installation and Preapare data
### 1.1. Prepare data
```bash
    bash prepare_data.sh
```    
It create file `data.txt`,the file `data.txt` contains all information of data. Each line of `data.txt` has format:
```bash
    <image_path> <label>
```
### 1.2. Installation

- Install requirements:
```bash
pip install -r requirements.txt
```
## 2. Training
### 2.1. Data Augmentation
```bash
python aug.py
```
### 2.2. Data Generator

You follow `TextRecognitionDataGenerator` repo installation (I prepared dictionary file in `dict.txt` file and some useful vietnamese fonts in `TextRecognitionDataGenerator/trdg/fonts/custom` folder)

You can edit `generate_data.sh` file to generate data with your own parameters. 

```bash
bash TextRecognitionDataGenerator/generate_data.sh
```
### 2.3. Training 

You can edit config in `train_vietocr.py` and  `train_vietocr_qualitazation.py` for training with your own parameters. 

```python
dataset_params = {
    'name':'kapapa_vietocr',
    'data_root':'OCR/training_data/images',
    'train_annotation':'train.txt',
    'valid_annotation':'val.txt',
    'image_height':64,
}
params = {
         'print_every':1000,
         'valid_every':1000,
          'iters':100000,
          'checkpoint':'weights/pretrained.pth',    
          'export':'weights/pretrained.pth',
          'metrics': 100000
         }
```
```bash
python train_vietocr.py
```
```bash
python train_vietocr_qualitazation.py
```
## 3. Inference
```bash
python3 predict.py --image_path <image_path> --weights_path <weights_path> --csv_path <csv_path>
```


