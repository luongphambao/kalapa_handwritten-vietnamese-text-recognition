import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
config = Cfg.load_config_from_name('vgg_seq2seq')
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
#config["weights"]="weights/seq2seq_new.pth"
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
trainer = Trainer(config, pretrained=False)

trainer.config.save('config_ocr.yml')
trainer.train()
print(trainer.precision())