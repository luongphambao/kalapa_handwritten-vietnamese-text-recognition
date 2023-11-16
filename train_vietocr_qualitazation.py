import matplotlib.pyplot as plt
from PIL import Image
import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.tool.translate import build_model
from torch import nn
from vietocr.model.trainer import Trainer
import os 
def save_models(model, file_name):
    output_path = './weights/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)   
    saved_path = os.path.join(output_path, file_name)
    if os.path.exists(saved_path):
        os.remove(saved_path)   
    print('Save files in: ', saved_path)
    torch.save(model.state_dict(), saved_path)
    
def save_torchscript_model(model, file_name):
    output_path = './weights/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)   
    model_filepath = os.path.join(output_path, file_name)
    torch.jit.save(torch.jit.script(model), model_filepath)
    print('Save in: ', model_filepath)
    return model_filepath

def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model
config = Cfg.load_config_from_name('vgg_seq2seq')
dataset_params = {
    'name':'kapapa_vietocr',
    'data_root':'OCR/training_data/images',
    'train_annotation':'final_train.txt',
    'valid_annotation':'val.txt',
    'image_height':64,
}

params = {
         'print_every':1000,
         'valid_every':1000,
          'iters':100000,
          'checkpoint':'./weights/seq2seq_quantize_new4.pth',    
          'export':'./weights/seq2seq_quantize_new4.pth',
          'metrics': 10000
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'
config['cnn']['pretrained']=False
config['weights'] = "weights/transformerocr.pth"
device = config['device']
model, vocab = build_model(config)
weights = config['weights']
model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
class QuantizedCNN(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedCNN, self).__init__()
        
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x
#model = model.train()
model=model.to(device)
model.eval()
for m in model.cnn.model.modules():
    if type(m) == nn.Sequential:
        for n, layer in enumerate(m):
            if type(layer) == nn.Conv2d:
                torch.quantization.fuse_modules(m, [str(n), str(n + 1), str(n + 2)], inplace=True)
quantized_cnn = QuantizedCNN(model_fp32=model.cnn)
quantized_cnn.qconfig = torch.quantization.get_default_qconfig("fbgemm")

# Print quantization configurations
print(quantized_cnn.qconfig)

# the prepare() is used in post training quantization to prepares your model for the calibration step
quantized_cnn = torch.quantization.prepare_qat(quantized_cnn, inplace=True)
model.cnn = quantized_cnn
   
model.train()
model = model.to(device)
trainer = Trainer(qmodel=model, config=config, pretrained=False)
trainer.train()