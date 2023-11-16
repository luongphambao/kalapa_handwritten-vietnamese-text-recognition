import matplotlib.pyplot as plt
from PIL import Image
import torch 
from torch import nn
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import os
from vietocr.tool.translate import build_model
import argparse 
import pandas as pd
def parse_args():
    #img_folder and csv_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', type=str, default='OCR/public_test/images/')
    parser.add_argument('--csv_path', type=str, default='submit_public_test.csv')
    parser.add_argument('--weights', type=str, default='weights/ocr.pth')
    args = parser.parse_args()
    return args
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
if __name__ == '__main__':
    args = parse_args()
    data_path=args.img_folder
    csv_path=args.csv_path
    weights_path=args.weights
    # Pytorch support only cpu device
    config = Cfg.load_config_from_name('vgg_seq2seq')
    config['device'] = 'cpu'
    config['cnn']['pretrained']=False
    config['weights'] = weights_path
    config['predictor']['beamsearch']=False
    #config['dataset']['image_height']=64
    model, vocab = build_model(config)
    # fuse layer
    model = model.to('cpu')
    model.eval()
    for m in model.cnn.model.modules():
        if type(m) == nn.Sequential:
            for n, layer in enumerate(m):
                if type(layer) == nn.Conv2d:
                    torch.quantization.fuse_modules(m, [str(n), str(n + 1), str(n + 2)], inplace=True)
    # prepare model for quantize aware training
    quantized_cnn = QuantizedCNN(model_fp32=model.cnn)
    quantized_cnn.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    # Print quantization configurations
    print(quantized_cnn.qconfig)

    # the prepare() is used in post training quantization to prepares your model for the calibration step
    quantized_cnn = torch.quantization.prepare_qat(quantized_cnn, inplace=True)
    quantized_cnn = quantized_cnn.to(torch.device('cpu'))
    model.cnn = torch.quantization.convert(quantized_cnn, inplace=True)   
    # create detector
    detector = Predictor(config, qmodel=model)
    list_id=[]
    list_answer=[]
    list_prob=[]
    
    for fol in os.listdir(data_path):
        fol_path=os.path.join(data_path,fol)
        for img_name in os.listdir(fol_path):
            img_path=os.path.join(fol_path,img_name)
            img = Image.open(img_path)
            list_id.append(fol+"/"+img_name)
            s,prob = detector.predict(img,return_prob=True)
            print(s,prob)
            list_answer.append(s)
            list_prob.append(prob)
    df = pd.DataFrame(list(zip(list_id, list_answer)), columns =["id","answer"])
    df.to_csv(csv_path,index=False)
    df_prob = pd.DataFrame(list(zip(list_id,list_answer,list_prob)), columns =["id","answer","prob"])
    df_prob.to_csv(csv_path[:-4]+"_prob.csv",index=False)