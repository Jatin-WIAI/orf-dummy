import subprocess
import os
os.system('apt-get update')
os.system('apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev')
subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
subprocess.run(['pip', 'install','denoiser'])
subprocess.run(['pip', 'install', 'hydra-core==1.0.7'])
from utils.orf_master import ORFMaster
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM,Wav2Vec2Processor, pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import logging
import numpy as np
import json
import io
import tarfile
# def load_model_processor(model_id):
#     model =  Wav2Vec2ForCTC.from_pretrained(model_id)
#     processor = Wav2Vec2Processor.from_pretrained(model_id)
#     return {
#         "model": model,
#         "processor": processor
#     }
def input_fn(input_data, content_type='application/json'):  
    logger.info("Input data is processed")
    logger.info("Content type sent to the model: {}".format(content_type))
    logger.info("Input data type is {}".format(type(input_data)))
    # input_data = bytes(input_data) 
    if type(input_data) == str:
        with open(input_data, 'rb') as f:
            input_data_bytes = f.read()
    else:
        input_data_bytes = input_data
    io_bytes = io.BytesIO(input_data_bytes)
    tar = tarfile.open(fileobj=io_bytes, mode='r')
    f=tar.extractfile("ref_text.txt")
    content=f.read()
    f_audio=tar.extractfile("audio.mp3")
    content_audio=f_audio.read()

    input_data = {
        "audio": content_audio,
        "ref_text": str(content,"utf-8")
    }
    logger.info("Input data type sent to model is {}".format(type(input_data)))
    return input_data

logger = logging.getLogger()
def model_fn(model_dir):
# Load model from HuggingFace Hub
    config_path = os.path.join(model_dir, "config.json")
    config = json.load(open(config_path))
    orf_obj = ORFMaster(config, device=0)
    # print(pipe)
    return orf_obj

def predict_fn(data, orf_obj):
    logger.info("Data Type in predict function is {}".format(type(data)))
    ref_text= data["ref_text"]
    audio_data = data["audio"]
    output = orf_obj.preprocess_audio_get_results(audio_data, ref_text)
    return output
