import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

from Wav2vec2 import Wav2vec2
from VakyanshWav2vec2 import VakyanshWav2vec2

class ASRModelLoader():
    def __init__(self, model_dict) -> None:
        self.supported_model_types = ["wav2vec2", "vakyansh_wav2vec2"]
        self.model_dict = model_dict
        self.model_type = model_dict["model_type"]
        self.model_path = model_dict["model_path"]
        self.processor_path = model_dict["processor_path"]  
        try: 
            self.use_decoder = model_dict["use_decoder"]
        except:
            self.use_decoder = True
        pass

    def get_model(self):
        assert self.model_type in self.supported_model_types
        if self.model_type == "wav2vec2":
            model_obj = Wav2vec2(self.model_path, self.processor_path, self.use_decoder).register()
            return model_obj
        if self.model_type == "vakyansh_wav2vec2":
            model_obj = VakyanshWav2vec2(self.model_path, self.processor_path, self.use_decoder).register()
            return model_obj
        else:
            return None
        
    