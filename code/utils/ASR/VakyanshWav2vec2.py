from typing import Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM,Wav2Vec2Processor, pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
sys.path.append(script_dir)

from CTCDecoder import CTCDecoder
from custom_logger import CustomLogger
logger = CustomLogger(__name__)

class VakyanshWav2vec2():
    def __init__(self, model_path: str, processor_path: str, use_decoder: bool = True) -> None:
        """
        Wav2vec2 class for HuggingFace Wav2vec2 model
        Args:
            model_path: path to the model
            processor_path: path to the processor
            use_decoder: whether to use decoder or not
        """

        self.model = None
        self.processor = None
        self.model_type = "wav2vec2"
        self.use_decoder = use_decoder
        self.model_path = model_path
        self.processor_path = processor_path
        self.name = self.model_path+"_"+self.processor_path+"_"+self.model_type
        pass

    def load_model_processor(self):
        """
        Load model and processor
        Returns:
            model_processor_dict: dictionary containing model and processor
        """
        assert len(self.model_path)!=0

        model =  Wav2Vec2ForCTC.from_pretrained(self.model_path)
        if len(self.processor_path)==0:
            processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        else:
            processor = Wav2Vec2Processor.from_pretrained(self.processor_path)
        return {
            "model": model,
            "processor": processor
        }
    
    def get_pipeline(self,decoder_path: str="", device: int=0):
        """
        Get pipeline for inference
        Args:
            decoder_path: path to decoder
            device: device to use
        Returns:
            pipe: pipeline for inference
        """
        assert self.model is not None 
        pipe = VakyanshWav2vec2Pipeline(self,decoder_path,device)
        return pipe
    
    def register(self):
        """
        Register model and processor
        Returns:
            self: wav2vec2 object
        """
        assert len(self.model_path)!=0
        model_processor_dict = self.load_model_processor()
        self.model = model_processor_dict["model"]
        self.processor = model_processor_dict["processor"]
        return self
    

        
class VakyanshWav2vec2Pipeline():
    def __init__(self, model_obj: VakyanshWav2vec2,decoder_path: str, device: int=0) -> None:
        """
        Wav2vec2 pipeline class for HuggingFace Wav2vec2 model
        Args:
            model_obj: model object
            decoder_path: path to the decoder
            device: device to use
        """

        self.model = model_obj.model
        self.processor = model_obj.processor
        self.decoder_path = decoder_path
        self.device = device
        self.pipe_type = "vakyansh_wav2vec2"
        self.use_decoder = model_obj.use_decoder
        if self.use_decoder: 
            self.name = model_obj.name+"_"+self.decoder_path+"_"+self.pipe_type
        else:
            self.name = model_obj.name+"_"+self.pipe_type
        self.pipe = self.get_pipeline()
        pass
    
    def load_decoder(self,decoder_path):
        """
        Load decoder for Language model
        Args:
            decoder_path: path to the decoder
        Returns:
            decoder: decoder object
        """
        decoder_obj = CTCDecoder()
        if len(decoder_path)!=0 and self.use_decoder:
            decoder = decoder_obj.load_decoder(self.processor,decoder_path)
            return decoder
        else:
            return None
        
    def get_pipeline(self):
        """
        Get pipeline for inference
        Returns:
            pipe: pipeline for inference
        """
        assert self.model is not None
        self.decoder = self.load_decoder(self.decoder_path) 
        if self.decoder is None:
            pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor,device=self.device)
        else:
            pipe = pipeline("automatic-speech-recognition", model=self.model, tokenizer=self.processor.tokenizer, feature_extractor=self.processor.feature_extractor,decoder=self.decoder,device=self.device)
        return pipe

    def __call__(self,audio_file: str, return_timestamps: str="word",chunk_length_s: float=10 ) -> Any:
        """
        Call pipeline for inference
        Args:
            audio_file: audio file
            return_timestamps: return timestamps
            chunk_length_s: chunk length in seconds
        Returns:
            output: output of the pipeline. The output is a list of dictionaries with keys "text" and "chunks". "text" contains the transcribed text and "chunks" contains the timestamps of the words in the text
        """
        output = self.pipe(audio_file,return_timestamps=return_timestamps,chunk_length_s=chunk_length_s)
        new_output = {}
        new_output["text"] = output["text"].replace("<s>","").replace("</s>","")
        new_output["chunks"]  = []
        for chunk in output["chunks"]:
            temp_dict = {}
            temp_dict = chunk
            temp_dict["text"] = chunk["text"].replace("<s>","").replace("</s>","")
            if len(temp_dict["text"])==0:
                continue
            new_output["chunks"].append(temp_dict)
        return new_output
