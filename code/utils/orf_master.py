import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir)

from amplitude_normalizer import AmplitudeNormalizer
from denoiser_model import Denoiser
from ASR.ASRModelLoader import ASRModelLoader
from ASR.ASRPipeline import ASRPipeline
from custom_logger import CustomLogger
from typing import Dict

logger = CustomLogger(__name__)

class ORFMaster():
    def __init__(self, config: Dict, device: int) -> None:
        """ ORFMaster is the main class that handles all the operations of the ORF pipeline. It is responsible for loading all the models and pipelines and also for orchestrating the entire pipeline.

        Args:
            config (Dict): Configuration dictionary
            device (int): Device to be used for inference, 0 for GPU and -1 for CPU
        """
        self.config = config
        self.lang = config["lang"]
        self.device = device
        self.denoiser_obj = self.load_denoiser(device)
        self.amp_normalizer_obj = self.load_amp_normalizer(config["target_dBFS"])
        self.model_obj_list = self.load_models()
        self.pipelines_list = self.load_pipelines()
        logger.info("ORFMaster initialized")
    
    def load_denoiser(self,device):
        """Loads the denoiser object

        Returns:
            _type_: Denoiser object
        """
        return Denoiser(device=device)
    
    def load_amp_normalizer(self, target_dBFS):
        """Loads the amplitude normalizer object
        Args:
            target_dBFS (int): Target dBFS value
        Returns:
            _type_: AmplitudeNormalizer object
        """
        return AmplitudeNormalizer(target_dBFS=target_dBFS)
    
    def load_models(self):
        """Loads all the ASR models
        Returns:
            _type_: List of ASRModel objects
        """
        model_obj_list = []
        for model_dict in self.config["models"]:
            model_obj = ASRModelLoader(model_dict).get_model()
            model_obj_list.append(model_obj)
        logger.info("All ASR Models loaded")
        return model_obj_list

    def load_pipelines(self):
        """Loads all the ASR pipelines
        Returns:
            _type_: List of ASRPipeline objects
        """
        pipelines_list = []
        for model_obj in self.model_obj_list:
            if model_obj.use_decoder == True:
                for decoder_path in self.config["decoder_path_list"]:
                    pipelines_list.append(ASRPipeline(model_obj,decoder_path,self.device).register())
            else:
                pipelines_list.append(ASRPipeline(model_obj,"",self.device).register())
        logger.info("All ASR Pipelines loaded")
        return pipelines_list
    
    def get_asr_outputs(self,audio_file:str):
        """Gets the ASR outputs for all the ASR pipelines
        Args:
            audio_file (str): Path to the audio file
        Returns:
            List: List of ASR outputs
        """
        asr_outputs = []
        for pipe in self.pipelines_list:
            # print(get_asr_output(audio_file,pipe))
            asr_output = pipe.get_asr(audio_file)
            asr_outputs.append(asr_output)
        logger.info("ASR outputs obtained")
        return asr_outputs
    
    def get_denoised_audio(self,audio_input):
        """Gets the denoised wav file
        Args:
            audio_input (str or bytes): Path to the audio file or bytes of the audio file
        Returns:
            np.array: Numpy array of the denoised wav file 
        """
        denoised_audio_array =  self.denoiser_obj.get_denoised_wav(audio_input)
        logger.info("Audio denoised")
        return denoised_audio_array
    
    def get_normalized_audio(self,audio_input):
        """Gets the normalized wav file
        Args:
            audio_input (str or np.array): Path to the audio file or numpy array of the audio file
        Returns:
            np.array: Numpy array of the normalized wav file
        """
        amp_normalized =  self.amp_normalizer_obj.get_normalized_audio(audio_input)
        logger.info("Audio normalized")
        return amp_normalized
    
    def get_preprocessed_audio(self,audio_input):
        """Gets the preprocessed audio file. Performs denoising and amplitude normalization
        Args:
            audio_input (str or bytes): Path to the audio file or bytes of the audio file
        Returns:
            np.array: normalized audio array
        """
        denoised_audio= self.get_denoised_audio(audio_input)
        normalized_audio = self.get_normalized_audio(denoised_audio)
        logger.info("Audio preprocessed")
        return normalized_audio
    
    def preprocess_audio_get_results(self,audio_input, ref_text:str):
        """Preprocesses the audio file and gets the results
        Args:
            audio_input (str or bytes): Path to the audio file or bytes of the audio file
            ref_text (str): Reference text
        Returns:
            tuple: Final transcript, matched dataframe, metrics, recommendation, preprocessed audio array
        """
        preprocessed_audio_array= self.get_preprocessed_audio(audio_input)
        # preprocessed_audio = preprocessed_audio.squeeze()
        asr_outputs = self.get_asr_outputs(preprocessed_audio_array.squeeze())
        logger.info("Results obtained after preprocessing")
        return asr_outputs