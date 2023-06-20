from custom_logger import CustomLogger
logger = CustomLogger(__name__)

class ASRPipeline():
    def __init__(self,model_obj,decoder_path="", device=0) -> None:
        self.pipeline = None
        self.model_obj = model_obj
        self.decoder_path = decoder_path
        self.device = device

        pass

    def load_pipeline(self):
        pipeline = self.model_obj.get_pipeline(self.decoder_path,self.device)
        return pipeline
    
    def register(self):
        self.pipeline = self.load_pipeline()
        return self
    
    def get_asr(self,audio_file, return_timestamps="word",chunk_length_s=10 ):
        
        if self.pipeline is None:
            logger.error("Pipeline not registered. Please register a pipeline using register() method.")
            raise Exception("Pipeline not registered. Please register a pipeline using register() method.")
        logger.info("Getting asr from ASR Pipeline: {}".format(self.pipeline.name))
        output = self.pipeline(audio_file,return_timestamps=return_timestamps,chunk_length_s=chunk_length_s)
        logger.info("ASR Pipeline: {} completed".format(self.pipeline.name))
        
        return output