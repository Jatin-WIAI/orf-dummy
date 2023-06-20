from denoiser import pretrained
from denoiser.dsp import convert_audio
import torchaudio
import torch
import numpy as np
from custom_logger import CustomLogger
import io

logger = CustomLogger(__name__)

class Denoiser():
    def __init__(self,device) -> None:
        """Denoiser class
        """
        if device == -1:
            self.denoising_model = pretrained.dns64().cpu()
            self.device = torch.device('cpu')
        else:
            self.denoising_model = pretrained.dns64().to(torch.device('cuda:'+str(device)))   
            self.device = torch.device('cuda:'+str(device))

    def denoise_audio(self,audio_path:str):
        """Denoise audio file

        Args:
            auido_path (str): Path to audio file

        Returns:
            denoised_wav (np.array): denoised audio array
        """
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav.to(self.device), sr, self.denoising_model.sample_rate, self.denoising_model.chin)
        with torch.no_grad():
            denoised_wav = self.denoising_model(wav)[0]

        return denoised_wav.cpu()
    
    def denoise_audio_by_slicing(self,audio_input:str,chunk_length_s:int=20, stride_length:int=19):
        """Denoise audio file by slicing

        Args:
            auido_input: Path to audio file or audio bytes
            chunk_length_s (int, optional): Length of chunk in seconds. Defaults to 20.
            stride_length (int, optional): Stride length in seconds. Defaults to 19.

        Returns:
            denoised_wav (np.array): denoised audio array
        """
        if type(audio_input) == str:
            wav, sr = torchaudio.load(audio_input)
        else:
            s = io.BytesIO(audio_input)
            wav, sr = torchaudio.load(s)
        wav = convert_audio(wav.to(self.device), sr, self.denoising_model.sample_rate, self.denoising_model.chin)
        denoised_wav = []
        for i in range(0,wav.shape[1],stride_length*self.denoising_model.sample_rate):
            with torch.no_grad():
                denoised_wav.append(self.denoising_model(wav[:,i:i+chunk_length_s*self.denoising_model.sample_rate])[0].cpu())
        # concatenate denosied audio in a sliced fashion
        denoised_wav_final = torch.zeros(wav.shape)
        for i in range(len(denoised_wav)):
            denoised_wav_final[:,i*stride_length*self.denoising_model.sample_rate:i*stride_length*self.denoising_model.sample_rate+chunk_length_s*self.denoising_model.sample_rate] = denoised_wav[i]
        return denoised_wav_final.numpy()

    def get_denoised_wav(self,audio_input):
        """Denoise and return audio file
        Args:
            audio_input (str or bytes): Path to audio file or audio bytes
        Returns:
            denoised_wav (np.array): array of denoised audio
        """
        logger.info("Denoising audio file")
        denoised_wav = self.denoise_audio_by_slicing(audio_input)
        # new_path = audio_path.replace(".wav","_denoised.wav")
        # torchaudio.save(new_path,denoised_wav.cpu(),self.denoising_model.sample_rate)
        return denoised_wav