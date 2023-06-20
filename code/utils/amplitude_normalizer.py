import os
from pydub import AudioSegment
import numpy as np

from custom_logger import CustomLogger
logger = CustomLogger(__name__)
class AmplitudeNormalizer():
    def __init__(self, target_dBFS=-30, increase_only=False, decrease_only=False, verbose=False):
        """Audio normalizer
        Args:
            target_dBFS (int, optional): Target dBFS. Defaults to -30.
            increase_only (bool, optional): Increase only. Defaults to False.
            decrease_only (bool, optional): Decrease only. Defaults to False.
            verbose (bool, optional): Verbose. Defaults to False.
        """
        self.target_dBFS = target_dBFS
        self.increase_only = increase_only
        self.decrease_only = decrease_only
        self.verbose = verbose

    def normalize_audiosegment(self, audio_segment: AudioSegment):
        """Normalize audio segment
        Args:
            audio_segment (pydub.AudioSegment): Audio segment
        Returns:
            normalized_audio (pydub.AudioSegment): normalized audio segment
        """

        change_in_dBFS = self.target_dBFS - audio_segment.dBFS
        if self.verbose:
            print("File dBFS: ", audio_segment.dBFS)
            print("Target dBFS: ", self.target_dBFS)
            print("Change in dBFS: ", change_in_dBFS)
        if self.increase_only:
            if change_in_dBFS < 0:
                change_in_dBFS = 0
        if self.decrease_only:
            if change_in_dBFS > 0:
                change_in_dBFS = 0
        normalized_audio = audio_segment.apply_gain(change_in_dBFS)
        return normalized_audio

    def normalizer_nparray(self, audio_array: np.array):
        """Normalize audio array
        Args:
            audio_array (np.array): Audio array
        Returns:
            adjusted_audio (np.array): Adjusted audio array
        """
        desired_linear = 10 ** (self.target_dBFS/ 20)
        current_db = 20 * np.log10(np.sqrt(np.mean(audio_array ** 2)))
        scaling_factor = desired_linear / (10 ** (current_db / 20))
        adjusted_audio = audio_array * scaling_factor

        return adjusted_audio
    
    def convert_audio_array_to_audio_segment(self,audio_array,sample_rate=16000):
        """Convert audio array to audio segment

        Args:
            audio_array (np.array): Audio array
            sample_rate (int, optional): Sample rate. Defaults to 16000.

        Returns:
            audio_segment (pydub.AudioSegment): Audio segment
        """
        if audio_array.dtype != np.int16:
            audio_array = np.int16(audio_array * (2**15 - 1))
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )
        return audio_segment
    
    def convert_audio_segment_to_bytes(self,audio_segment):
        """Convert audio segment to bytes

        Args:
            audio_segment (pydub.AudioSegment): Audio segment

        Returns:
            audio_bytes (bytes): Audio bytes
        """
        audio_bytes = audio_segment.export().read()
        return audio_bytes
    
    def get_normalized_audio(self,audio_input):
        """Normalize audio file

        Args:
            audio_input (str or array): Path to audio file or audio numpy array

        Returns:
            normalized_audio_array (numpy array): normalized audio array
        """
        logger.info("Normalizing audio")
        if isinstance(audio_input,str):
            audio_segment = AudioSegment.from_wav(audio_input)
            normalized_audio = self.normalize_audiosegment(audio_segment)
            normalized_audio_array = self.convert_audio_segment_to_bytes(normalized_audio)
            return normalized_audio_array
            # return normalized_audio
        else:
            normalized_audio_array = self.normalizer_nparray(audio_input)
        # new_path = audio_path.replace(".wav","_norm.wav")
        # normalized_audio.export(new_path,format="wav")
        return normalized_audio_array