import numpy as np
from pydub import AudioSegment

def convert_audio_array_to_audio_file(audio_array, output_file_name,sample_rate=16000):
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
        audio_segment.export(output_file_name, format="wav")
        return output_file_name