import torch
from TTS.api import TTS
from pydantic import BaseModel

class VoiceGen(BaseModel):
    speaker : str
    tts_model: str
    language: str = "en"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def tts(self):
        return TTS(self.tts_model).to(self.device)
    
    def to_file(self, text: str, dummy_params: tuple[str], output_path: str, split_sentences: bool = None, **options):
        head, tail = dummy_params[:2]
        text = f"{head} {text} {tail}"
        return self.tts.tts_to_file(text=text, speaker_wav=self.speaker, language=self.language, file_path=output_path, split_sentences=bool(split_sentences), **options)

tts = VoiceGen(
    speaker="LJ001-0004.mp3",
    tts_model="tts_models/multilingual/multi-dataset/xtts_v2",
    language="tr"
)

tts.to_file(
    text="kırmızı",
    dummy_params=("istanbul", "ankara"),
    output_path="output.wav",
    split_sentences=True
)
