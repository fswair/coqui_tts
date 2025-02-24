import re
import whisper
from pydub import AudioSegment

class Whisper:
    whisper_model: str = "base"
    def __init__(self, audio_path: str, word_timestamps: bool = True):
        self.audio_path = audio_path
        self.word_timestamps = word_timestamps
        
        self.model = whisper.load_model(self.whisper_model)
        self.audio = whisper.load_audio(self.audio_path)
        self.extras = []
        
        self.first_dummy = None
        self.last_dummy = None
            
    def to_file(self, output_path: str, dummy_params: tuple[int], format: str = "wav"):
        result = self.model.transcribe(self.audio, word_timestamps=self.word_timestamps)
        dummies = [re.compile(dummy, re.I) for dummy in dummy_params]
        
        print(result)
        
        for segment in result["segments"]:
            for word in segment["words"]:
                if not any(self.verify(word["word"], dummies)): continue
                print("word: ", word)
                if self.first_dummy is None:
                    self.first_dummy = word
                elif self.last_dummy is None:
                    self.last_dummy = word
                else:
                    self.extras.append(word)
        try:
            start = self.first_dummy["end"]
            end = self.last_dummy["start"]
        except Exception as e:
            print(e)
            print(self.first_dummy)
            print(self.last_dummy)
            print(f"dummy parameters can not retrieved.\nfirst: {self.first_dummy} - last: {self.last_dummy}")
            return
        audio_segment = AudioSegment.from_wav(self.audio_path)

        start_ms = start * 1000
        end_ms = end * 1000
        
        cropped_audio = audio_segment[start_ms:end_ms]
        cropped_audio.export(output_path, format=format)
    
    
    def verify(self, text: str, dummies: list[str]):
        for dummy in dummies:
            yield dummy.search(text)

whisper = Whisper("output.wav")

whisper.to_file(
    output_path="origin.wav",
    dummy_params=("istanbul", "ankara")
)
