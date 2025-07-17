import librosa as lr 
import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
import sys


def load_audio_file(file_path, target_sr=44100, max_len_seconds=10):
    waveform, sr = lr.load(file_path, sr=44100, mono=True)
    
    D = lr.stft(waveform, n_fft=2048, hop_length=512
                , win_length=2048, window='hann')
    return D 

def mix_spectrograms(source_spectrogram, target_spectrogram, alpha=0.6):
    mixed_spectrogram = alpha * source_spectrogram + (1 - alpha) * target_spectrogram
    return mixed_spectrogram

def visualize_spectrogram(spectrogram, title="Spectrogram"):    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max), 
                             sr=44100, x_axis='time', y_axis='log')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    # audio_path  = sys.argv[1] if len(sys.argv[0]) > 1 else sys.error("Please provide the path to the audio files as an argument.")
    # source_audio_file  = sys.argv[2] if len(sys.argv) > 1 else "source.wav"
    # tgt_audio_file = sys.argv[3] if len(sys.argv) > 2 else "target.wav"
    # alpha = sys.argv[4] if len(sys.argv) > 3 else 0.6

    audio_path  = "C:/Users/inkl-2024/Documents/Code/dom_ada/Data/Dcase/audio"
    source_audio_file  = "bus-helsinki-20-806-a.wav"
    tgt_audio_file = "bus-helsinki-20-805-b.wav"
    alpha =  0.6
    src_spectogram = load_audio_file(f"{audio_path}/{source_audio_file}")
    tgt_spectogram = load_audio_file(f"{audio_path}/{tgt_audio_file}")
    mixed_spectrogram = mix_spectrograms(src_spectogram, tgt_spectogram, alpha=alpha)
    visualize_spectrogram(src_spectogram, title="Source Spectrogram")
    visualize_spectrogram(tgt_spectogram, title="Target Spectrogram")
    visualize_spectrogram(mixed_spectrogram, title="Mixed Spectrogram")

