import torch
import librosa
from hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings , get_basic_model_2
from hear21passt.passt import get_model_passt
from hear21passt.wrapper import PasstBasicWrapper2
from dataset.Augment_mel_stft_mixup import AugmentMelSTFT2
import numpy as np 

def load_real_audio(file_path, target_sr=32000, max_len_seconds=10):
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)

    max_len_samples = target_sr * max_len_seconds
    if len(waveform) < max_len_samples:
        pad_len = max_len_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:max_len_samples]

    return torch.tensor(waveform, dtype=torch.float32)

##########################
###   Model loading   ####
##########################

# model = get_basic_model_2(mode="embed_only")
# model.eval()
# model = model.cuda()


#############################
### Custom model loading ####
#############################

mel = AugmentMelSTFT2(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                        timem=192,
                        htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                        fmax_aug_range=2000)
net = get_model_passt(arch="passt_s_swa_p16_128_ap476")
model = PasstBasicWrapper2(mel=mel, net=net)




audio_path = "/sd/scd/s23103/Dcase-2020/TAU-urban-acoustic-scenes-2020-mobile-development/audio"
src_audio = "tram-vienna-285-8638-a.wav"
tgt_audio = "tram-vienna-285-8638-c.wav"


###########################################################
####### Get embedding for two audio files using mixup #####
##### The Spectograms are mixed up using a alpha value ####
###########################################################

def get_embedding(model, audio_src, audio_tgt, mixup_alpha=0.6):
    with torch.no_grad():
        audio_src = load_real_audio(f"{audio_path}/{src_audio}")
        audio_tgt = load_real_audio(f"{audio_path}/{tgt_audio}")
        audio_src = audio_src.unsqueeze(0)
        audio_src = audio_src.cuda()
        audio_tgt = audio_tgt.unsqueeze(0)
        audio_tgt = audio_tgt.cuda()
        embedding = model(audio_src ,audio_tgt , 0.6)
    return embedding






