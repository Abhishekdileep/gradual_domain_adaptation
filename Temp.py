from hear21passt.model.preprocess import AugmentMelSTFT

def load_real_audio(file_path, target_sr=32000, max_len_seconds=10):
    waveform, sr = librosa.load(file_path, sr=target_sr, mono=True)

    max_len_samples = target_sr * max_len_seconds
    if len(waveform) < max_len_samples:
        pad_len = max_len_samples - len(waveform)
        waveform = np.pad(waveform, (0, pad_len))
    else:
        waveform = waveform[:max_len_samples]

    return torch.tensor(waveform, dtype=torch.float32)
model = get_basic_model_2(mode="embed_only")
model.eval()
model = model.cuda()
audio_path = "/sd/scd/s23103/Dcase-2020/TAU-urban-acoustic-scenes-2020-mobile-development/audio"
src_audio = "tram-vienna-285-8638-a.wav"
tgt_audio = "tram-vienna-285-8638-c.wav"


model = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)


value = model(load_real_audio(src_audio))

print(value.shape)

