import torch
from hear21passt.base import get_basic_model, get_model_passt
from hear21passt.models.preprocess import AugmentMelSTFT
from dataset_passt import AugmentMelSTFT2
from wrapper import PasstBasicWrapper2
import ipdb
# high-res pre-trained on Audioset
net = get_model_passt(arch="passt_s_swa_p16_128_ap476")

# hopsize=160 for this pretrained model
mel = AugmentMelSTFT2(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000)

model = PasstBasicWrapper2(net=net, mode="embed_only")
model = model.cuda()
model.eval()
seconds = 10
audio1 = torch.ones((32, 32000 * seconds))
audio2 = torch.ones((32, 32000 * seconds))
alpha = 0.5
mel = mel.cuda()
ipdb.set_trace()
audio3 = torch.load('saved_files/first_wave.pt')
with torch.no_grad():
    audio1 = audio1.cuda()
    audio2 = audio2.cuda()
    alpha = torch.tensor([alpha]).cuda()
    audio1 = mel(audio1, audio2, alpha)
    audio1 = audio1.unsqueeze(1)
    # forward pass
    output = model(audio1)
    print(f"Final output : {output.shape}")  # should be (3, 128, 192)