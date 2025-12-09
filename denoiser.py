import sys
import torch
from stft import TorchSTFT as STFT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Denoiser(torch.nn.Module):

    def __init__(self, istftnet, filter_length=1024, n_overlap=4, win_length=1024, window= "hann",mode="zeros"):
        super(Denoiser, self).__init__()

        self.stft = STFT(
            filter_length=filter_length,
            hop_length=int(filter_length / n_overlap),
            win_length=win_length,
            window=window
        ).to(DEVICE)


        mel_size = (1,80,88)
        if mode == "zeros":
            mel_input = torch.zeros(mel_size).to(DEVICE)
        elif mode == "normal":
            mel_input = torch.randn(mel_size).to(DEVICE)
        elif mode == "mean": #Experimental
            mel_input = torch.full(mel_size, -60).to(DEVICE)
        elif mode == "silence": #Experimental
            mel_input = torch.full(mel_size, -11.5129).to(DEVICE)
        elif mode == "ones": #Experimental
            mel_input = torch.ones(mel_size).to(DEVICE)
        
        else:
            raise Exception("Mode {} if not supported".format(mode))
        

        with torch.no_grad():
            spec, phase = istftnet(mel_input)
        
        self.register_buffer("bias_spec", spec[:, :, 0][:, :, None])
    

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.to(DEVICE).float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)

        return audio_denoised

        
