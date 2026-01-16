import torch
import torchaudio
import numpy as np
from pathlib import Path
from hubconf_offline import knn_vc  # Import the knn_vc function from hubconf.py


knnvc_model = knn_vc(pretrained=True, prematched=True, device='cpu')

# path to 16kHz, single-channel, source waveform
src_wav_path = './src/src.wav'
# list of paths to all reference waveforms (each must be 16kHz, single-channel) from the target speaker
ref_wav_paths = ['./ref/ref1.wav', ]

query_seq = knnvc_model.get_features(src_wav_path)
print(query_seq.shape)
matching_set = knnvc_model.get_matching_set(ref_wav_paths)
print(matching_set.shape)

out_wav = knnvc_model.match(query_seq, matching_set, topk=4)
print(out_wav.shape)

torchaudio.save('./out/out.wav', out_wav[None], 16000)


