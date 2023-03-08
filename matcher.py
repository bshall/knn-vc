
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from hifigan.models import Generator as HiFiGAN
from hifigan.utils import AttrDict
from torch import Tensor
from torchaudio.sox_effects import apply_effects_tensor
from wavlm.WavLM import WavLM


SPEAKER_INFORMATION_WEIGHTS = [
    0, 0, 0, 0, 0, 0,  # layer 0-5
    1.0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, # layer 15
    0, 0, 0, 0, 0, 0, # layer 16-21
    0, # layer 22 
    0, 0 # layer 23-24
]


def fast_cosine_dist(source_feats, matching_pool):
    """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).cpu()
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].cpu(), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None].cpu() * matching_norms[None]) )
    return dists


class KNN_VC(nn.Module):

    def __init__(self,
        wavlm: WavLM,
        hifigan: HiFiGAN,
        hifigan_cfg: AttrDict,
        device='cuda'
    ) -> None:
        """ kNN-VC matcher. 
        Arguments:
            - `wavlm` : trained WavLM model
            - `hifigan`: trained hifigan model
            - `hifigan_cfg`: hifigan config to use for vocoding.
        """
        super().__init__()
        # set which features to extract from wavlm
        self.weighting = torch.tensor(SPEAKER_INFORMATION_WEIGHTS, device=device)[:, None]
        # load hifigan
        self.hifigan = hifigan.eval()
        self.h = hifigan_cfg
        # store wavlm
        self.wavlm = wavlm.eval()
        self.device = torch.device(device)
        self.sr = self.h.sampling_rate

    def get_matching_set(self, wavs: list[Path] | list[Tensor], weights=None) -> Tensor:
        """ Get concatenated wavlm features for the matching set using all waveforms in `wavs`, 
        specified as either a list of paths or list of loaded waveform tensors of 
        shape (channels, T), assumed to be of 16kHz sample rate.
        Optionally specify custom WavLM feature weighting with `weights`.
        """
        feats = []
        for p in wavs:
            feats.append(self.get_features(p, weights=self.weighting if weights is None else weights))
        
        feats = torch.concat(feats, dim=0).cpu()
        return feats
        

    @torch.inference_mode()
    def vocode(self, c: Tensor) -> Tensor:
        """ Vocode features with hifigan. `c` is of shape (bs, seq_len, c_dim) """
        y_g_hat = self.hifigan(c)
        y_g_hat = y_g_hat.squeeze(1)
        return y_g_hat


    @torch.inference_mode()
    def get_features(self, path, weights=None, vad_trigger_level=7):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim)"""
        # load audio
        if weights == None: weights = self.weighting
        if type(path) in [str, path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]
        assert sr == self.sr, "input audio sample rate must be 16kHz."
        
        # trim silence from front and back
        transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
        x_front_trim = transform(x)
        waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
        waveform_reversed_front_trim = transform(waveform_reversed)
        waveform_end_trim, sr = apply_effects_tensor(
            waveform_reversed_front_trim, sr, [["reverse"]]
        )
        x = waveform_end_trim
        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        rep, layer_results = self.wavlm.extract_features(wav_input_16khz, output_layer=self.wavlm.cfg.encoder_layers, ret_layer_results=True)[0]
        features = torch.cat([x.transpose(0, 1) for x, _ in layer_results], dim=0) # (n_layers, seq_len, dim)

        # save full sequence
        features = ( features*weights[:, None] ).sum(dim=0) # (seq_len, dim)
        return features


    @torch.inference_mode()
    def match(self, query_seq: Tensor, matching_set: Tensor, synth_set: Tensor = None, topk: int = 4, tgt_loudness_db: float | None = -16) -> Tensor:
        """ Given `query_seq`, `matching_set`, and `synth_set` tensors of shape (N, dim), perform kNN regression matching
        with k=`topk`. Inputs:
            - `query_seq`: Tensor (N1, dim) of the input/source query features.
            - `matching_set`: Tensor (N2, dim) of the matching set used as the 'training set' for the kNN algorithm.
            - `synth_set`: optional Tensor (N2, dim) corresponding to the matching set. We use the matching set to assign each query
                vector to a vector in the matching set, and then use the corresponding vector from the synth set during HiFiGAN synthesis.
                By default, and for best performance, this should be identical to the matching set. 
            - `topk`: k in the kNN -- the number of nearest neighbors to average over.
            - `tgt_loudness_db`: float db used to normalize the output volume. Set to None to disable. 
        Returns:
            - converted waveform of shape (T,)
        """
        if synth_set is None: synth_set = matching_set
        dists = fast_cosine_dist(query_seq.cpu(), matching_set.cpu())
        best = dists.topk(k=topk, largest=False, dim=-1)
        out_feats = synth_set[best.indices].mean(dim=1).cpu()
        
        prediction = self.vocode(out_feats[None].cuda()).cpu().squeeze()
        
        # normalization
        if tgt_loudness_db is not None:
            src_loudness = torchaudio.functional.loudness(prediction[None], self.h.sampling_rate)
            tgt_loudness = tgt_loudness_db
            pred_wav = torchaudio.functional.gain(prediction, tgt_loudness - src_loudness)
        else: pred_wav = prediction
        return pred_wav


