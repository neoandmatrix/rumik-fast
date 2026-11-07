"""Streaming Mimi decoder: give it audio codes a few frames at a time, get exactly the audio a full decode would give.

Hugging Face's MimiModel can stream the decoder *transformer* (KV cache), but its SEANet decoder convolutions have
no streaming state, so decoding frame by frame would click at every boundary. This wrapper runs the same modules
(same weights, called layer by layer) and keeps the history each layer needs between calls:

  causal Conv1d (stride 1)    the full decode left-pads with (kernel-1)*dilation zeros. Streaming: keep the last
                              (kernel-1)*dilation input samples and put them in front of the next chunk (zeros at
                              the start), so every output sample sees exactly the same inputs.
  ConvTranspose1d (upsample)  each input step writes `kernel` output samples, overlapping the next step's by
                              kernel - stride. The full decode keeps the first n*stride samples (causal trim).
                              Streaming: emit n*stride samples, carry the overhanging tail (minus the bias, which
                              the next chunk adds again) and add it to the start of the next chunk (overlap-add).
  transformer                 HF's own KV cache (past_key_values), sliding window 250 steps.

One Mimi frame (12.5 Hz) = 1,920 samples at 24 kHz = 80 ms.
"""
import torch
from transformers.models.mimi.modeling_mimi import MimiConv1d, MimiConvTranspose1d, MimiResnetBlock

SAMPLES_PER_FRAME = 1920


class MimiStreamer:
    def __init__(self, mimi):
        self.m = mimi
        self.reset()

    def reset(self):
        self.state = {}          # per-layer carried samples
        self.kv = None           # decoder transformer cache
        self.frames = 0

    @torch.inference_mode()
    def decode(self, codes):
        """codes [1, 8, n] (long) -> audio [n * 1920] (float, on the codec's device)"""
        dec = self.m.decoder
        lat = self.m.quantizer.decode(codes)                                  # [1, 512, n]
        h = self._convT(dec.upsample, lat) if dec.upsample is not None else lat
        out = dec.decoder_transformer(h.transpose(1, 2), past_key_values=self.kv, use_cache=True)
        self.kv = out.past_key_values
        h = out.last_hidden_state.transpose(1, 2)
        for layer in dec.decoder_seanet.layers:
            h = self._layer(layer, h)
        self.frames += codes.shape[-1]
        return h[0, 0]

    def _layer(self, layer, h):
        if isinstance(layer, MimiConv1d):
            return self._conv(layer, h)
        if isinstance(layer, MimiConvTranspose1d):
            return self._convT(layer, h)
        if isinstance(layer, MimiResnetBlock):
            res = h
            for sub in layer.block:
                h = self._layer(sub, h)
            assert isinstance(layer.shortcut, torch.nn.Identity)             # Mimi: use_conv_shortcut = False
            return res + h
        return layer(h)                                                        # ELU: no state

    def _conv(self, layer, h):
        assert layer.causal and int(layer.stride) == 1
        p = int(layer.kernel_size) - 1                                         # (kernel-1)*dilation
        if p == 0:
            return layer.conv(h)
        prev = self.state.get(id(layer))
        if prev is None:
            prev = h.new_zeros(h.shape[0], h.shape[1], p)                      # = the full decode's zero padding
        x = torch.cat([prev, h], -1)
        self.state[id(layer)] = x[..., -p:]
        return layer.conv(x)

    def _convT(self, layer, h):
        conv = layer.conv
        s, k = conv.stride[0], conv.kernel_size[0]
        assert layer.padding_left == 0 and layer.padding_right == k - s       # causal, trim_right_ratio = 1
        raw = conv(h)                                                          # [.., n*s + (k - s)]
        carry = self.state.get(id(layer))
        if carry is not None:
            raw = raw.clone()
            raw[..., :k - s] += carry
        n = h.shape[-1] * s
        tail = raw[..., n:]
        if conv.bias is not None:
            tail = tail - conv.bias[None, :, None]                             # the next chunk adds it again
        self.state[id(layer)] = tail
        return raw[..., :n]
