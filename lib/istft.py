# Originally adapted from conv-stft (https://github.com/echocatzh/conv-stft)
# and then rewritten for ONNX export with DFT + polyphase overlap-add.

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.onnx import symbolic_helper


class _InverseDFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spec, n_fft):
        n_fft = int(n_fft)
        time = torch.fft.ifft(torch.view_as_complex(spec), n=n_fft, dim=1)
        return torch.view_as_real(time)

    @staticmethod
    def symbolic(g, spec, n_fft):
        n_fft = symbolic_helper._get_const(n_fft, 'i', 'n_fft')
        dft_length = g.op(
            'Constant',
            value_t=torch.tensor(n_fft, dtype=torch.int64)
        )
        # Axis=1 means FFT over frequency bins for [B, F, T, 2].
        return g.op('DFT', spec, dft_length, axis_i=1, inverse_i=1, onesided_i=0)


class iSTFT(nn.Module):
    def __init__(
            self, win_len=1024, win_hop=512, fft_len=1024,
            window=None, enframe_mode='continue',
            win_sqrt=False
    ):
        """
        iSTFT implementation for ONNX export.

        This version reconstructs full spectrum from one-sided input,
        performs inverse transform through ONNX DFT (inverse mode), and
        applies overlap-add using polyphase reshaping instead of a large
        identity transposed-conv kernel.

        More information about `perfect reconstruction`:
        1. https://ww2.mathworks.cn/help/signal/ref/stft.html
        2. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html

        Args:
            win_len (int): Number of points in one frame.  Defaults to 1024.
            win_hop (int): Number of framing stride. Defaults to 512.
            fft_len (int): Number of DFT points. Defaults to 1024.
            enframe_mode (str, optional): `break` and `continue`.
                Only `continue` is supported in this ONNX path.
            window (tensor, optional): The window tensor. Defaults to hann window.
            win_sqrt (bool, optional): Use square-root analysis/synthesis window pair.
        """
        super(iSTFT, self).__init__()
        assert enframe_mode in ['break', 'continue']
        assert fft_len >= win_len
        assert enframe_mode == 'continue', 'Only continue mode is supported in DFT iSTFT'
        assert fft_len % win_hop == 0, 'fft_len must be divisible by win_hop for polyphase OLA'
        self.win_len = win_len
        self.win_hop = win_hop
        self.fft_len = fft_len
        self.n_phase = fft_len // win_hop
        self.mode = enframe_mode
        self.win_sqrt = win_sqrt
        self.pad_amount = self.fft_len // 2

        if window is None:
            window = torch.hann_window(win_len)
        self.__init_kernel__(window)

    def __init_kernel__(self, window):
        if self.mode == 'continue':
            left_pad = (self.fft_len - self.win_len) // 2
            right_pad = left_pad + (self.fft_len - self.win_len) % 2
            window = F.pad(window, (left_pad, right_pad))
        if self.win_sqrt:
            self.register_buffer('norm_window', window, persistent=False)
            window = torch.sqrt(window)
        else:
            self.register_buffer('norm_window', window ** 2, persistent=False)
        self.register_buffer('synthesis_window', window, persistent=False)

    def _build_full_spectrum(self, real, imag):
        # real/imag: [B, N, T], where N = floor(n_fft / 2) + 1
        # Build conjugate-symmetric full spectrum [B, fft_len, T, 2].
        n_bins = real.size(1)
        if self.fft_len % 2 == 0:
            mirror_src_real = real[:, 1:n_bins - 1, :]
            mirror_src_imag = imag[:, 1:n_bins - 1, :]
        else:
            mirror_src_real = real[:, 1:n_bins, :]
            mirror_src_imag = imag[:, 1:n_bins, :]
        mirror_real = torch.flip(mirror_src_real, dims=[1])
        mirror_imag = -torch.flip(mirror_src_imag, dims=[1])
        real_full = torch.cat([real, mirror_real], dim=1)
        imag_full = torch.cat([imag, mirror_imag], dim=1)
        return torch.stack([real_full, imag_full], dim=-1)

    def _polyphase_overlap_add(self, frames):
        # frames: [B, N, T], N = n_phase * hop
        # OLA via polyphase decomposition to avoid a huge eye-kernel constant.
        batch, _, n_frames = frames.size()
        frames = frames.reshape(batch, self.n_phase, self.win_hop, n_frames)

        merged = None
        for phase in range(self.n_phase):
            part = frames[:, phase, :, :]
            shifted = F.pad(part, (phase, self.n_phase - 1 - phase))
            merged = shifted if merged is None else (merged + shifted)

        merged = merged.transpose(1, 2).contiguous()  # [B, T + n_phase - 1, hop]
        return merged.reshape(batch, 1, -1)

    def forward(self, spec, length):
        """Inverse STFT from one-sided complex spectrum.

        Args:
            spec (tensors): Input tensor with shape
            complex [num_batch, num_frequencies, num_frames]
            or real [num_batch, num_frequencies, num_frames, 2]
            length (int): Expected number of samples in the output audio.

        Returns:
            tensors: Reconstructed waveform of shape [num_batch, num_samples]
        """
        if torch.is_complex(spec):
            real, imag = spec.real, spec.imag
        else:
            assert spec.size(-1) == 2
            real, imag = spec[..., 0], spec[..., 1]

        full_spec = self._build_full_spectrum(real, imag)  # [B, fft_len, T, 2]
        frames = _InverseDFT.apply(full_spec, self.fft_len)[..., 0]  # [B, fft_len, T]
        frames = frames * self.synthesis_window[None, :, None]

        outputs = self._polyphase_overlap_add(frames)
        t = self.norm_window[None, :, None].repeat(1, 1, frames.size(-1))
        t = t.to(frames.device)
        coff = self._polyphase_overlap_add(t)
        rm_start, rm_end = self.pad_amount, self.pad_amount + length
        outputs = outputs[..., rm_start:rm_end]
        coff = coff[..., rm_start:rm_end]
        coff = torch.where(coff > 1e-8, coff, torch.ones_like(coff))
        outputs /= coff
        return outputs.squeeze(dim=1)
