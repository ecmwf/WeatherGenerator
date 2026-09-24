import math

import pytest
import torch

from weathergen.model.diffusion import NoiseEmbedder

# c_noise = ln(sigma) / 4 over the log-uniform training range sigma in [0.4, 50]
C_NOISE = torch.linspace(math.log(0.4), math.log(50.0), 64) / 4


def _legacy_reference(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.bfloat16) / half
    )
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def test_legacy_is_default_and_unchanged():
    emb = NoiseEmbedder(embedding_dim=32, frequency_embedding_dim=16)
    assert emb.embedding_type == "dit_legacy"
    assert emb.mlp[0].in_features == 16
    torch.testing.assert_close(emb.timestep_embedding(C_NOISE), _legacy_reference(C_NOISE, 16))


@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("dim", [16, 17])
def test_log_fourier_shapes(include_raw, dim):
    emb = NoiseEmbedder(32, dim, embedding_type="log_fourier", include_raw=include_raw)
    feats = emb.log_fourier_embedding(C_NOISE)
    assert feats.shape == (len(C_NOISE), dim + int(include_raw))
    assert emb.mlp[0].in_features == dim + int(include_raw)
    assert emb(C_NOISE).shape == (len(C_NOISE), 32)
    assert emb(C_NOISE[:1].reshape(())).shape == (1, 32)
    if include_raw:
        torch.testing.assert_close(feats[:, -1], C_NOISE)


def test_log_fourier_frequencies_span_config():
    emb = NoiseEmbedder(32, 256, embedding_type="log_fourier", f_min=0.5, f_max=25.0)
    # at t=pi/(2f) the sin channel of frequency f is 1 -> check endpoints via small t
    t = torch.tensor([1e-3])
    sin = emb.log_fourier_embedding(t)[0, 128:256]
    torch.testing.assert_close(sin[0], torch.tensor(math.sin(0.5e-3)))
    torch.testing.assert_close(sin[-1], torch.tensor(math.sin(25e-3)))


def test_log_fourier_separates_noise_levels_better_than_legacy():
    legacy = NoiseEmbedder(32, 256).timestep_embedding(C_NOISE)
    new = NoiseEmbedder(32, 256, embedding_type="log_fourier").log_fourier_embedding(C_NOISE)
    cos = torch.nn.functional.cosine_similarity
    assert cos(legacy[0], legacy[-1], dim=0) > 0.95
    assert cos(new[0], new[-1], dim=0) < 0.5
