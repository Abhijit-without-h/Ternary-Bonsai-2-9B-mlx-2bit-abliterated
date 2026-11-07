---
license: apache-2.0
library_name: mlx
pipeline_tag: image-text-to-text
base_model:
- prism-ml/Ternary-Bonsai-2-9B-mlx-2bit
tags:
- mlx
- ternary
- bonsai
- refusal-direction
- steering
---

# Ternary-Bonsai-2-9B-mlx-2bit-abliterated

Residual-stream steering directions for [`prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit). The packed 2-bit weights are unchanged. At inference, four language-model blocks subtract the refusal direction from the residual stream:

```
x' = x - alpha * (x · s) s
```

Default: `alpha = 1` on layers 11, 19, 23, and 27. On a 12-prompt refusal probe set this dropped refusals from 12/12 to 0/12. Math, code, and short prose prompts still completed. The model still describes why a request is dangerous.

## Files

| File | Contents |
| --- | --- |
| `refusal_dir.npz` | Unit directions `dir_{11,19,23,27}` (float64, dim 4096), plus the harmful and harmless means they were built from |

Download the base pack separately (~3.46 GB). Do not re-upload those weights.

```bash
hf download prism-ml/Ternary-Bonsai-2-9B-mlx-2bit --local-dir bonsai2-9b-mlx
hf download Abhijit-2k5/Ternary-Bonsai-2-9B-mlx-2bit-abliterated refusal_dir.npz
pip install "mlx-vlm>=0.7.8" numpy
```

Wrap each chosen block. `lm.layers[i]` must return the same structure it returned before the wrap (tensor, or a tuple whose first element is the hidden state).

```python
import numpy as np
import mlx.core as mx
from mlx_vlm import load

class Steer:
    def __init__(self, inner, vec, alpha):
        self.inner = inner
        self.v = mx.array(vec.astype(np.float32))
        self.alpha = alpha

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        proj = (h @ self.v).astype(mx.float32)
        h = h - self.alpha * proj[..., None] * self.v
        return (h,) + out[1:] if isinstance(out, tuple) else h

data = np.load("refusal_dir.npz")
model, processor = load("bonsai2-9b-mlx")
lm = model.language_model
for i in (11, 19, 23, 27):
    lm.layers[i] = Steer(lm.layers[i], data[f"dir_{i}"], alpha=1.0)
```

Pass the Bonsai sampler explicitly (`temperature=1.0`, `top_p=0.95`, `top_k=20` with thinking on). `mlx_vlm.generate` is greedy if you omit them.

## How the directions were measured

Teacher-forced forward passes over 100 `mlabonne/harmful_behaviors` prompts and 100 `tatsu-lab/alpaca` prompts. Last 8 residual positions at layers 11, 19, 23, and 27, accumulated in float64. Direction is `normalize(mu_harmful - mu_harmless)` after two Gram-Schmidt steps against `mu_harmless`.

The residual stream is in the Hadamard-rotated basis used by this pack. A normalized Hadamard is orthonormal, so the projection is the same as in the unrotated basis.

Cosine between the two means: layer 11 = 0.880, 19 = 0.739, 23 = 0.734, 27 = 0.748.

## Limits

- Apple Silicon, MLX, mlx-vlm 0.7.8 or newer (`model_type: prism_hadamard_qwen35`).
- Vision tower and MTP head are not steered.
- Thinking mode was not measured under steering.
- Re-measure if you change packs. These vectors match this 9B MLX 2-bit release only.
- Context on an 8 GB machine is memory-bound, not the nominal 262K.

## License

Apache 2.0, following the base model.
