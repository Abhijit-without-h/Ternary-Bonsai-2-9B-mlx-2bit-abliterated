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

# Ternary Bonsai 2 9B, MLX 2-bit, refusal steering

MLX pack of [prism-ml/Ternary-Bonsai-2-9B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-mlx-2bit), with refusal directions in `refusal_dir.npz`. The 2-bit weights are the upstream files. Steering runs at inference: four language-model blocks subtract the refusal direction from the residual stream.

```
x' = x - alpha * (x · s) s
```

Default is `alpha = 1` on layers 11, 19, 23, and 27. On a 12-prompt refusal probe this went from 12/12 refusals to 0/12. Math, code, and short prose prompts still completed. The model still describes why a request is dangerous.

## Contents

| File | What it is |
| --- | --- |
| `model.safetensors` | Packed 2-bit weights, about 3.2 GB |
| `mtp/model.safetensors` | Multi-token prediction head from the same pack |
| `refusal_dir.npz` | Unit directions `dir_11`, `dir_19`, `dir_23`, `dir_27` (float64, dim 4096), plus the harmful and harmless means |
| `config.json`, tokenizer, `hadamard.json`, `runtime/` | Files required to load the pack in mlx-vlm |

## Use

```bash
hf download Abhijit-2k5/Ternary-Bonsai-2-9B-mlx-2bit-abliterated --local-dir bonsai2-9b-mlx
pip install "mlx-vlm>=0.7.8" numpy
```

`lm.layers[i]` has to return the same structure it returned before the wrap: a tensor, or a tuple whose first element is the hidden state.

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

data = np.load("bonsai2-9b-mlx/refusal_dir.npz")
model, processor = load("bonsai2-9b-mlx")
lm = model.language_model
for i in (11, 19, 23, 27):
    lm.layers[i] = Steer(lm.layers[i], data[f"dir_{i}"], alpha=1.0)
```

Pass the Bonsai sampler yourself: `temperature=1.0`, `top_p=0.95`, `top_k=20` with thinking on. `mlx_vlm.generate` is greedy if those are omitted.

## Measurement

Directions come from teacher-forced forward passes on 100 prompts from `mlabonne/harmful_behaviors` and 100 from `tatsu-lab/alpaca`. The last 8 residual positions at layers 11, 19, 23, and 27 were accumulated in float64. The direction is `normalize(mu_harmful - mu_harmless)` after two Gram-Schmidt steps against `mu_harmless`.

The residual stream is in the Hadamard-rotated basis of this pack. A normalized Hadamard is orthonormal, so the projection matches the unrotated basis.

Cosine between the two means: layer 11 = 0.880, 19 = 0.739, 23 = 0.734, 27 = 0.748.

## License

Apache 2.0, the same terms as `prism-ml/Ternary-Bonsai-2-9B-mlx-2bit`, which is a pack of Qwen/Qwen3.5-9B. `LICENSE` and `NOTICE.txt` are the upstream notices.
