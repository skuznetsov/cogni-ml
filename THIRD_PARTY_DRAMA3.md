# DRAMA-3 third-party notices

The native Crystal DRAMA-3 backbone, audio tower, and codec implement behavior
from the pinned upstream MLX-audio sources listed below. The tiny fixture uses
deterministic synthetic weights and activations run through pinned upstream
implementations; it contains no released checkpoint weights or copied
upstream source files.

| Source | Revision | Use and notice |
| --- | --- | --- |
| [MLX-audio](https://github.com/cguldogan/mlx-audio) | `e7488935ec2ebfafd38ef2b58e290d79f3d05dc7` | Backbone reference: [`qwen3_5.py`](https://github.com/cguldogan/mlx-audio/blob/e7488935ec2ebfafd38ef2b58e290d79f3d05dc7/mlx_audio/lm/models/qwen3_5.py); tower and codec references are in `fish_omni/tower.py` and `fish_s1_dac.py`. Repository license is MIT, Copyright (c) 2024 Prince Canuma. |
| [Fish Speech](https://github.com/fishaudio/fish-speech) | `541566932abbaebcb0a40e562a1f846738383f38` | The synthetic reference exporter runs the official audio tower and layers. See the upstream [Fish Audio Community License, Version 1.0](https://github.com/fishaudio/fish-speech/blob/541566932abbaebcb0a40e562a1f846738383f38/LICENSE). Required notice: “These Fish Audio Materials are licensed under the Fish Audio Community License, Version 1.0. Copyright © 39 AI, INC. All Rights Reserved.” |

The pinned MLX-audio `qwen3_5.py` source identifies the Qwen3.5 text model as
vendored from mlx-lm and carries the notice **Copyright © 2026 Apple Inc.**
This repository's native backbone follows that model behavior; it does not
copy the upstream Python file. See the [upstream source file](https://github.com/cguldogan/mlx-audio/blob/e7488935ec2ebfafd38ef2b58e290d79f3d05dc7/mlx_audio/lm/models/qwen3_5.py)
and [Apple mlx-lm source](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py).

The optional real-checkpoint replays read local files from
[`cguldogan/drama-3-mlx-q4`](https://huggingface.co/cguldogan/drama-3-mlx-q4)
at revision `5331ab509359c0e4a7e353a19e255ddf3e8152a8`. Those checkpoint weights
are separately licensed under the Fish Audio Community License, Version 1.0,
as stated in the model repository's pinned [`LICENSE`](https://huggingface.co/cguldogan/drama-3-mlx-q4/blob/5331ab509359c0e4a7e353a19e255ddf3e8152a8/LICENSE). The checkpoint is not
part of this repository. Actual-checkpoint traces and replay manifests are
local validation artifacts; configuration, index, selected source, and tensor
row hashes record provenance without hashing multi-gigabyte shards by default.

## MLX-audio MIT license notice

The following is the complete MIT notice from the pinned MLX-audio checkout:

```text
MIT License

Copyright (c) 2024 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Apple mlx-lm MIT license notice

The upstream mlx-lm repository's MIT license notice is included because the
pinned Qwen3.5 reference file carries Apple's copyright notice:

```text
MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
