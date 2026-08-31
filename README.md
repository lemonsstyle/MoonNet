# MoonNet

**Learning to Trust Monocular Candidate Injection for Multi-View Stereo**

MoonNet is a research branch built on [MonoMVSNet](https://github.com/JianfeiJ/MonoMVSNet) (ICCV 2025). It studies one focused question:

> When should a multi-view stereo network trust an aligned monocular depth prior?

MonoMVSNet injects an aligned monocular depth candidate at valid edge pixels during cascade sampling. MoonNet keeps that design and adds a lightweight reliability gate that learns how strongly each candidate should be injected.

## Status

This repository contains a research implementation, not a paper release with verified benchmark claims.

- Implemented: reliability gate, differentiable soft candidate injection, supervised trust loss, training/inference flags and unit tests.
- Statically verified: Python compilation succeeds.
- Not yet verified in this repository: full DTU training, DTU point-cloud scores, BlendedMVS fine-tuning, Tanks and Temples scores, runtime and peak VRAM.

Do not treat the original MonoMVSNet numbers below or any planned MoonNet targets as newly reproduced results.

The complete hypothesis, method boundary, experiment matrix and go/no-go criteria are documented in [docs/research-plan.md](docs/research-plan.md).

## Method

For cascade stages 2–4, let:

- `h` be the original inverse-depth candidate selected for possible replacement;
- `m` be the aligned monocular inverse-depth candidate;
- `r` be the predicted trust score in `[0, 1]`.

MoonNet performs soft candidate injection:

```text
h_new = h + r * (m - h)
```

- `r = 1` reproduces the original MonoMVSNet candidate replacement;
- `r = 0` preserves the original MVS candidate;
- intermediate values reduce the effect of uncertain priors.

The gate uses four observable signals:

1. relative disagreement between aligned monocular and coarse MVS inverse depth;
2. coarse-stage photometric confidence;
3. reference-view edge response;
4. the monocular candidate's normalized offset from the current inverse-depth interval center.

The first cascade stage remains unchanged because no previous metric MVS estimate is available yet.

### Training target

During training, the gate receives a soft oracle target derived from which candidate is closer to ground-truth depth:

```text
target = sigmoid((base_error - mono_error) / (temperature * depth_interval))
```

The total objective is the original MonoMVSNet loss plus a weighted trust loss.

## Repository changes

```text
models/trust.py             reliability gate and trust supervision
models/module.py            soft candidate injection and supervision metadata
models/monomvsnet.py        gate integration and trust loss
train_dtu.py                DTU training flags and logging
train_bld.py                BlendedMVS fine-tuning flags and logging
test_*.py                   checkpoint-compatible trust configuration
tests/test_trust.py         degeneration, shape and gradient checks
docs/research-plan.md       research and experiment plan
```

## Installation

The upstream environment is retained:

```bash
conda create -n moonnet python=3.10.8
conda activate moonnet
pip install -r requirements.txt
```

Install `xformers` separately if you need to reproduce the memory-efficient attention setting used by MonoMVSNet.

## Pretrained components

Download the upstream pretrained weights:

- [Depth Anything V2 ViT-S](https://drive.google.com/file/d/1M1JQWZ9jEa1H0lblt3B6yJU_LyjqF60_/view?usp=drive_link)
- [TEED](https://drive.google.com/file/d/1j8wRHMozt_eJwBjs1JXxTDRboP_lKiSp/view?usp=drive_link)

Place them under:

```text
pre_trained_weights/
├── depth_anything_v2_vits.pth
└── TEED_model.pth
```

For initializing from MonoMVSNet, use the upstream [pretrained models](https://drive.google.com/drive/folders/1xf01LEp0IvEgFBhXxTq0jY4Duo7eiRJG?usp=drive_link). When `--trust_mode learned` is enabled, the loader accepts the missing MoonNet gate parameters and initializes them locally.

## Data preparation

Use the same prepared DTU, BlendedMVS and Tanks and Temples layouts as MonoMVSNet. The upstream project refers to [RRT-MVS](https://github.com/JianfeiJ/RRT-MVS) for preparation details.

Before running the shell scripts, replace their placeholder or author-local dataset paths:

```text
scripts/train_dtu.sh
scripts/train_bld.sh
scripts/test_dtu_dypcd.sh
scripts/test_dtu_pcd.sh
scripts/test_tnt_inter.sh
scripts/test_tnt_adv.sh
```

## Training

### MoonNet on DTU

After editing `DTU_TRAINING` in `scripts/train_dtu.sh`:

```bash
bash scripts/train_dtu.sh _moon \
  --trust_mode learned \
  --trust_loss_weight 0.1 \
  --trust_temperature 1.0
```

### MonoMVSNet-compatible control

Use `always` to disable the learned gate and reproduce the original unconditional candidate replacement path:

```bash
bash scripts/train_dtu.sh _always --trust_mode always
```

For the MVS-only sampling ablation, disable monocular candidate injection while retaining the rest of the architecture:

```bash
bash scripts/train_dtu.sh _mvs_only --disable_mono_sampling --trust_mode always
```

### Fine-tune on BlendedMVS

After editing `BLD_TRAINING` and `BLD_CKPT_FILE` in `scripts/train_bld.sh`:

```bash
bash scripts/train_bld.sh _moon \
  --trust_mode learned \
  --trust_loss_weight 0.1 \
  --trust_temperature 1.0
```

## Evaluation

### DTU dynamic fusion

Edit the paths in `scripts/test_dtu_dypcd.sh`, then run:

```bash
bash scripts/test_dtu_dypcd.sh _moon --trust_mode learned
```

### DTU fixed fusion control

```bash
bash scripts/test_dtu_pcd.sh _moon --trust_mode learned
```

### Tanks and Temples

Edit the dataset and checkpoint paths in the corresponding scripts:

```bash
bash scripts/test_tnt_inter.sh _moon --trust_mode learned
bash scripts/test_tnt_adv.sh _moon --trust_mode learned
```

For official MonoMVSNet checkpoints without gate parameters, instantiate with `--trust_mode always`. MoonNet checkpoints must be evaluated with `--trust_mode learned`.

## Self-checks

Run syntax compilation:

```bash
python -m compileall -q models train_dtu.py train_bld.py test_dtu_dypcd.py \
  test_dtu_pcd.py test_dypcd_tnt_inter.py test_dypcd_tnt_adv.py tests
```

Run the focused tests after PyTorch and the project requirements are installed:

```bash
python -m unittest tests.test_trust -v
```

The tests check:

- initial gate output and gradient flow;
- finite trust-loss gradients;
- `trust = 1` equivalence to original MonoMVSNet replacement;
- `trust = 0` equivalence to unmodified MVS candidates;
- supervision metadata shapes and masks.

## Required experiments before making research claims

At minimum, compare:

| Experiment | Monocular sampling | Trust mode |
|---|---:|---|
| MVS-only control | disabled | n/a |
| MonoMVSNet | enabled | `always` |
| MoonNet | enabled | `learned` |

Report standard geometry metrics together with candidate quantization error, candidate coverage, gate discrimination/calibration, additional parameters, peak VRAM and latency. Controlled prior corruptions must be labeled as synthetic stress tests rather than natural-scene results.

## Upstream reference results

The following are quoted from the upstream README and are **not reproduced by this branch**:

| Dataset / setting | Reported upstream result |
|---|---:|
| DTU Overall, 5 views | 0.278 mm |
| DTU Overall, 9 views | 0.275 mm |
| Tanks and Temples Intermediate | 68.63 F-score |
| Tanks and Temples Advanced | 43.58 F-score |

## Acknowledgements

MoonNet is derived from [JianfeiJ/MonoMVSNet](https://github.com/JianfeiJ/MonoMVSNet), which in turn acknowledges ET-MVSNet, TransMVSNet, MVSFormer++, Depth Anything V2 and TEED. The original license is retained in [LICENSE](LICENSE).

Please cite MonoMVSNet when using this codebase:

```bibtex
@inproceedings{monomvsnet,
  author    = {Jiang, Jianfei and Liu, Qiankun and Yu, Haochen and Liu, Hongyuan and Wang, Liyong and Chen, Jiansheng and Ma, Huimin},
  title     = {MonoMVSNet: Monocular Priors Guided Multi-View Stereo Network},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  month     = {October},
  year      = {2025},
  pages     = {27806--27816}
}
```

No MoonNet paper citation is provided yet because no paper has been published.
