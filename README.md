# GraphVSelect

**GraphVSelect: Verified Graph-guided Selection for Zero-shot Referring Expression Comprehension**

[Jeonghyun Choi](https://github.com/Jeong-hyunChoi)<sup>1\*</sup>,
[Yao Wei](https://weiyao1996.github.io/)<sup>2\*</sup>,
Hyungcheol Noh<sup>1</sup>,
[Andrea Cavallaro](https://people.epfl.ch/andrea.cavallaro)<sup>3</sup>,
[Changjae Oh](https://cj-oh.github.io/)<sup>2&dagger;</sup>

<sup>1</sup>Innodep Inc. &nbsp; <sup>2</sup>Queen Mary University of London &nbsp; <sup>3</sup>EPFL
&nbsp;&nbsp; <sup>\*</sup>Equal contribution &nbsp; <sup>&dagger;</sup>Corresponding author

[**Project Page**](https://jeong-hyunchoi.github.io/GraphVSelect/) | Paper (coming soon) | arXiv (coming soon)

<p align="center">
  <img src="https://jeong-hyunchoi.github.io/GraphVSelect/static/images/figure2.png" width="100%" alt="GraphVSelect overview">
</p>

GraphVSelect is a **training-free** framework for zero-shot REC. It (i) builds a
query-conditioned scene graph under object- and caption-level verification, and (ii) jointly
ranks all candidates under the verified graph, re-answering only ambiguous cases selected by a
calibrated top-2 likelihood margin. Every component is a frozen off-the-shelf model.

## Installation

```bash
git clone https://github.com/Jeong-hyunChoi/GraphVSelect.git
cd GraphVSelect
docker build -t graphvselect:latest -f docker/Dockerfile .
docker run --gpus all -it -v "$PWD":/workspace/GraphVSelect -e PYTHONPATH=/workspace/GraphVSelect graphvselect:latest bash
```

A `requirements.txt` is provided for non-Docker installs (Python 3.11, PyTorch 2.8, vLLM 0.11).
Model weights (GroundingDINO, Qwen3-VL-8B, LLaVA-OneVision-7B) are downloaded from HuggingFace
on first use.

## Quick Start

The repository ships a 100-record Ref-L4 sample with images, so the full pipeline runs without
any dataset download:

```bash
bash run_pipeline.sh     # verified scene graph -> list-wise selection -> Acc@0.5/0.75/0.9, mAcc
bash run_boxwise.sh      # box-wise verification baseline
bash run_ablation.sh     # verification ablation (none / object / caption / both)
```

To reproduce the paper's numbers on the full Ref-L4 or RefCOCO/+/g benchmarks, see
[docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## Results

| Benchmark | Metric | GraphVSelect | Best zero-shot baseline |
| --- | --- | ---: | ---: |
| RefCOCO/+/g (8 splits, avg) | Acc@0.5 | **74.89** | 65.10 |
| Ref-L4 (val+test) | mAcc | **60.30** | 51.71 |

Qwen3-VL-8B-Instruct backbone, single deterministic run. Full tables are on the
[project page](https://jeong-hyunchoi.github.io/GraphVSelect/).

## Citation

```bibtex
@inproceedings{choi2027graphvselect,
  title     = {GraphVSelect: Verified Graph-guided Selection for Zero-shot Referring Expression Comprehension},
  author    = {Choi, Jeonghyun and Wei, Yao and Noh, Hyungcheol and Cavallaro, Andrea and Oh, Changjae},
  year      = {2027}
}
```

## Acknowledgements

This work builds on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO),
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT),
[Molmo](https://github.com/allenai/molmo) and [vLLM](https://github.com/vllm-project/vllm).

## License

Released under the [Apache License 2.0](LICENSE).
