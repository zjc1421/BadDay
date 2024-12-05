# Multimodal Attack Detection for Action Recognition Models


This is the official repository for VLAD (Vision-Language Attack Detection) as proposed in the paper "Multimodal Attack Detection for Action Recognition Models." VLAD is an adversarial attack detection method for video action recognition models.

VLAD is compatible with any action recognition model, and any implementation of CLIP can be used. In this official implementation, we use CSN from the PyTorchVideo library as the target action recognition model, and CLIP from the Huggingface Transformers package.

In this official implementation, we demonstrate the calculation of detection score proposed in VLAD. Detection scores are calculated for both clean videos and attacked videos. We use PGD-v as the adversarial attack. The expected behavior is that clean videos will have low scores, while attacked videos will have high scores. More details can be found in our [paper](https://arxiv.org/pdf/2404.10790).

## Installation

To install the required packages:

```
pip install -r requirements.txt
```

## Data

We provide 8 example videos from Kinetics dataset. Namely: vid1, vid2, vid3, vid4 are from class 7 (arranging flowers); vid5, vid6, vid7, vid8 are from class 8 (assembling computer). To extract the videos follow these steps:

```
cd videos
tar -xf part1.tar.xz
tar -xf part2.tar.xz
```


## Usage

```vlad.py``` function that calculates the detection score for a given video.

```attacks.py``` has the implementation of PGD-v attack.

```run_vlad.py``` reports the detection scores for a given video's clean and attacked versions.

To calculate and report VLAD scores on all provided videos for the clean and attacked versions:

```
python run_vlad_all.py
```
Additionally, jupyter notebook version is provided in ```run_vlad_all.ipynb```.


## Cite

```
@article{mumcu2024multimodal,
  title={Multimodal Attack Detection for Action Recognition Models},
  author={Mumcu, Furkan and Yilmaz, Yasin},
  journal={arXiv preprint arXiv:2404.10790},
  year={2024}
}
```
