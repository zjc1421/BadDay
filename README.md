# BadDay: Black-box Adversarial Attack and Detection for Video Action Recognition

This repository contains the final project for **CSE 493G1: Deep Learning** at the University of Washington, completed by **Lindsey Wei** and **Jim Zhou**. The project focuses on evaluating the state-of-the-art detection technique, **VLAD**, for video action recognition under black-box adversarial attacks.

---

## Overview

Adversarial attacks are a growing concern in machine learning, especially in complex tasks like video action recognition. This project implements and evaluates black-box adversarial attacks and detection methods using VLAD for state-of-the-art action recognition models.

---

## Data

Due to computational constraints, we randomly sampled 100 images from the test sets of the **Kinetics400** and **UCF101** datasets. Pre-processed data (converted to `.npy` format) is provided for convenience.

- **Kinetics400 subset:**
  - Original: [Google Drive Link](https://drive.google.com/drive/folders/1mhztjbBoFoWVPHS2VFMWF43VlY0JtyBL?usp=sharing)
  - Processed (.npy): [Google Drive Link](https://drive.google.com/drive/folders/1JrwHCu3I90KCJcuqefC_EBQBDNscaEil?usp=sharing)

- **UCF101 subset:**
  - Original: [Google Drive Link](https://drive.google.com/drive/folders/1UE8t98EtopowsHf0XYSxLCAn1vcYktDd?usp=sharing)
  - Processed (.npy): [Google Drive Link](https://drive.google.com/drive/folders/1Ciopnh9fkyqxKmu5OYPJxL6ivRojmsl9?usp=sharing)

---

## Getting Started

To work with the VLAD detection technique, use the provided `run_vlad.ipynb` notebook. This notebook contains all necessary steps to start evaluating VLAD with the supplied datasets.

---

## References

We appreciate and acknowledge the contributions of the following resources:

- **Code Base:**
  - [Efficient Decision-based Black-box Adversarial Attacks](https://github.com/sgmath12/efficient-decision-based-black-box-adversarial-attacks-on-face-recognition)
  - [VLAD Reference Code](https://drive.google.com/drive/folders/1Ciopnh9fkyqxKmu5OYPJxL6ivRojmsl9?usp=sharing)

- **Papers:**
  - [Efficient Decision-based Black-box Adversarial Attacks on Face Recognition](https://arxiv.org/abs/2404.10790)
  - [VLAD for Video Action Recognition](https://arxiv.org/abs/1904.04433)

---

## Authors

- **Lindsey Wei**
- **Jim Zhou**

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
