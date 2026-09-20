# BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation

Official implementation of:

**BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation**

Paper: https://arxiv.org/abs/2603.00156

## Project Description

BiCLIP is a vision-language framework for robust medical image segmentation. It introduces Asymmetric Bidirectional Fusion (ABF) to enhance image-text interaction and Image Augmentation Consistency (IAC) to improve model robustness under limited annotations and image degradation conditions.

## Overview

This repository provides the official implementation of BiCLIP, including:

* Vision-language segmentation framework
* Bidirectional multimodal fusion components
* Image augmentation consistency learning
* JSON-based medical reports used as textual inputs during training
* Configuration files and utility functions

## Repository Structure

```text
BiCLIP/
│
├── JSON_Files/       # Medical report JSON files used for vision-language training
├── config/           # Configuration file for training setting
├── engine/           # Training and evaluation engine components
├── utils/            # Utility functions and helper modules
├── train.py          # Training script
├── evaluate.py       # Evaluation script
├── requirements.txt  # Required Python packages
└── README.md         # Documentation
```

## Installation

Clone the repository:

```bash
git clone https://github.com/SeivanTalaie/BiCLIP.git
cd BiCLIP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train BiCLIP, use the following command:

```bash
python train.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name covid

# Only change the dataset name (e.g., covid, kvasir) according to the target dataset.
# Keep the rest of the command unchanged.

````
## Evaluation

To evaluate a trained BiCLIP model, use the following command:

```bash
python evaluate.py --model_name utils.githubUNET.github_UNET_aug_text_gene.UNet --dataset_name mosmed --ckpt "/home/asosoft/Seivan/MosMedData+_Model/MosMed_Best.ckpt"

# Change the dataset name and checkpoint path according to the target dataset and trained model.
# Keep the remaining arguments unchanged.
````

## Dataset Information

The experiments in this work are conducted on four medical image segmentation benchmarks:

- **QaTa-COV19**
- **MosMedData+**
- **Kvasir-SEG**
- **CVC-ClinicDB**

For **QaTa-COV19** and **MosMedData+**, the datasets are obtained from their original sources, and the official data split protocols provided by **<a href="https://github.com/HUANGLIZI/LViT">LViT</a>** are adopted for training and evaluation.

For **Kvasir-SEG** and **CVC-ClinicDB**, we use the datasets and follow the corresponding data split protocols provided by **<a href="https://github.com/naamiinepal/medvlsm">MedVLSM</a>**.

The original medical image datasets are not included in this repository. Please refer to the above resources and the paper for detailed information regarding dataset preparation.

## Citation

If you find this work useful for your research, please consider citing:

```bibtex
@misc{talaei2026biclipbidirectionalconsistentlanguageimage,
      title={BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation}, 
      author={Saivan Talaei and Fatemeh Daneshfar and Abdulhady Abas Abdullah and Mourad Oussalah},
      year={2026},
      eprint={2603.00156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.00156}, 
}
````

## License

This repository is released for academic and research purposes.
