# Open-Vocabulary Sound Event Localization and Detection (SELD)
This codebase is the source code for [Open-Vocabulary Sound Event Localization and Detection with Joint Learning of CLAP Embedding and Activity-Coupled Cartesian DOA Vector](https://ieeexplore.ieee.org/document/11074724).

## Abstract
We aim for an open-vocabulary sound event localization and detection (SELD) system that detects and localizes sound events in any category described by prompt texts. An open-vocabulary SELD system can be applied to various SELD tasks by changing prompt texts. A simple approach is to combine a language-audio model, such as a contrastive language-audio pretraining (CLAP) model, and a direction-of-arrival (DOA) estimation model. However, this combining approach cannot tackle overlapping sound events because the language-audio models output only one audio embedding, even for an audio input containing multiple events. Also, such a naive combination of two models can be sub-optimal regarding joint localization and detection. In this study, we present an embed-ACCDOA model, which jointly learns to output an embedding and the corresponding activity-coupled Cartesian DOA (ACCDOA) vector of each event in a track-wise manner, thereby tackling overlapping events. Each event’s embedding is trained to align with its corresponding audio and text embeddings inferred by a pretrained language-audio model, distilling its knowledge into the embed-ACCDOA model. We evaluate the proposed embed-ACCDOA model using an original synthetic dataset and two external SELD datasets. The knowledge distillation using both audio and text embeddings performs better than distillation using only one of the embeddings. The embed-ACCDOA model outperforms the naive combination. Moreover, it performs better than the official baseline system trained on the fully annotated training data of the target categories.

## Getting started
### Prerequisites
The provided system has been tested on python 3.10.12 and pytorch 2.3.1.

You can install the requirements by running the following lines.
```bash
python3 -m venv ~/venv/ovseld
source ~/venv/ovseld/bin/activate
pip3 install --upgrade pip setuptools
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

### Data preparation for training example
We provide example training data, config, and pre-made parameter files for a quick start.
The data and files can be downloaded from the link [OpenVocabularySELD Supplemental Materials](https://zenodo.org/records/17481905).

You can unzip the train and val ZIP files on `data_fsd50k_tau-srir/`.
```bash
cd data_fsd50k_tau-srir/
unzip train_2250files_example.zip
unzip val_example.zip
```

After unzip, the directory structure can be below.
```
.
├── data_fsd50k_tau-srir
│   ├── fsd50k (already set on this repo)
│   ├── list_dataset (already set on this repo)
│   ├── train
│   └── val
...
```

The PICKLE file `dict_fsdidwav2clap.pickle` should be put under `dict_pickle/630k-audioset-best/`.
Other small PICKLE and NPY files are already set on the GitHub repository.

### Training example
After the preparation, you can run the script below.
This will dump the logs and models in the `data_fsd50k_tau-srir/model_monitor/<DateTime>_<JobID>`.

The training requires 40GB of GPU memory.
We tested the script on one NVIDIA H100, and it takes around half a dozen hours.
```bash
bash script/train_seld_foa_medium_2250.sh
```

The small model size version is the same.
```bash
bash script/train_seld_foa_small_2250.sh
```

You can check the training details using TensorBoard.

If you would like to run training with large data (e.g., 90,000 min), you need to make such data in your environment.
Please see [fsd50k_tau-srir_data_generator/README_data_generator.md](fsd50k_tau-srir_data_generator/README_data_generator.md).

### Data preparation for evaluation example
We provide a pre-trained model parameter file (Model size: medium, Train data size: 90,000 min).
The PTH file can be downloaded from the same link [OpenVocabularySELD Supplemental Materials](https://zenodo.org/records/17481905).

The PTH file `params_swa_20251029062140_154940_0040000.pth` should be put under `data_fsd50k_tau-srir/model_monitor/20251029062140_154940/`.

The TNSSE21 and STARSS23 datasets can be downloaded from their links [TAU-NIGENS Spatial Sound Events 2021](https://zenodo.org/records/5476980) and [STARSS23: Sony-TAu Realistic Spatial Soundscapes 2023](https://zenodo.org/records/7880637).

Please follow the directory structure below.
Note that we use only `dev-test` directories.
```
.
├── data_dcase2021_task3
│   ├── foa_dev
│   ├── list_dataset (already set on this repo)
│   └── metadata_dev
├── data_dcase2023_task3
│   ├── foa_dev
│   ├── list_dataset (already set on this repo)
│   └── metadata_dev
...
```

### Evaluation example
After the preparation, you can run the script below.
```bash
bash script/eval_seld_foa.sh
```

You can get the result below.
* TNSSE21
```
SELD scores
All     ER      F       LE      LR      SELD    ...
All     0.782   0.381   22.32   0.592   0.483   ...
```
* STARSS23
```
SELD scores
All     ER      F       LE      LR      SELD    ...
All     0.725   0.212   23.40   0.332   0.578   ...
```

This is a reproduced result on our model (Model size: medium, Train data size: 90,000 min).
You can get a result similar to our publication.

### Advanced data preparation
For the synthesis of large training data, please visit [fsd50k_tau-srir_data_generator/README_data_generator.md](fsd50k_tau-srir_data_generator/README_data_generator.md).

(Optional) If you would like to make parameter NPY files yourself, you can run the code below.
```bash
python make_data_python/make_class_clap_npy.py
```

(Optional) Making PICKLE files yourself requires the `data_fsd50k_tau-srir/fsd50k/fsd50k_all/` directory.
Please check the data generator's README.
```bash
python make_data_python/make_dict_fsdid2categoryid.py
python make_data_python/make_dict_fsdidwav2clap.py  # takes a few hours
```

## Citation
If you found this repository useful, please consider citing
```bibtex
@article{shimada2025open,
  title={Open-Vocabulary Sound Event Localization and Detection with Joint Learning of CLAP Embedding and Activity-Coupled Cartesian DOA Vector},
  author={Shimada, Kazuki and Uchida, Kengo and Koyama, Yuichiro and Shibuya, Takashi and Takahashi, Shusuke and Mitsufuji, Yuki and Kawahara, Tatsuya},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  year={2025},
  publisher={IEEE}
}
```

## References
This codebase is built on the following papers and the open source repositories.

1. CLAP model: "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation"

2. Data generator: https://github.com/danielkrause/DCASE2022-data-generator

3. FOA rotation for data augmentation: "First Order Ambisonics Domain Spatial Augmentation for DNN-based Direction of Arrival Estimation"

4. Equalized mixture data augmentation: "AENet: Learning Deep Audio Features for Video Analysis"

5. Two-branch network design: "An Improved Event-Independent Network for Polyphonic Sound Event Localization and Detection"
    - a part of `net/` are derived from https://github.com/yinkalario/EIN-SELD

6. Loss function: "Open-Vocabulary Object Detection via Vision and Language Knowledge Distillation"

7. Evaluation metrics: "Baseline Models and Evaluation of Sound Event Localization and Detection with Distance Estimation in DCASE2024 Challenge"
    - `dcase2024_task3_seld_metrics/` are based on https://github.com/partha2409/DCASE2024_seld_baseline
