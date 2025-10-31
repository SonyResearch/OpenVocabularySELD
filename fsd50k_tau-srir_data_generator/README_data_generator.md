# FSD50K_TAU-SRIR Data Generator
This `fsd50k_tau-srir_data_generator/` directory provides a data generator for the FSD50K_TAU-SRIR synthetic audio dataset.
The source code is based on https://github.com/danielkrause/DCASE2022-data-generator.
We made a few minor edits to the code for the FSD50K_TAU-SRIR dataset.

## Getting started
### Prerequisites
**NOTE: This data generator uses a different python environment than the main scripts.**

The provided system has been tested on python 3.8.17.

You can install the requirements by running the following lines.
```bash
cd fsd50k_tau-srir_data_generator/
python3 -m venv ~/venv/ovseld_data_generator
source ~/venv/ovseld_data_generator/bin/activate
pip3 install -r requirements_data_generator.txt
```

### Data preparation for train data synthesis
We provide a pre-made parameter file for a quick start.
The OBJ files can be downloaded from the link [OpenVocabularySELD Supplemental Materials](https://zenodo.org/records/17481905).

The OBJ file `db_config_fsd50k_tau-srir.obj` should be put under `fsd50k_tau-srir_data_generator/`.

The FSD50K and TAU-SRIR datasets can be downloaded from their links [FSD50K](https://zenodo.org/records/4060432) and [TAU Spatial Room Impulse Response Database (TAU-SRIR DB)](https://zenodo.org/records/6408611).

Please follow the directory structure below.
```
../
├── data_fsd50k_tau-srir
│   ├── fsd50k
│   │   ├── FSD50K.dev_audio
│   │   ├── FSD50K.eval_audio
│   │   ├── FSD50K.ground_truth
│   │   ├── fsd50k_all (made with code below)
│   │   └── fsd50k_all_class.txt (already set or made with code below)
│   ├── list_dataset
│   ├── tau-srir
│   │   ├── TAU-SNoise_DB
│   │   └── TAU-SRIR_DB
│   ├── train
│   └── val
...
```

Before data synthesis, you need to prepare the `fsd50k_all/` directory from the original `FSD50K.*/` directories.
`fsd50k_all_class.txt` can also be made. (Or you can use the preset one.)
```bash
source ~/venv/ovseld/bin/activate
python setup_fsd50k_all/make_fsd50k_all_dir.py
python setup_fsd50k_all/make_fsd50k_all_class_txt.py
```

### Train data synthesis
After the preparation, you can run the script below.

```bash
source ~/venv/ovseld_data_generator/bin/activate
python example_script_FSD50K_all.py
```

You can change the `num_files` variable (default: 2250) in `generation_parameters.py`.
When generating large training data, e.g., 90,000 minutes, we recommend setting `num_files = 9000` and running the code 10 times.
We sometimes found that too large `num_files` variable caused an unexpected error during data synthesis.

If you change `db_config.py` regarding the fold, RIR data, or audio samples, please create the OBJ file yourself (ref: comments in `example_script_FSD50K_all.py`).
Please do not use the pre-made `db_config_fsd50k_tau-srir.obj` at that time since the db_config parameters are no longer the same.

### Val data synthesis
Before val data synthesis, you need to prepare partial category lists (e.g., [10, 55, ..., 149]).
```bash
source ~/venv/ovseld/bin/activate
python setup_val/make_part_category_txt.py
```

Then, you can run the lines below.
```bash
source ~/venv/ovseld_data_generator/bin/activate
for i in `ls ../data_fsd50k_tau-srir/val/list_category/*.txt`; do python example_script_FSD50K_all.py ${i}; done
```
```bash
cd ../  # i.e., OpenVocabularySELD/
for count in `seq -w 0 15`; do ls ./data_fsd50k_tau-srir/val/part_category_${count}/foa/dev-val/*.wav > ./data_fsd50k_tau-srir/val/list_dataset/fsd50k_tau-srir_foa_val_part_category_${count}.txt; done
```
