# MusicBART

MusicBART is a project that aims to fine-tune the BART (Bidirectional and Auto-Regressive Transformers) model for generating MIDI files in ABC notation. The goal is to train the model on a dataset of prompt-midi pairs and enable it to generate new MIDI files based on given prompts.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
-
## Installation

To install the necessary dependencies for MusicBART, run the following command:

```
pip install -r requirements.txt
```


## Usage

1. The dataset is in `data/sample.json`

2. Modify the `dataset_path` variable in `main.py` to point to your dataset directory.

3. Run the `main.py` script to preprocess the data, train the model, and generate MIDI files:

```
python main.py
```

4. The generated MIDI files will be saved in the model directory, run

```
python test_model.py
```
to test some sample prompts.

## Dataset

The dataset used for training MusicBART should consist of prompt-midi pairs. Each pair should include a text prompt and its corresponding MIDI file in ABC notation. The dataset should be organized in a specific format, with each prompt-midi pair stored in a separate directory.

## Model Architecture

MusicBART is based on the BART architecture, which is a transformer-based model that combines bidirectional and auto-regressive transformers. The model is fine-tuned on the prompt-midi dataset to learn the mapping between text prompts and MIDI files in ABC notation.

## Evaluation


