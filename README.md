# MusicBART

MusicBART is a project that aims to fine-tune the BART (Bidirectional and Auto-Regressive Transformers) model for generating MIDI files in ABC notation. The goal is to train the model on a dataset of prompt-midi pairs and enable it to generate new MIDI files based on given prompts.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To install the necessary dependencies for MusicBART, run the following command:

```
pip install -r requirements.txt
```

Make sure you have Python 3.x installed on your system.

## Usage

1. Prepare your dataset of prompt-midi pairs and place it in the designated directory.

2. Modify the `dataset_path` variable in `main.py` to point to your dataset directory.

3. Run the `main.py` script to preprocess the data, train the model, and generate MIDI files:

```
python main.py
```

4. The generated MIDI files will be saved in the output directory, and the evaluation metrics will be displayed in the console.

## Dataset

The dataset used for training MusicBART should consist of prompt-midi pairs. Each pair should include a text prompt and its corresponding MIDI file in ABC notation. The dataset should be organized in a specific format, with each prompt-midi pair stored in a separate directory.

## Model Architecture

MusicBART is based on the BART architecture, which is a transformer-based model that combines bidirectional and auto-regressive transformers. The model is fine-tuned on the prompt-midi dataset to learn the mapping between text prompts and MIDI files in ABC notation.

## Evaluation

The generated MIDI files are evaluated using several metrics, including:

- Note Density: Measures the average number of notes per unit of time in the generated MIDI file.
- Pitch Range: Calculates the range of pitches used in the generated MIDI file.
- Rhythmic Complexity: Assesses the complexity of the rhythmic patterns in the generated MIDI file.

The evaluation metrics provide insights into the quality and characteristics of the generated MIDI files.

## Results

The performance of MusicBART is evaluated on a validation set using the defined evaluation metrics. The average values of the metrics are reported to assess the overall quality of the generated MIDI files.
