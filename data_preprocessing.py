import os
import numpy as np
import pretty_midi
import json

import json

def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset_str = f.read()
    dataset = json.loads(dataset_str)
    print(f"Type of dataset: {type(dataset)}")  # Debug print statement
    return dataset

def preprocess_data(dataset):
    preprocessed_data = []
    for sample in dataset:
        prompt = sample['prompt']
        midi_text = sample['abc_content']
        tokens = tokenize_midi(midi_text)
        preprocessed_data.append((prompt, tokens))
    return preprocessed_data



def tokenize_midi(midi_data):
    
    # Tokenize the abc notation
    tokens = tokenize_abc(midi_data)
    
    return tokens


def tokenize_abc(abc_notation):
    # Tokenize the abc notation
    # You can use regular expressions or implement your own tokenization logic
    # This is a placeholder implementation
    tokens = abc_notation.split()
    
    return tokens


def main():
    dataset_path = 'data/sample.json'
    dataset = load_dataset(dataset_path)
    preprocessed_data = preprocess_data(dataset)
    
    # Save the preprocessed data
    np.save('preprocessed_data.npy', preprocessed_data)

if __name__ == '__main__':
    main()