import os
import numpy as np
import pretty_midi

def load_dataset(dataset_path):
    dataset = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.txt'):
                prompt_file = os.path.join(root, file)
                midi_file = os.path.join(root, file[:-4] + '.mid')
                
                with open(prompt_file, 'r') as f:
                    prompt = f.read().strip()
                
                midi_data = pretty_midi.PrettyMIDI(midi_file)
                
                dataset.append((prompt, midi_data))
    
    return dataset

def tokenize_midi(midi_data):
    # Convert MIDI data to abc notation
    abc_notation = midi_to_abc(midi_data)
    
    # Tokenize the abc notation
    tokens = tokenize_abc(abc_notation)
    
    return tokens

def midi_to_abc(midi_data):
    # Convert MIDI data to abc notation
    # You can use libraries like music21 or implement your own conversion logic
    # This is a placeholder implementation
    abc_notation = ''
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitch = pretty_midi.note_number_to_name(note.pitch)
            duration = note.end - note.start
            abc_notation += f'{pitch}{duration} '
    
    return abc_notation.strip()

def tokenize_abc(abc_notation):
    # Tokenize the abc notation
    # You can use regular expressions or implement your own tokenization logic
    # This is a placeholder implementation
    tokens = abc_notation.split()
    
    return tokens

def preprocess_data(dataset):
    preprocessed_data = []
    for prompt, midi_data in dataset:
        tokens = tokenize_midi(midi_data)
        preprocessed_data.append((prompt, tokens))
    
    return preprocessed_data

def main():
    dataset_path = 'path/to/your/dataset'
    dataset = load_dataset(dataset_path)
    preprocessed_data = preprocess_data(dataset)
    
    # Save the preprocessed data
    np.save('preprocessed_data.npy', preprocessed_data)

if __name__ == '__main__':
    main()