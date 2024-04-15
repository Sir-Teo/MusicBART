# evaluation.py

import pretty_midi

def evaluate_note_density(generated_midi):
    # Calculate the note density of the generated midi file
    midi_data = pretty_midi.PrettyMIDI(generated_midi)
    duration = midi_data.get_end_time()
    num_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
    note_density = num_notes / duration
    return note_density

def evaluate_pitch_range(generated_midi):
    # Calculate the pitch range of the generated midi file
    midi_data = pretty_midi.PrettyMIDI(generated_midi)
    pitches = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)
    pitch_range = max(pitches) - min(pitches)
    return pitch_range

def evaluate_rhythmic_complexity(generated_midi):
    # Calculate the rhythmic complexity of the generated midi file
    midi_data = pretty_midi.PrettyMIDI(generated_midi)
    note_durations = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            duration = note.end - note.start
            note_durations.append(duration)
    unique_durations = len(set(note_durations))
    rhythmic_complexity = unique_durations / len(note_durations)
    return rhythmic_complexity

def evaluate_midi(generated_midi):
    # Evaluate the generated midi file using multiple metrics
    note_density = evaluate_note_density(generated_midi)
    pitch_range = evaluate_pitch_range(generated_midi)
    rhythmic_complexity = evaluate_rhythmic_complexity(generated_midi)
    
    print(f"Note Density: {note_density:.2f}")
    print(f"Pitch Range: {pitch_range}")
    print(f"Rhythmic Complexity: {rhythmic_complexity:.2f}")

def evaluate_model(model, dataset, device):
    # Evaluate the model on a given dataset
    model.to(device)
    model.eval()
    
    total_note_density = 0
    total_pitch_range = 0
    total_rhythmic_complexity = 0
    
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            generated_sequence = model.generate(input_ids, attention_mask)
            generated_midi = sequence_to_midi(generated_sequence)
            
            note_density = evaluate_note_density(generated_midi)
            pitch_range = evaluate_pitch_range(generated_midi)
            rhythmic_complexity = evaluate_rhythmic_complexity(generated_midi)
            
            total_note_density += note_density
            total_pitch_range += pitch_range
            total_rhythmic_complexity += rhythmic_complexity
    
    avg_note_density = total_note_density / len(dataset)
    avg_pitch_range = total_pitch_range / len(dataset)
    avg_rhythmic_complexity = total_rhythmic_complexity / len(dataset)
    
    print(f"Average Note Density: {avg_note_density:.2f}")
    print(f"Average Pitch Range: {avg_pitch_range:.2f}")
    print(f"Average Rhythmic Complexity: {avg_rhythmic_complexity:.2f}")