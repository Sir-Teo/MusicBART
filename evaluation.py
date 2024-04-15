# evaluation.py

import torch
import mido
from io import BytesIO

def sequence_to_midi(sequence, tempo=120, ticks_per_beat=480):
    # Create a new MIDI file
    midi_file = mido.MidiFile()
    track = mido.MidiTrack()
    midi_file.tracks.append(track)

    # Set the tempo
    tempo_bpm = mido.bpm2tempo(tempo)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_bpm))

    # Parse the generated sequence and create MIDI events
    current_time = 0
    for event in sequence.split():
        if event[0] == 'N':
            # Note event
            pitch = int(event[1:])
            track.append(mido.Message('note_on', note=pitch, velocity=64, time=current_time))
            current_time = 0
        elif event[0] == 'R':
            # Rest event
            rest_duration = int(event[1:])
            current_time += rest_duration
        else:
            # Unknown event, skip it
            continue

    # End of track
    track.append(mido.MetaMessage('end_of_track', time=current_time))

    # Save the MIDI file to a BytesIO object
    midi_bytes = BytesIO()
    midi_file.save(file=midi_bytes)
    midi_data = midi_bytes.getvalue()

    return midi_data


def evaluate_note_density(generated_midi):
    # Count the number of notes in the generated MIDI string
    note_count = generated_midi.count('|')
    
    # Calculate the note density based on the number of notes and the length of the MIDI string
    note_density = note_count / len(generated_midi)
    
    return note_density

def evaluate_pitch_range(generated_midi):
    # Extract pitch information from the generated MIDI string
    pitches = [int(note[1:]) for note in generated_midi.split('|') if note.startswith('N')]
    
    # Calculate the pitch range
    min_pitch = min(pitches) if pitches else 0
    max_pitch = max(pitches) if pitches else 0
    pitch_range = max_pitch - min_pitch
    
    return pitch_range

def evaluate_rhythm_complexity(generated_midi):
    # Extract rhythm information from the generated MIDI string
    rhythms = [len(note) for note in generated_midi.split('|') if note.startswith('N')]
    
    # Calculate the average rhythm length
    avg_rhythm_length = sum(rhythms) / len(rhythms) if rhythms else 0
    
    # Calculate the rhythm complexity based on the average rhythm length
    rhythm_complexity = 1 / (avg_rhythm_length + 1)
    
    return rhythm_complexity

def evaluate_structural_coherence(generated_midi):
    # Split the generated MIDI string into sections
    sections = generated_midi.split('|')
    
    # Calculate the average section length
    avg_section_length = sum(len(section) for section in sections) / len(sections)
    
    # Calculate the structural coherence based on the average section length
    structural_coherence = 1 / (avg_section_length + 1)
    
    return structural_coherence




def evaluate_midi(generated_midi):
    try:
        # Evaluate metrics
        note_density = evaluate_note_density(generated_midi)
        pitch_range = evaluate_pitch_range(generated_midi)
        rhythm_complexity = evaluate_rhythm_complexity(generated_midi)
        structural_coherence = evaluate_structural_coherence(generated_midi)
        
        # Print the evaluation results
        print("Note Density:", note_density)
        print("Pitch Range:", pitch_range)
        print("Rhythm Complexity:", rhythm_complexity)
        print("Structural Coherence:", structural_coherence)
    
    except Exception as e:
        print("Error evaluating MIDI:", str(e))

def evaluate_model(model, dataset, device):
    # Evaluate the model on a given dataset
    model.to(device)
    model.eval()
    
    total_note_density = 0
    total_pitch_range = 0
    total_rhythm_complexity = 0
    
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            generated_sequence = model.generate(input_ids, attention_mask, num_beams=4)
            
            note_density = evaluate_note_density(generated_sequence)
            pitch_range = evaluate_pitch_range(generated_sequence)
            rhythm_complexity = evaluate_rhythm_complexity(generated_sequence)
            
            total_note_density += note_density
            total_pitch_range += pitch_range
            total_rhythm_complexity += rhythm_complexity
    
    avg_note_density = total_note_density / len(dataset)
    avg_pitch_range = total_pitch_range / len(dataset)
    avg_rhythmic_complexity = total_rhythm_complexity / len(dataset)
    
    print(f"Average Note Density: {avg_note_density:.2f}")
    print(f"Average Pitch Range: {avg_pitch_range:.2f}")
    print(f"Average Rhythmic Complexity: {avg_rhythmic_complexity:.2f}")