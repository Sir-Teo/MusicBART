import json

def crop_midi(midi, max_length):
    if len(midi) > max_length:
        return midi[:max_length]
    return midi

def main():
    input_filename = "data/prompt_midi.json"
    output_filename = "data/prompt_midi_cropped.json"
    max_midi_length = 1024  # Specify the maximum length for MIDI strings

    with open(input_filename, "r") as file:
        data = json.load(file)

    num_entries = len(data)
    max_prompt_length = 0
    max_midi_length_found = 0
    total_prompt_length = 0
    total_midi_length = 0

    for entry in data:
        prompt_length = len(entry["prompt"])
        midi_length = len(entry["midi"])

        max_prompt_length = max(max_prompt_length, prompt_length)
        max_midi_length_found = max(max_midi_length_found, midi_length)

        total_prompt_length += prompt_length
        total_midi_length += midi_length

        entry["midi"] = crop_midi(entry["midi"], max_midi_length)

    avg_prompt_length = total_prompt_length / num_entries
    avg_midi_length = total_midi_length / num_entries

    print("Statistics:")
    print(f"Number of entries: {num_entries}")
    print(f"Maximum prompt length: {max_prompt_length}")
    print(f"Maximum MIDI length: {max_midi_length_found}")
    print(f"Average prompt length: {avg_prompt_length:.2f}")
    print(f"Average MIDI length: {avg_midi_length:.2f}")

    with open(output_filename, "w") as file:
        json.dump(data, file, indent=2)

    print(f"\nMIDI strings exceeding {max_midi_length} characters have been cropped.")
    print(f"Modified JSON data has been saved to {output_filename}.")

if __name__ == "__main__":
    main()