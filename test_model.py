import torch
from model import MusicBART
from tokenizer import PromptTokenizer, MidiTokenizer
from evaluation import evaluate_midi

def load_model(model_path, device):
    model = MusicBART().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_midi(model, prompt, prompt_tokenizer, device):
    # Tokenize the prompt using the PromptTokenizer and convert to tensor
    prompt_tensor = torch.tensor(prompt_tokenizer.tokenize(prompt), dtype=torch.long).to(device)
    # Add a batch dimension
    prompt_tensor = prompt_tensor.unsqueeze(0)
    
    # Create attention_mask considering the BartTokenizer does not explicitly use a pad token for zeros
    attention_mask = torch.where(prompt_tensor != prompt_tokenizer.pad_token_id, 
                                 torch.ones_like(prompt_tensor), 
                                 torch.zeros_like(prompt_tensor)).to(device)

    # Generate the sequence with the model
    generated_sequence = model.generate(
        input_ids=prompt_tensor,
        attention_mask=attention_mask,
        num_beams=2,
        max_length=256  # Define the maximum length of the generated sequence
    )
    
    # Convert the generated token IDs back to MIDI notation using the MidiTokenizer
    midi_output = midi_tokenizer.detokenize(generated_sequence.squeeze(0).tolist())

    return midi_output



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/trained_model.pth"
    model = load_model(model_path, device)

    prompt_tokenizer = PromptTokenizer()
    midi_tokenizer = MidiTokenizer()

    prompts = [
        "Compose a piece that evokes a sense of deep sorrow and loneliness. Use minor chords and slow tempo to create a mournful and reflective atmosphere. Try incorporating long, expressive phrases with subtle dynamics to enhance the emotional depth of the melody.",
        "Generate an epic orchestral theme",
        "Generate a relaxing jazz tune, very relaxing",
        "Generate a spooky and mysterious melody",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated_sequence = generate_midi(model, prompt, prompt_tokenizer, device)
        print(f"Generated sequence: {generated_sequence}")

if __name__ == "__main__":
    main()