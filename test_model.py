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
    prompt_tensor = torch.tensor(prompt_tokenizer.tokenize(prompt)).to(device)
    prompt_tensor = prompt_tensor.unsqueeze(0)
    attention_mask = torch.ones(prompt_tensor.shape, dtype=torch.long, device=device)
    
    generated_sequence = model.generate(
        prompt_tensor,
        attention_mask,
        num_beams=3,
        max_length=256,  # Increase the maximum length of the generated sequence
    )
    
    return generated_sequence

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "./model/trained_model.pth"
    model = load_model(model_path, device)

    prompt_tokenizer = PromptTokenizer()
    midi_tokenizer = MidiTokenizer()

    prompts = [
        "Generate a happy and upbeat melody",
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