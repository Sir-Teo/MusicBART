# main.py
from data_preprocessing import preprocess_data, load_dataset
from model import MusicBART, train, evaluate, MusicGPT
from evaluation import evaluate_model, evaluate_midi
import torch
import argparse
from test_model import generate_midi

# torchrun --nproc_per_node=2 main_parallel.py --model_name bart --epochs 30 --batch_size 6 --learning_rate 5e-6


def collate_fn(batch, device):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).to(device)
    attention_mask = torch.where(input_ids != 0, torch.ones_like(input_ids), torch.zeros_like(input_ids)).to(device)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the dataset
    dataset_path = "data/sample.json"
    dataset = load_dataset(dataset_path)
    dataset = preprocess_data(dataset)

    # Initialize the MusicBART model
    if args.model_name == "bart":
        model = MusicBART().to(device)
    elif args.model_name == "gpt":
        model = MusicGPT().to(device)

    # Create the tokenizers
    prompt_tokenizer = model.prompt_tokenizer
    midi_tokenizer = model.midi_tokenizer
    
    # Tokenize the dataset
    tokenized_dataset = []
    tokenized_dataset = []
    for prompt, midi_data in dataset:

        input_ids = prompt_tokenizer.tokenize(prompt)
        labels = midi_tokenizer.tokenize(midi_data)
        tokenized_dataset.append({"input_ids": input_ids, "labels": labels})



    # Split the dataset into train and validation sets
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset[:train_size]
    val_dataset = tokenized_dataset[train_size:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda batch: collate_fn(batch, device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, collate_fn=lambda batch: collate_fn(batch, device))
    
    
    # Set the training parameters
    epochs = 10
    batch_size = 8
    learning_rate = 1e-5
    
    # Train the model
    trained_model = train(model, train_loader, epochs, batch_size, learning_rate, device)
    
    # Evaluate the trained model on the validation set
    val_loss = evaluate(trained_model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    # Save the trained model
    torch.save(trained_model.state_dict(), "./model/trained_model.pth")
    
    # Generate and evaluate midi files
    prompts = [
        "Compose a piece that evokes a sense of deep sorrow and loneliness. Use minor chords and slow tempo to create a mournful and reflective atmosphere. Try incorporating long, expressive phrases with subtle dynamics to enhance the emotional depth of the melody.",
        "Create an energetic and frantic music piece that captures a sense of urgency and turmoil. Experiment with rapid tempo changes and dissonant harmonies to convey a feeling of chaos and unrest. Incorporate dynamic shifts and abrupt pauses to add tension and unpredictability to the composition. Let the music build to a climactic and unsettling conclusion that leaves the listener on edge.",
        "Generate a relaxing jazz tune, very relaxing",
        "Generate a spooky and mysterious melody",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        generated_sequence = generate_midi(model, prompt, prompt_tokenizer, device)
        print(f"Generated sequence: {generated_sequence}")

    
    # Evaluate the model on the validation set using evaluation metrics
    # evaluate_model(trained_model, val_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the MusicBART model")
    parser.add_argument("--model_name", type=str, default="bart", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    args = parser.parse_args()
    main()