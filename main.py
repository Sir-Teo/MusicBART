# main.py
from data_preprocessing import preprocess_data, load_dataset
from tokenizer import PromptTokenizer, MidiTokenizer
from model import MusicBART, train, evaluate
from evaluation import evaluate_model, evaluate_midi
import torch


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
    dataset_path = "data/prompt_midi_cropped.json"
    dataset = load_dataset(dataset_path)
    dataset = preprocess_data(dataset)
    
    # Create the tokenizers
    prompt_tokenizer = PromptTokenizer()
    midi_tokenizer = MidiTokenizer()
    # Initialize the MusicBART model
    model = MusicBART().to(device)
    
    # Tokenize the dataset
    tokenized_dataset = []
    tokenized_dataset = []
    for prompt, midi_data in dataset:
        input_ids = prompt_tokenizer.tokenize(prompt)
        labels = midi_tokenizer.tokenize(midi_data)
        tokenized_dataset.append({"input_ids": input_ids, "labels": labels})

        #print(f"Prompt: {prompt}")
        #print(f"Input IDs: {input_ids}")
        #print(f"MIDI Data: {midi_data}")
        #print(f"Labels: {labels}")
        #print("---")
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset[:train_size]
    val_dataset = tokenized_dataset[train_size:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda batch: collate_fn(batch, device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, collate_fn=lambda batch: collate_fn(batch, device))
    
    
    # Set the training parameters
    epochs = 50
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
    num_samples = 10
    for _ in range(num_samples):
        prompt_tensor = torch.tensor(prompt_tokenizer.tokenize(prompt)).to(device)
        # Add batch dimension to prompt_tensor
        prompt_tensor = prompt_tensor.unsqueeze(0)
        # Generate the attention mask for the prompt tensor
        attention_mask = torch.ones(prompt_tensor.shape, dtype=torch.long, device=device)

        # Generate the sequence
        generated_sequence = trained_model.generate(prompt_tensor, attention_mask)
        #evaluate_midi(generated_sequence)
    
    # Evaluate the model on the validation set using evaluation metrics
    evaluate_model(trained_model, val_loader, device)

if __name__ == "__main__":
    main()