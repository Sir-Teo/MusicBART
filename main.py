from data_preprocessing import preprocess_data
from tokenizer import MidiTokenizer
from model import MusicBART, train, evaluate
from evaluation import evaluate_model
import torch

def main():
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the dataset
    dataset_path = "path/to/your/dataset"
    dataset = preprocess_data(dataset_path)
    
    # Create the tokenizer
    tokenizer = MidiTokenizer()
    
    # Tokenize the dataset
    tokenized_dataset = []
    for prompt, midi_tokens in dataset:
        input_ids = tokenizer.tokenize(prompt)
        labels = midi_tokens
        tokenized_dataset.append({"input_ids": input_ids, "labels": labels})
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset[:train_size]
    val_dataset = tokenized_dataset[train_size:]
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8)
    
    # Initialize the MusicBART model
    model = MusicBART()
    
    # Set the training parameters
    epochs = 10
    batch_size = 8
    learning_rate = 1e-4
    
    # Train the model
    trained_model = train(model, train_loader, epochs, batch_size, learning_rate, device)
    
    # Evaluate the trained model on the validation set
    val_loss = evaluate(trained_model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Generate and evaluate midi files
    num_samples = 10
    for _ in range(num_samples):
        prompt = "Sample prompt"
        generated_sequence = trained_model.generate(prompt)
        generated_midi = sequence_to_midi(generated_sequence)
        evaluate_midi(generated_midi)
    
    # Evaluate the model on the validation set using evaluation metrics
    evaluate_model(trained_model, val_loader, device)

if __name__ == "__main__":
    main()