# main_parallel# main.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from data_preprocessing import preprocess_data, load_dataset
from tokenizer import PromptTokenizer, MidiTokenizer
from model import MusicBART, train, evaluate, MusicGPT
from evaluation import evaluate_model, evaluate_midi
import argparse


def collate_fn(batch, device):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).to(device)
    attention_mask = torch.where(input_ids != 0, torch.ones_like(input_ids), torch.zeros_like(input_ids)).to(device)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Set the device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Preprocess the dataset
    dataset_path = "data/prompts_clean.json"
    dataset = load_dataset(dataset_path)
    dataset = preprocess_data(dataset)
    
    # Create the tokenizers
    prompt_tokenizer = PromptTokenizer()
    midi_tokenizer = MidiTokenizer()
    
    # Initialize the MusicBART model
    if args.model_name == "bart":
        model = MusicBART().to(device)
    elif args.model_name == "gpt":
        model = MusicGPT().to(device)
    
    # Wrap the model with DDP
    model = DDP(model, device_ids=[rank])
    
    # Tokenize the dataset
    tokenized_dataset = []
    for prompt, midi_data in dataset:
        input_ids = prompt_tokenizer.tokenize(prompt)
        labels = midi_tokenizer.tokenize(midi_data)
        tokenized_dataset.append({"input_ids": input_ids, "labels": labels})
    
    # Split the dataset into train and validation sets
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset[:train_size]
    val_dataset = tokenized_dataset[train_size:]
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=lambda batch: collate_fn(batch, device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, sampler=val_sampler, collate_fn=lambda batch: collate_fn(batch, device))
    
    # Set the training parameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # Train the model
    trained_model = train(model, train_loader, epochs, batch_size, learning_rate, device)
    
    # Evaluate the trained model on the validation set
    val_loss = evaluate(trained_model, val_loader, device)
    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the trained model
    if rank == 0:
        torch.save(trained_model.module.state_dict(), "./model/trained_model.pth")
    
    # Generate and evaluate midi files
    num_samples = 10
    if rank == 0:
        for _ in range(num_samples):
            prompt_tensor = torch.tensor(prompt_tokenizer.tokenize(prompt)).to(device)
            # Add batch dimension to prompt_tensor
            prompt_tensor = prompt_tensor.unsqueeze(0)
            # Generate the attention mask for the prompt tensor
            attention_mask = torch.ones(prompt_tensor.shape, dtype=torch.long, device=device)

            # Generate the sequence
            generated_sequence = trained_model.module.generate(prompt_tensor, attention_mask)
            #evaluate_midi(generated_sequence)
    
    # Evaluate the model on the validation set using evaluation metrics
    if rank == 0:
        evaluate_model(trained_model.module, val_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the MusicBART model")
    parser.add_argument("--model_name", type=str, default="bart", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    args = parser.parse_args()
    main()