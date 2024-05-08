# main_parallel.py
# srun --time=3:00:00 --mem=64GB --cpus-per-task=2 --gres=gpu:2 --pty /bin/bash
# torchrun --nproc_per_node=2 main_parallel.py --model_name bart --epochs 10 --batch_size 6 --learning_rate 1e-5
import torch
import os
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from data_preprocessing import preprocess_data, load_dataset
from model import MusicBART, train, evaluate, MusicGPT
from test_model import generate_midi

def collate_fn(batch, device):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).to(device)
    attention_mask = torch.where(input_ids != 0, torch.ones_like(input_ids), torch.zeros_like(input_ids)).to(device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def log_memory_usage():
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    print(f"Allocated memory: {allocated} GB, Cached memory: {cached} GB")

def main():
    # Get the local rank from the environment variable
    local_rank = int(os.environ['LOCAL_RANK'])

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set the device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)  # Explicitly setting device

    # Preprocess and load the dataset
    dataset_path = "data/prompts_clean.json"
    dataset = load_dataset(dataset_path)
    dataset = preprocess_data(dataset)
    log_memory_usage()

    # Initialize the MusicBART model
    if args.model_name == "bart":
        model = MusicBART().to(device)
    elif args.model_name == "gpt":
        model = MusicGPT().to(device)
    prompt_tokenizer = model.prompt_tokenizer
    midi_tokenizer = model.midi_tokenizer
    model = DDP(model, device_ids=[local_rank])

    # Tokenize the dataset
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
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=lambda batch: collate_fn(batch, device))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, collate_fn=lambda batch: collate_fn(batch, device))

    # Train and evaluate the model
    trained_model = train(model, train_loader, args.epochs, args.batch_size, args.learning_rate, device)
    val_loss = evaluate(trained_model, val_loader, device)
    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}")
        torch.save(trained_model.module.state_dict(), "./model/trained_model.pth")

        prompts = [
            "Compose a piece that evokes a sense of deep sorrow and loneliness. Use minor chords and slow tempo to create a mournful and reflective atmosphere. Try incorporating long, expressive phrases with subtle dynamics to enhance the emotional depth of the melody.",
            "Craft a piece that builds like a tidal wave, with swelling sound layers that represent a surge of overwhelming emotions",
            "Generate a relaxing jazz tune, very relaxing",
            "Generate a spooky and mysterious melody",
        ]

        for prompt in prompts:
            print(f"Prompt: {prompt}")
            generated_sequence = generate_midi(model.module, prompt, prompt_tokenizer, device)
            print(f"Generated sequence: {generated_sequence}")
        
        midi_tokenizer.save_vocab('tokenizer.json')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the MusicBART model")
    parser.add_argument("--model_name", type=str, default="bart", help="Pretrained model name")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    args = parser.parse_args()
    main()
