# model.py

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tokenizer import MidiTokenizer, PromptTokenizer

class MusicGPT(nn.Module):
    def __init__(self, model_name="gpt2", max_length=128):
        super(MusicGPT, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.prompt_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.midi_tokenizer = MidiTokenizer()  # Add this line to instantiate the MidiTokenizer
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, num_beams=8, max_length=256):
        num_beams = int(num_beams)  # Convert num_beams to a scalar value
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            attention_mask=attention_mask,
            num_beams=num_beams,
        ) 
        generated_sequence = outputs[0].tolist()  # Convert the generated sequence to a list
        midi_sequence = self.midi_tokenizer.detokenize(generated_sequence)  # Use the instance to call detokenize
        return midi_sequence


class MusicBART(nn.Module):
    def __init__(self, model_name="facebook/bart-large", max_length=256):
        super(MusicBART, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.prompt_tokenizer = PromptTokenizer()
        self.midi_tokenizer = MidiTokenizer()

        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, num_beams=5, max_length=256):
        num_beams = int(num_beams)  # Convert num_beams to a scalar value
        outputs = self.model.generate(
        input_ids=input_ids,
        max_length=max_length,
        attention_mask=attention_mask,
        num_beams=num_beams,
        ) 
        generated_sequence = outputs[0].tolist()  # Convert the generated sequence to a list
        print(generated_sequence)
        midi_sequence = self.midi_tokenizer.detokenize(generated_sequence)  # Use the instance to call detokenize
        return midi_sequence

def train(model, train_loader, epochs, batch_size, learning_rate, device, accumulation_steps=1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    total_loss = 0
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss = loss / accumulation_steps  # Normalize the loss to account for accumulation
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:  # Perform optimization at the defined accumulation step
                optimizer.step()
                optimizer.zero_grad()  # Clear gradients after updating weights

            total_loss += loss.item()

            # Optional: Clear unused memory to avoid fragmentation
            if batch_idx % 10 == 0:  # Every 10 batches
                torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")
        total_loss = 0  # Reset total loss after each epoch

    model.eval()
    return model


def evaluate(model, dataset, device):
    model.to(device)
    model.eval()
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataset)
    print(f"Evaluation Loss: {avg_loss:.4f}")
    
    return avg_loss