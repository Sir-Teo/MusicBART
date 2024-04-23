# model.py

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from tokenizer import MidiTokenizer, PromptTokenizer

class MusicBART(nn.Module):
    def __init__(self, model_name="facebook/bart-large", max_length=4096):
        super(MusicBART, self).__init__()
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, input_ids, attention_mask, num_beams=4, max_length=4096):
        num_beams = int(num_beams)  # Convert num_beams to a scalar value
        outputs = self.model.generate(
        input_ids=input_ids,
        max_length=max_length,
        attention_mask=attention_mask,
        num_beams=num_beams,
        ) 
        generated_sequence = outputs[0].tolist()  # Convert the generated sequence to a list
        midi_tokenizer = MidiTokenizer()  # Create an instance of MidiTokenizer
        midi_sequence = midi_tokenizer.detokenize(generated_sequence)  # Use the instance to call detokenize
        return midi_sequence

def train(model, train_loader, epochs, batch_size, learning_rate, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss


            total_loss = 0
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

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