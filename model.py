import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer

class MusicBART(nn.Module):
    def __init__(self, model_name="facebook/bart-base", max_length=512):
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
    
    def generate(self, prompt, num_beams=4, max_length=512):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        outputs = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True
        )
        
        generated_sequence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_sequence

def train(model, dataset, epochs, batch_size, learning_rate, device):
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch in dataset:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
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