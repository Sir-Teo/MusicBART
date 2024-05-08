# tests.py

import torch

# Define the collate function (as from your script)
def collate_fn(batch, device):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0).to(device)
    attention_mask = torch.where(input_ids != 0, torch.ones_like(input_ids), torch.zeros_like(input_ids)).to(device)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Create a mock batch of data
mock_batch = [
    {'input_ids': [1, 2, 3], 'labels': [1, 2, 3, 4, 5]},
    {'input_ids': [1, 2], 'labels': [1, 2, 3]}
]

# Device for tensors (assuming CPU for simplicity in testing)
device = torch.device("cpu")

# Call the collate function with the mock batch
output = collate_fn(mock_batch, device)

# Print the outputs
print("Input IDs:", output['input_ids'])
print("Labels:", output['labels'])
print("Attention Mask:", output['attention_mask'])
