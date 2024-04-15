import re

class MidiTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        
        self.build_vocab()
    
    def build_vocab(self):
        # Define the vocabulary of tokens
        tokens = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G',  # Pitch classes
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '/',  # Durations and separators
            '(', ')', '[', ']', '|', ':',  # Grouping and repeats
            '<', '>',  # Octave shifts
            '!', '_', '^', '=', '-',  # Accidentals and ties
            ' '  # Space
        ]
        
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token[self.vocab_size] = token
                self.vocab_size += 1
    
    def tokenize(self, abc_notation):
        # Tokenize the abc notation
        tokens = []
        
        # Split the abc notation into tokens based on space
        abc_tokens = abc_notation.split()
        
        for token in abc_tokens:
            # Split each token into pitch and duration
            match = re.match(r'^([A-Ga-g][#=]?)(\d+(\.\d+)?)$', token)
            
            if match:
                pitch, duration = match.group(1), match.group(2)
                tokens.extend([pitch, duration])
            else:
                # Handle special tokens
                tokens.extend(list(token))
        
        # Convert tokens to token IDs
        token_ids = [self.token_to_id[token] for token in tokens]
        
        return token_ids
    
    def detokenize(self, token_ids):
        # Convert token IDs back to tokens
        tokens = [self.id_to_token[token_id] for token_id in token_ids]
        
        # Reconstruct the abc notation from tokens
        abc_notation = ''
        
        i = 0
        while i < len(tokens):
            if tokens[i] in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
                pitch = tokens[i]
                duration = tokens[i+1] if i+1 < len(tokens) else ''
                abc_notation += pitch + duration + ' '
                i += 2
            else:
                abc_notation += tokens[i]
                i += 1
        
        return abc_notation.strip()