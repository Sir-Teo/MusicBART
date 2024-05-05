import json
import re

class MidiTokenizer:
    def __init__(self, max_length=1024, padding=False, pad_token="[PAD]", special_vocabs=None):
        self.max_length = max_length
        self.padding = padding
        self.pad_token = pad_token

        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

        # Add special vocabs, including pad_token
        self.special_vocabs = [pad_token] if special_vocabs is None else [pad_token] + special_vocabs
        for vocab in self.special_vocabs:
            self._update_vocab(vocab)

    def _update_vocab(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def tokenize(self, midi_notation):
        # Tokenize using regex that captures fractions and standalone digits, and ensures special vocabs are recognized
        special_pattern = r'|'.join(re.escape(vocab) for vocab in self.special_vocabs)
        general_pattern = r'\b\d+/\d+\b|\b\d+\b|.'
        pattern = rf'({special_pattern})|({general_pattern})'
        tokens = [token for token in re.findall(pattern, midi_notation) if token]
        tokens = [token[0] or token[1] for token in tokens]  # Flatten tuple

        token_ids = []
        for token in tokens:
            if token not in self.token_to_id:
                self._update_vocab(token)
            token_ids.append(self.token_to_id[token])

        # Handle truncation
        if self.max_length and len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        # Handle padding
        if self.padding and self.max_length:
            pad_id = self.token_to_id[self.pad_token]
            token_ids = token_ids + [pad_id] * (self.max_length - len(token_ids))

        return token_ids

    def detokenize(self, token_ids):
        tokens = [self.id_to_token.get(token_id, "[UNK]") for token_id in token_ids]
        result = ''.join(tokens)
        return result.replace(self.pad_token, '')

    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
                'vocab_size': self.vocab_size
            }, file)

    def load_vocab(self, file_path):
        with open(file_path, 'r') as file:
            vocab = json.load(file)
            self.token_to_id = vocab['token_to_id']
            self.id_to_token = {int(k): v for k, v in vocab['id_to_token'].items()}
            self.vocab_size = vocab['vocab_size']

    def save_tokenizer(self, file_path):
        with open(file_path, 'w') as file:
            json.dump({
                'max_length': self.max_length,
                'padding': self.padding,
                'pad_token': self.pad_token,
                'token_to_id': self.token_to_id,
                'id_to_token': {int(k): v for k, v in self.id_to_token.items()},
                'vocab_size': self.vocab_size,
                'special_vocabs': self.special_vocabs
            }, file)

    @staticmethod
    def load_tokenizer(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            tokenizer = MidiTokenizer(
                max_length=data['max_length'],
                padding=data['padding'],
                pad_token=data['pad_token'],
                special_vocabs=data['special_vocabs']
            )
            tokenizer.token_to_id = data['token_to_id']
            tokenizer.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
            tokenizer.vocab_size = data['vocab_size']
            return tokenizer


if __name__ == '__main__':
    special_vocabs = ["clef", "sharp", "\n"]
    tokenizer = MidiTokenizer(max_length=1024, padding=True, special_vocabs=special_vocabs)
    sample_text = "X: 1\nM: 4/4\nL: 1/8\nQ:1/4=120\nK:C % 0 sharps\nV:1\n%%clef treble\nzA,- [E-A,]3/2[B-E-]/2 [c-B-E-][e-c-BE-] [g-ec-E-]/2[g-e-cE-]/2[g-e-E-]/2[g-e-c-E-]/2| \\\n[g-e-c-B-E]2 [g-e-c-B-E-]/2[g-e-c-B-EA,-][g-ec-BE-A,-]3/2[g-cB-E-A,-]/2[g-B-E-A,-]/2 [g-c-B-E-A,-]/2[ge-c-B-E-A,-][g-ec-B-E-A,-]/2| \\\n[g-e-c-B-E-A,]/2[g-e-cB-E-]/2[g-e-c-BE-] [g-e-c-B-E]3/2[g-ecBE-][g-E-]/2[gE-C-] [E-C-]/2[G-EC-][A-G-C-]/2| \\\n[c-A-G-C-][e-cA-G-C-C,,-]/2[e-A-G-C-C,,-]/2 [e-c-AG-C-C,,-]/2[e-c-AGC-C,,][e-c-A-G-C]/2"
    token_ids = tokenizer.tokenize(sample_text)
    print(f"Token IDs: {token_ids}")
    print(f"Token to ID mapping: {tokenizer.token_to_id}")
    print(f"ID to Token mapping: {tokenizer.id_to_token}")
    text = tokenizer.detokenize(token_ids)
    print(f"Detokenized text: {text}")

    # Save and load the tokenizer
    tokenizer.save_tokenizer("tokenizer.json")
    new_tokenizer = MidiTokenizer.load_tokenizer("tokenizer.json")
    new_text = new_tokenizer.detokenize(token_ids)
    print(f"New Detokenized text: {new_text}")
