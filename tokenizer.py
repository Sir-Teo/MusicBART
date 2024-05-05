# tokenizer.py

import re
from transformers import BartTokenizer


class PromptTokenizer:
    def __init__(self):
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def tokenize(self, prompt):
        # Tokenize the prompt text
        
        tokens = self.tokenizer.tokenize(prompt)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def detokenize(self, token_ids):
        # Convert token IDs back to prompt text
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        prompt = self.tokenizer.convert_tokens_to_string(tokens)
        return prompt
        
class MidiTokenizer:
    def __init__(self):
        self.special_tokens = set([
            '[', ']', '|', '\\n', ':', '::', '/', '(', ')', '{', '}', '!', '+', '-', '*', '"', '%', '$', '<', '>', '\\'
        ])
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0

    def _update_vocab(self, token):
        if token not in self.token_to_id:
            self.token_to_id[token] = self.vocab_size
            self.id_to_token[self.vocab_size] = token
            self.vocab_size += 1

    def tokenize(self, midi_notation):
        pattern = r'([{}\s])'.format(''.join(re.escape(tok) for tok in self.special_tokens))
        tokens = [token for token in re.split(pattern, midi_notation) if token]
        token_ids = []
        for token in tokens:
            if token not in self.token_to_id:
                self._update_vocab(token)
            token_ids.append(self.token_to_id[token])
        return token_ids

    def detokenize(self, token_ids):
        tokens = [self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token]
        return ''.join(tokens).replace('\\n', '\n')

if __name__ == '__main__':
    tokenizer = MidiTokenizer()
    sample_text = "X: 1\nM: 4/4\nL: 1/8\nQ:1/4=120\nK:C % 0 sharps\nV:1\n%%clef treble\nzA,- [E-A,]3/2[B-E-]/2 [c-B-E-][e-c-BE-] [g-ec-E-]/2[g-e-cE-]/2[g-e-E-]/2[g-e-c-E-]/2| \\\n[g-e-c-B-E]2 [g-e-c-B-E-]/2[g-e-c-B-EA,-][g-ec-BE-A,-]3/2[g-cB-E-A,-]/2[g-B-E-A,-]/2 [g-c-B-E-A,-]/2[ge-c-B-E-A,-][g-ec-B-E-A,-]/2| \\\n[g-e-c-B-E-A,]/2[g-e-cB-E-]/2[g-e-c-BE-] [g-e-c-B-E]3/2[g-ecBE-][g-E-]/2[gE-C-] [E-C-]/2[G-EC-][A-G-C-]/2| \\\n[c-A-G-C-][e-cA-G-C-C,,-]/2[e-A-G-C-C,,-]/2 [e-c-AG-C-C,,-]/2[e-c-AGC-C,,][e-c-A-G-C]/2"
    token_ids = tokenizer.tokenize(sample_text)
    text = tokenizer.detokenize(token_ids)
