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
        self.token_to_id = {}
        self.id_to_token = {}
        self.vocab_size = 0
        self.build_vocab()

    def build_vocab(self):
        tokens = [
                    'A', 'B', 'C', 'D', 'E', 'F', 'G',  # Uppercase pitch classes
                    'a', 'b', 'c', 'd', 'e', 'f', 'g',  # Lowercase pitch classes
                    '^', '_', '=',  # Accidentals (sharp, flat, natural)
                    ',', '\'',  # Octave modifiers (lower and upper octave)
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Durations
                    '/', '.',  # Duration modifiers (division and dot)
                    '(', ')',  # Slurs and ties
                    '[', ']',  # Chord brackets
                    '|',  # Barline
                    ':', '::',  # Repeats
                    '{', '}',  # Grace note braces
                    '!', '+', '-', '*',  # Ornaments and articulations
                    '"',  # Chord symbol quotes
                    '%', '$',  # Directives and inline fields
                    '<', '>',  # Dynamics
                    'V&',  # Voice overlay (pair of tokens)
                    '\\',  # Escape character
                    'z',  # Rest
                    ' ',  # Space
                    'X:', 'T:', 'C:', 'L:', 'M:', 'Q:', 'P:', 'V:', 'K:',  # Header fields
                    'w:', 'W:', 'u:', 'U:', 'H:', 'I:', 'J:', 'R:', 'S:', 's:', 'y:', 'Y:', 'Z:',  # Inline fields
                    
                    # Key signatures
                    'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F#m', 'E', 'C#m', 'B', 'G#m', 'F#', 'D#m',
                    'C#', 'A#m', 'F', 'Dm', 'Bb', 'Gm', 'Eb', 'Cm', 'Ab', 'Fm', 'Db', 'Bbm', 'Gb', 'Ebm',
                    
                    # Chord symbols
                    'm', 'M', 'maj', 'min', 'dim', 'aug', 'sus2', 'sus4', '7', 'M7', 'maj7', 'm7', 'dim7',
                    'aug7', 'sus2/7', 'sus4/7', 'add9', 'm7b5', '6', 'm6', 'M9', 'maj9', 'm9', '9', '11',
                    '13', 'add11', 'add13', '7#5', '7b5', '7#9', '7b9', '7#11', '7b13', 'o', '+', 'ø',
                    'Δ', 'ᵒ', '⁺', '⁻', 'ᶿ', 'ᵇ', 'ᵍ', 'ʳ', 'ᶜ', 'ᵈ', 'ᵉ', 'ˢ', 'ˡ', 'ʲ', 'ᵐ', 'ⁿ'
                ]
        for token in tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.id_to_token[self.vocab_size] = token
                self.vocab_size += 1

    def tokenize(self, midi_notation):
        if isinstance(midi_notation, list):
            midi_notation = ' '.join(midi_notation)

        tokens = []
        i = 0
        while i < len(midi_notation):
            if midi_notation[i] == '\\' and i < len(midi_notation) - 1:
                if midi_notation[i + 1] == 'n':
                    tokens.append('\\n')
                    i += 2
                else:
                    tokens.append(midi_notation[i])
                    i += 1
            elif midi_notation[i] == '[':
                j = i + 1
                while j < len(midi_notation) and midi_notation[j] != ']':
                    j += 1
                if j < len(midi_notation):
                    tokens.extend(['['] + re.findall(r'[A-Ga-g][#b]?\d+|\d+', midi_notation[i + 1:j]) + [']'])
                    i = j + 1
                else:
                    tokens.append(midi_notation[i])
                    i += 1
            elif re.match(r'[A-Ga-g][#b]?\d+', midi_notation[i:]):
                match = re.match(r'[A-Ga-g][#b]?\d+', midi_notation[i:])
                tokens.extend(re.findall(r'[A-Ga-g][#b]?|\d+', match.group()))
                i += len(match.group())
            elif re.match(r'\d+/', midi_notation[i:]):
                match = re.match(r'\d+/', midi_notation[i:])
                tokens.extend(re.findall(r'\d+|/', match.group()))
                i += len(match.group())
            elif midi_notation[i] == '&':
                tokens.extend(['V&'])  # Voice overlay as a pair of tokens
                i += 1
            else:
                tokens.append(midi_notation[i])
                i += 1

        token_ids = [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        return token_ids

    def detokenize(self, token_ids):
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(f"<UNKNOWN_{token_id}>")
        detokenized_text = "".join(tokens)
        return detokenized_text