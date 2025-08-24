import re
from src.vocabulary import SpecialTokens

def convert_to_tokens(text: str):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

# Throw an error when decode unknown words!
class Tokenizer:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {id:token for token, id in vocabulary.items()}
        
    def encode(self, text):
        tokens = convert_to_tokens(text)
        tokens = [
            item if item in self.str_to_int
            else SpecialTokens.UNK.value for item in tokens
        ]
        ids = [self.str_to_int[token] for token in tokens]
        
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[id] for id in ids])

        # Remove spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text