import re

def convert_to_tokens(text: str):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [token for token in tokens if token.strip()]
    
    return tokens

# Throw an error when decode unknown words!
class TokenizerV1:
    def __init__(self, vocabulary):
        self.str_to_int = vocabulary
        self.int_to_str = {id:token for token, id in vocabulary.items()}
        
    def encode(self, text):
        tokens = convert_to_tokens(text)
        ids = [self.str_to_int[s] for s in tokens]
        
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        
        # Remove spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        
        return text