import re
from enum import StrEnum

class SpecialTokens(StrEnum):
    EOT = "<|endOfText|>"
    UNK = "<|unk|>"


# Throw an error when decode unknown words!
class Vocabulary:
    def __init__(self, data: str):
        tokens = self.__tokenize(data)

        # Creating token IDs
        all_tokens = sorted(set(tokens))
        all_tokens.extend([SpecialTokens.EOT.value, SpecialTokens.UNK.value])
        self.vocabulary = {token:id for id, token in enumerate(all_tokens)}

    def tokens(self):
        return self.vocabulary
        
    def size(self):
        size = len(self.vocabulary)
        return size
    
    def __tokenize(self, text: str):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token for token in tokens if token.strip()]
        
        return tokens