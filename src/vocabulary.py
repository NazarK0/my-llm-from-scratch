import re


# Throw an error when decode unknown words!
class Vocabulary:
    def __init__(self, data: str):
        tokens = self.__tokenize(data)

        # Creating token IDs
        all_words = sorted(set(tokens))
        self.vocabulary = {token:id for id, token in enumerate(all_words)}
        
    def tokens(self):
        return self.vocabulary
        
    def size(self):
        size = len(self.vocabulary)
        return size
    
    def __tokenize(self, text: str):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token for token in tokens if token.strip()]
        
        return tokens