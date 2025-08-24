from src.tokenizer import TokenizerV1
from src.vocabulary import Vocabulary

with open("./data/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# test
print("Total number of character:", len(raw_text))
print(raw_text[:99])  # Print first 100 characters

vocabulary = Vocabulary(raw_text)
# test
PRINT_PAIRS = 50
for i, item in enumerate(vocabulary.tokens().items()):
    print(item, end=", ")
    
    if i >= PRINT_PAIRS:
        break
print("")
print("Vocabulary size2: ", vocabulary.size())

tokenizer1 = TokenizerV1(vocabulary.tokens())
text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer1.encode(text)
print(ids)

decoded_text = tokenizer1.decode(ids)
print(decoded_text)