from src.tokenizer import Tokenizer
from src.vocabulary import SpecialTokens, Vocabulary

with open("./data/the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# test
print("Total number of character:", len(raw_text))
print(raw_text[:99])  # Print first 100 characters

vocabulary = Vocabulary(raw_text)
# test
PRINT_PAIRS = 20
for i, item in enumerate(vocabulary.tokens().items()):
    print(item, end=", ")
    
    if i >= PRINT_PAIRS:
        break
print("")

print(f"Last {PRINT_PAIRS} token pairs: {list(vocabulary.tokens().items())[-PRINT_PAIRS:]}")
print("Vocabulary size: ", vocabulary.size())

tokenizer = Tokenizer(vocabulary.tokens())
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = f"{SpecialTokens.EOT.value} ".join((text1, text2))
print("Original text:", text)
ids = tokenizer.encode(text)
print(ids)

decoded_text = tokenizer.decode(ids)
print(decoded_text)