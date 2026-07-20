from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer/tokenizer.json"
)

text = "The farmer grows wheat"

encoded = tokenizer.encode(text)

print("Original:")
print(text)

print("\nIDs:")
print(encoded.ids)

print("\nTokens:")
print(encoded.tokens)

decoded = tokenizer.decode(encoded.ids)

print("\nDecoded:")
print(decoded)