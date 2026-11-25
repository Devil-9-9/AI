class DiamondTokenizer:
    def __init__(self, vocab_size=50000):
        self.vocab_size = vocab_size
    def encode(self, s):
        return [ord(c) % self.vocab_size for c in s]
    def decode(self, ids):
        return ''.join([chr(i % 256) for i in ids])
