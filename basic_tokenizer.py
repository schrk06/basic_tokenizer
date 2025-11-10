class BasicTokenizer:
    def __init__(self, vocab_size=512):
        # vocab initial : 0-255 pour chaque byte
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.bigram_tree = {}
        self.vocab_size = vocab_size

    def freq(self, tokens):
        """Compter les bigrams consécutifs dans une liste de tokens"""
        stats = {}
        for id1, id2 in zip(tokens, tokens[1:]):
            stats[(id1, id2)] = stats.get((id1, id2), 0) + 1
        return stats

    def replace(self, tokens, pair, idx):
        """Remplacer les occurrences d'une paire par un nouvel idx"""
        newtokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == pair[0] and i < len(tokens) - 1 and tokens[i+1] == pair[1]:
                newtokens.append(idx)
                i += 2
            else:
                newtokens.append(tokens[i])
                i += 1
        return newtokens

    def train(self, txt):
        """Entraîne le vocab BPE sur un texte"""
        tokens = list(txt.encode("utf-8"))
        req = self.vocab_size - 256

        for i in range(req):
            stats = self.freq(tokens)
            if not stats:  # plus de bigrams à fusionner
                break
            # prendre la paire la plus fréquente
            maxi = max(stats, key=stats.get)
            tokens = self.replace(tokens, maxi, 256 + i)
            self.bigram_tree[maxi] = 256 + i

        # construire le vocab final
        for (p0, p1), idx in self.bigram_tree.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, txt):
        """Encode un texte en liste de tokens BPE"""
        tokenz = list(txt.encode("utf-8"))

        # convertir en int pour correspondre au vocab
        tokenz = list(map(int, tokenz))

        while True:
            stats = self.freq(tokenz)
            # garder seulement les bigrams qui sont dans le vocab
            candidates = [pair for pair in stats if pair in self.bigram_tree]
            if not candidates:
                break
            # prendre la paire la plus fréquente
            maxi = max(candidates, key=lambda p: stats[p])
            tokenz = self.replace(tokenz, maxi, self.bigram_tree[maxi])

        return tokenz

    def decode(self, tokens):
        """Retourne le texte UTF-8 à partir des tokens BPE"""
        t = b"".join(self.vocab[idx] for idx in tokens)
        return t.decode("utf-8", errors="replace")
