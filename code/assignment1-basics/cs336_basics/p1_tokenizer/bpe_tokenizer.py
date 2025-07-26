from typing import List, Tuple, Dict, Iterator, IO, Optional

class BPETokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.id_to_token = vocab.copy()  # int -> bytes
        self.token_to_id = {v: k for k, v in vocab.items()}  # bytes -> int
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [s.encode("utf-8") for s in self.special_tokens]
        self.special_token_to_id = {s.encode("utf-8"): i for i, s in enumerate(self.special_tokens, start=len(self.id_to_token)) if s.encode("utf-8") not in self.token_to_id}
        # Add special tokens to vocab if not present
        for b, i in self.special_token_to_id.items():
            self.id_to_token[i] = b
            self.token_to_id[b] = i
        # Build BPE ranks
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8")
        tokens = []
        i = 0
        while i < len(b):
            # 优先匹配最长的特殊token
            matched = False
            for s in sorted(self.special_token_bytes, key=len, reverse=True):
                if b[i:i+len(s)] == s:
                    tokens.append(self.token_to_id[s])
                    i += len(s)
                    matched = True
                    break
            if matched:
                continue
            # 找下一个特殊token出现的位置
            next_special = len(b)
            for s in self.special_token_bytes:
                pos = b.find(s, i)
                if pos != -1 and pos < next_special:
                    next_special = pos
            chunk = b[i:next_special]
            if chunk:
                tokens.extend(self._bpe_encode(chunk))
            i = next_special
        return tokens

    def _bpe_encode(self, chunk: bytes) -> List[int]:
        if not chunk:
            return []
        word = [bytes([c]) for c in chunk]
        pairs = self._get_pairs(word)
        while pairs:
            min_rank = float('inf')
            min_pair = None
            for pair in pairs:
                if pair in self.bpe_ranks and self.bpe_ranks[pair] < min_rank:
                    min_rank = self.bpe_ranks[pair]
                    min_pair = pair
            if min_pair is None:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
        ids = []
        # 如何合并后的词汇不在vocab里面，则需要拆开
        for w in word:
            if w in self.token_to_id:
                # 如果w是连续的\n，比如b"\n\n"，需要拆开
                if w == b"\n\n":
                    ids.append(self.token_to_id[b"\n"])
                    ids.append(self.token_to_id[b"\n"])
                else:
                    ids.append(self.token_to_id[w])
            else:
                # OOV: 拆成单字节
                for c in w:
                    ids.append(self.token_to_id[bytes([c])])
        return ids

    def _get_pairs(self, word) -> set[tuple[bytes, bytes]]:
        pairs: set[tuple[bytes, bytes]] = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs

    def decode(self, ids: List[int]) -> str:
        # 先将id映射为bytes，再decode
        b = b"".join([self.id_to_token[i] for i in ids])
        try:
            return b.decode("utf-8")
        except Exception:
            return b.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: IO) -> Iterator[int]:
        # 逐行读取，保留所有换行符（包括连续的换行和特殊token混合的情况）
        for line in iterable:
            # 检查line是否以\n结尾
            has_newline = line.endswith("\n")
            # 去掉末尾的\n进行编码
            line_content = line[:-1] if has_newline else line
            for token in self.encode(line_content):
                yield token
            if has_newline:
                # 只编码一个换行符
                for token in self.encode("\n"):
                    yield token