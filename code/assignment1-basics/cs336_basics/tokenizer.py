
# 构造tokenizer.py 
import re
from typing import List, Tuple, Dict, Iterator, IO, Optional

# 返回一个Tokenizer对象，初始化时传入上述vocab、merges和special_tokens
# Tokenizer的子函数主要有以下几个，下面通过输入输出举例说明其作用：

# 1. _bpe_encode(chunk: bytes) -> List[int]
# 作用：对一个字节串chunk进行BPE编码，返回token id的列表。
# 输入示例: chunk = b"Hello"
# 输出示例: [0, 1, 2, 2, 3]  # 假设vocab和merges如get_simple_tokenizer所示
# 说明：将"Hello"按BPE规则切分并映射为token id。

# 2. _get_pairs(word: List[bytes]) -> Set[Tuple[bytes, bytes]]
# 作用：给定一个字节token的列表，返回所有相邻token对的集合。
# 输入示例: word = [b"H", b"e", b"l", b"l", b"o"]
# 输出示例: {(b"H", b"e"), (b"e", b"l"), (b"l", b"l"), (b"l", b"o")}
# 说明：用于BPE合并时，找出所有可合并的token对。

# 3. encode(text: str) -> List[int]
# 作用：将字符串text编码为token id的列表，优先匹配特殊token，然后对剩余部分用BPE编码。
# 输入示例: text = "Hello<|endoftext|>!"
# 输出示例: [0, 1, 2, 2, 3, special_token_id, 8]
# 说明：先识别特殊token"<|endoftext|>"，其余部分用BPE编码。

# 4. decode(ids: List[int]) -> str
# 作用：将token id的列表还原为原始字符串。
# 输入示例: ids = [0, 1, 2, 2, 3, special_token_id, 8]
# 输出示例: "Hello<|endoftext|>!"
# 说明：将id映射回字节串并解码为字符串，特殊token也能还原。

# 这些内部函数共同实现了Tokenizer的核心功能：高效、可控地将文本和token id互相转换，并支持BPE和特殊token机制。

class Tokenizer:
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