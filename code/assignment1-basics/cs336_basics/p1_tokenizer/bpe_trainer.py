import os
from collections import Counter, defaultdict
from typing import Any, TypeAlias
from .bpe_batcher import BPETokenizerBatcher
from cs336_basics.util.logger import Logger

Pair: TypeAlias = tuple[int, int]  # 例如: (65, 66) 表示两个token id的pair
PairCountDict: TypeAlias = dict[Pair, int]  # 例如: {(65, 66): 10, (66, 67): 5} 表示pair出现的次数
Pretoken: TypeAlias = list[bytes]  # 例如: [b'hello', b' world'] 表示一个tokens bytes序列
Pretokens: TypeAlias = list[Pretoken]  # 例如: [b'hello', b' world'] 表示一个tokens bytes序列
Vocab: TypeAlias = dict[int, bytes]  # 例如: {65: b'a', 66: b'b'} 表示token id到字节的映射
Merges: TypeAlias = list[tuple[bytes, bytes]]  # 例如: [(b'a', b'b'), (b'b', b'c')] 表示BPE合并规则

class BPETrainer:
    BYTE_VOCAB_SIZE = 256  # 字节级初始词表大小

    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None, num_processes: int = 4):
        if special_tokens is None:
            special_tokens = []
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = num_processes
        self.num_merges = max(self.vocab_size - len(self.special_tokens) - self.BYTE_VOCAB_SIZE, 0)
        self.batcher = BPETokenizerBatcher(special_tokens, num_processes)
        self.logger = Logger("bpe_trainer.log")

    def train(self, input_path: str | os.PathLike) -> tuple[Vocab, Merges]:
        """主训练流程，返回最终的vocab和merges。"""
        vocab = self._init_vocab()
        
        # self.logger.log(f"初始化vocab，前5项: {list(vocab.items())[:5]}")
        chunks = self.batcher.load_chunks(input_path)
        
        # self.logger.log(f"分块后chunks前5项: {chunks[:5]}")
        all_tokens = self.batcher.pretokenize_chunks(chunks)

        # 在这个位置将all_pretokens从类型list[bytes]转换为类型list[list[int]]，来方便self._count_pairs和self._bpe_merge_loop的处理
        all_tokens_id = [[b for b in token] for token in all_tokens] # 对字节对象遍历的时候会直接取成int
        
        # self.logger.log(f"预分词后pretokens前5项: {all_pretokens[:5]}")
        pair_counts, pair_to_pretoken_indices = self._count_pairs(all_tokens_id)
        
        # self.logger.log(f"pair_counts前5项: {list(pair_counts.items())[:5]}")
        merges = self._bpe_merge_loop(vocab, pair_counts, pair_to_pretoken_indices, all_tokens_id)
        
        # self.logger.log(f"merges前5项: {merges[:5]}")
        return vocab, merges

    def _init_vocab(self) -> Vocab:
        """初始化词表，包含所有单字节token和特殊token。"""
        vocab: Vocab = {x: bytes([x]) for x in range(self.BYTE_VOCAB_SIZE)}
        for i, token in enumerate(self.special_tokens):
            vocab[self.BYTE_VOCAB_SIZE + i] = token.encode("utf-8")
        return vocab

    def _count_pairs(self, pretokens: list[list[int]]) -> tuple[PairCountDict, dict[Pair, set[int]]]:
        """
        统计所有token对的出现频率。
        假设pretokens的类型为list[list[bytes]]，即每个pretoken是一个bytes的列表。
        这里我们将每个bytes token映射为一个唯一的int id（例如直接用bytes的int值），
        并统计相邻token id对的出现次数。
        """
        self.logger.log(f"_count_pairs输入pretokens前5项: {pretokens[:5]}")
        pair_counts: PairCountDict = defaultdict(int)
        pair_to_pretoken_indices: dict[Pair, set[int]] = defaultdict(set)
        for idx, token_ids in enumerate(pretokens):
            # 将每个bytes token转换为int（假设每个token为单字节bytes）
            for t1, t2 in zip(token_ids, token_ids[1:]):
                pair_counts[(t1, t2)] += 1
                pair_to_pretoken_indices[(t1, t2)].add(idx)
        self.logger.log(f"_count_pairs输出pair_counts前5项: {list(pair_counts.items())[:5]}")
        return pair_counts, pair_to_pretoken_indices   

    @staticmethod
    def _merge_pair(
        pair_counts: PairCountDict,
        pair_to_pretoken_indices: dict[Pair, set[int]],
        all_pretokens_ids: list[list[int]],
        max_pair: Pair,
        new_index: int
    ):
        """执行一次BPE合并操作，更新pretokens和pair计数。"""
        index_set = pair_to_pretoken_indices[max_pair].copy()
        for j in index_set:
            token_ids = all_pretokens_ids[j]
            new_token_ids = []
            i = 0
            while i < len(token_ids):
                if i < len(token_ids)-1 and (token_ids[i], token_ids[i+1]) == max_pair:
                    new_token_ids.append(new_index)
                    i += 2
                else:
                    new_token_ids.append(token_ids[i])
                    i += 1
                    
            # 2. 统计旧pair和新pair
            old_pairs = Counter(zip(token_ids, token_ids[1:]))
            new_pairs = Counter(zip(new_token_ids, new_token_ids[1:]))
            
            # 3. 统计消失的pair和新增的pair，更新pair_counts和pair_to_pretoken_indices
            # 删除
            for pair, count in (old_pairs - new_pairs).items():
                pair_counts[pair] -= count
                if j in pair_to_pretoken_indices[pair] and new_pairs[pair] == 0:
                    pair_to_pretoken_indices[pair].discard(j)
            
            # 添加
            for pair, count in (new_pairs - old_pairs).items():
                pair_counts[pair] += count
                pair_to_pretoken_indices[pair].add(j)

            all_pretokens_ids[j] = new_token_ids

    def _bpe_merge_loop(
        self,
        vocab: Vocab,
        pair_counts: PairCountDict,
        pair_to_pretoken_indices: dict[Pair, set[int]],
        all_pretokens_id: list[list[int]]
    ) -> Merges:
        """BPE主合并循环，返回合并规则。"""
        self.logger.log(f"_bpe_merge_loop输入pair_counts前5项: {list(pair_counts.items())[:5]}")
        merges: Merges = []
        for i in range(self.num_merges):
            if not pair_counts:
                break
            max_pair = max(
                pair_counts.items(),
                key=lambda x: (
                    x[1],
                    vocab[x[0][0]].decode("utf-8", errors="ignore"),
                    vocab[x[0][1]].decode("utf-8", errors="ignore")
                )
            )[0]
            if pair_counts[max_pair] == 0:
                break
            new_index = self.BYTE_VOCAB_SIZE + len(self.special_tokens) + i
            vocab[new_index] = vocab[max_pair[0]] + vocab[max_pair[1]]
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            # 打印每一轮新增的vocab和merge
            self.logger.log(f"第{i+1}轮: 新增vocab[{new_index}] = {vocab[new_index]!r}, merge = ({vocab[max_pair[0]]!r}, {vocab[max_pair[1]]!r})")
            self._merge_pair(pair_counts, pair_to_pretoken_indices, all_pretokens_id, max_pair, new_index)
        return merges

def test_counter():
    # INSERT_YOUR_CODE
    # 测试两个Counter相减
    from collections import Counter

    c1 = Counter({'a': 3, 'b': 2, 'c': 1})
    c2 = Counter({'a': 1, 'b': 2, 'd': 5})

    diff = c1 - c2  # 只保留正数部分
    print("c1 - c2:", diff)  # 应输出: Counter({'a': 2, 'c': 1})

    diff2 = c2 - c1
    print("c2 - c1:", diff2)  # 应输出: Counter({'d': 5})

    intersect = c1 & c2
    print("c1 & c2", intersect)  

    # 如果要看到所有项的差值（包括负数），可以这样：
    all_keys = set(c1.keys()) | set(c2.keys())
    full_diff = {k: c1[k] - c2[k] for k in all_keys}
    print("full diff (包含负数):", full_diff)
