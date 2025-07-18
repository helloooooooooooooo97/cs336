from multiprocessing.context import Process
import os
import regex as re
from typing import  BinaryIO
from multiprocessing import Process, Queue
from typing import TypeAlias

Pair: TypeAlias = tuple[int, int]  # 例如: (65, 66) 表示两个token id的pair
PairCountDict: TypeAlias = dict[Pair, int]  # 例如: {(65, 66): 10, (66, 67): 5} 表示pair出现的次数
Pretoken: TypeAlias = list[bytes]  # 例如: [b'hello', b' world'] 表示一个tokens bytes序列
Pretokens: TypeAlias = list[Pretoken]  # 例如: [b'hello', b' world'] 表示一个tokens bytes序列
Vocab: TypeAlias = dict[int, bytes]  # 例如: {65: b'a', 66: b'b'} 表示token id到字节的映射
Merges: TypeAlias = list[tuple[bytes, bytes]]  # 例如: [(b'a', b'b'), (b'b', b'c')] 表示BPE合并规则

class BPETokenizerBatcher:
    BYTE_VOCAB_SIZE = 256  # 字节级初始词表大小
    CHUNK_READ_SIZE = 4096  # 文件读取时的块大小
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""  # 分词正则

    def __init__(self, special_tokens: list[str], num_processes: int = 4):
        self.special_tokens = special_tokens
        self.num_processes = num_processes

    def load_chunks(self, input_path: str | os.PathLike) -> list[str]:
        """分块加载文本文件。"""
        chunks = []
        with open(input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunks.append(chunk)
        return chunks

    def pretokenize_chunks(self, chunks: list[str]) -> Pretoken:
        """对所有文本块进行预分词（多进程）。"""
        processes: list[Process] = []
        q: Queue = Queue()
        for chunk in chunks:
            p: Process = Process(target=BPETokenizerBatcher.worker, args=(chunk, self.special_tokens, q))
            p.start()
            processes.append(p)
        all_pretokens = [q.get() for _ in processes]
        for p in processes:
            p.join()
        return [token for tokens in all_pretokens for token in tokens]

    @staticmethod
    def split_by_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
        """按特殊token对字符串进行切分，保留特殊token。"""
        special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
        if not special_tokens_sorted:
            return [text]
        pattern = "|".join(re.escape(tok) for tok in special_tokens_sorted)
        return re.split('(' + pattern + ')', text)

    @staticmethod
    def pretokenize(text: str, special_tokens: list[str], drop_special_token: bool = True) -> list[bytes]:
        """对文本进行预分词，返回字节级token序列。"""
        parts = BPETokenizerBatcher.split_by_special_tokens(text, special_tokens)
        tokens_list = []
        for part in parts:
            if part in special_tokens:
                if not drop_special_token:
                    tokens_list.append([part.encode('utf-8')])
            else:
                str_tokens = re.findall(BPETokenizerBatcher.PAT, part)
                part_tokens = [s.encode('utf-8') for s in str_tokens]
                tokens_list.append(part_tokens)
        return [token for part_tokens in tokens_list for token in part_tokens]

    @staticmethod
    def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
        """
        均匀分块，保证特殊token不被拆分。
        """
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        chunk_size = file_size // desired_num_chunks
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)
            while True:
                mini_chunk = file.read(BPETokenizerBatcher.CHUNK_READ_SIZE)
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += BPETokenizerBatcher.CHUNK_READ_SIZE
        return sorted(set(chunk_boundaries))

    @staticmethod
    def worker(text: str, special_tokens: list[str], q: Queue):
        """多进程辅助函数，对文本块进行预分词并放入队列。"""
        pretokens: list[bytes] = BPETokenizerBatcher.pretokenize(text, special_tokens)
        q.put(pretokens)
