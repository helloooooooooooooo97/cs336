
# 以中文进行回复
from cs336_basics.tokenizer import Tokenizer

import io
from typing import Dict, List, Tuple, Any, Set

# 这个函数 `get_simple_tokenizer` 的作用是构造一个简单的Tokenizer对象，便于单元测试。
# 下面详细解释每一部分：

def get_simple_tokenizer() -> Tokenizer:
    # vocab 是一个字典，key是token的id（整数），value是对应的字节（bytes），比如0对应b"H"，1对应b"e"等。
    vocab: Dict[int, bytes] = {
        0: b"H",
        1: b"e",
        2: b"l",
        3: b"o",
        4: b" ",
        5: b"W",
        6: b"r",
        7: b"d",
        8: b"!",
    }
    # merges 是一个BPE（Byte Pair Encoding）合并规则的列表，每个元素是一个二元组，表示两个byte token可以合并成一个更长的token。
    merges: List[Tuple[bytes, bytes]] = [
        (b"H", b"e"),   # 比如 b"H" 和 b"e" 可以合并
        (b"l", b"l"),
        (b"l", b"o"),
        (b"W", b"o"),
        (b"r", b"l"),
        (b"o", b"r"),
        (b"r", b"d"),
    ]
    # special_tokens 是特殊token的字符串列表，比如 "<|endoftext|>"，用于分隔或标记特殊含义的内容。
    special_tokens: List[str] = ["<|endoftext|>"]

    return Tokenizer(vocab, merges, special_tokens)


# 对tokenizer的每一个子函数做测试说明其作用
def test_bpe_encode_explain():
    """
    测试 _bpe_encode(chunk: bytes) 的作用：
    输入一个字节串chunk，返回BPE编码后的token id列表。
    例如输入 b"Hello"，应输出每个token的id，合并后能还原原始字节。
    """
    tokenizer = get_simple_tokenizer()
    chunk = b"Hello"
    ids: List[int] = tokenizer._bpe_encode(chunk)
    print("测试_bpe_encode: 输入", chunk, "输出token id:", ids)
    # 检查输出类型
    assert isinstance(ids, list)
    # 检查能否还原
    decoded = tokenizer.decode(ids)
    assert decoded == "Hello"


def test_get_pairs_explain():
    """
    测试 _get_pairs(word: List[bytes]) 的作用：
    输入一个字节token列表，返回所有相邻token对的集合。
    例如输入 [b"H", b"e", b"l", b"l", b"o"]，应输出 {(b"H", b"e"), (b"e", b"l"), (b"l", b"l"), (b"l", b"o")}
    """
    tokenizer = get_simple_tokenizer()
    word = [b"H", b"e", b"l", b"l", b"o"]
    pairs = tokenizer._get_pairs(word)
    print("测试_get_pairs: 输入", word, "输出pairs:", pairs)
    # 检查输出类型
    assert isinstance(pairs, set)
    # 检查内容
    assert (b"H", b"e") in pairs
    assert (b"e", b"l") in pairs
    assert (b"l", b"l") in pairs
    assert (b"l", b"o") in pairs

def test_encode_explain():
    """
    测试 encode(text: str) 的作用：
    输入字符串，优先匹配特殊token，然后对剩余部分用BPE编码，输出token id列表。
    例如输入 "Hello<|endoftext|>!"，应能识别特殊token并正确编码。
    """
    tokenizer = get_simple_tokenizer()
    text = "Hello<|endoftext|>!"
    ids = tokenizer.encode(text)
    print("测试encode: 输入", text, "输出token id:", ids)
    # 检查特殊token id是否在结果中
    special_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    assert special_token_id in ids
    # 检查能否还原
    decoded = tokenizer.decode(ids)
    assert decoded == text

def test_decode_explain():
    """
    测试 decode(ids: List[int]) 的作用：
    输入token id列表，还原为原始字符串。
    例如输入 [0, 1, 2, 2, 3, special_token_id, 8]，应输出 "Hello<|endoftext|>!"
    """
    tokenizer = get_simple_tokenizer()
    special_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    ids = [0, 1, 2, 2, 3, special_token_id, 8]
    decoded = tokenizer.decode(ids)
    print("测试decode: 输入", ids, "输出字符串:", decoded)
    assert decoded == "Hello<|endoftext|>!"

def test_encode_iterable_explain():
    """
    测试 encode_iterable(iterable: IO) 的作用：
    输入一个可迭代的文本流（如文件），逐行编码并yield token id。
    """
    tokenizer: Tokenizer = get_simple_tokenizer()
    # 构造一个模拟文件流
    lines = ["Hello\n", "World!\n"]
    fake_file = io.StringIO("".join(lines))
    # encode_iterable 需要二进制流或文本流，这里用文本流
    tokens = list(tokenizer.encode_iterable(fake_file))
    print("测试encode_iterable: 输入多行文本，输出token id序列:", tokens)
    # 检查类型
    assert all(isinstance(t, int) for t in tokens)
    # 检查能否还原
    decoded = tokenizer.decode(tokens)
    assert decoded == "Hello\nWorld!\n"

def test_encode_basic() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    text: str = "Hello World!"
    ids: List[int] = tokenizer.encode(text)
    print("编码结果:", ids)
    assert isinstance(ids, list)
    # 检查是否能还原
    decoded: str = tokenizer.decode(ids)
    assert decoded == text

def test_decode_basic() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    # 直接用token id还原
    ids: List[int] = [0, 1, 2, 2, 3, 4, 5, 3, 6, 2, 7, 8]
    decoded: str = tokenizer.decode(ids)
    print("解码结果:", decoded)
    assert isinstance(decoded, str)
    assert decoded == "Hello World!"

def test_encode_with_special_token() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    text: str = "Hello<|endoftext|>!"
    ids: List[int] = tokenizer.encode(text)
    print("包含特殊token编码结果:", ids)
    # 检查特殊token id是否在结果中
    special_token_id: Any = tokenizer.token_to_id.get(b"<|endoftext|>")
    assert special_token_id in ids
    decoded: str = tokenizer.decode(ids)
    print("包含特殊token解码结果:", decoded)
    assert decoded == text

def test_bpe_encode_internal() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    # 直接测试_bpe_encode内部逻辑
    chunk: bytes = b"Hello"
    ids: List[int] = tokenizer._bpe_encode(chunk)
    print("BPE编码结果:", ids)
    # 检查返回的id是否都在vocab中
    for i in ids:
        assert i in tokenizer.id_to_token

def test_get_pairs_internal() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    word: List[bytes] = [b"H", b"e", b"l", b"l", b"o"]
    pairs: Set[Tuple[bytes, bytes]] = tokenizer._get_pairs(word)
    print("pair集合:", pairs)
    # 检查pair类型
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in pairs)

def test_encode_iterable() -> None:
    tokenizer: Tokenizer = get_simple_tokenizer()
    # 构造一个文本流
    text: str = "Hello\nWorld!\n"
    f: io.StringIO = io.StringIO(text)
    ids: List[int] = list(tokenizer.encode_iterable(f))
    print("encode_iterable结果:", ids)
    # 检查decode后能否还原
    decoded: str = tokenizer.decode(ids)
    assert decoded == text.replace("\n", "") + "\n"  # 因为每行rstrip("\n")，但encode("\n")补回
