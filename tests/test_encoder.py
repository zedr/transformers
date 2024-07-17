from encoder import OneHotEncoder, OneHotEncoding


def test_enconding_1():
    enc = list(OneHotEncoding(5, {0: "find"}))
    assert enc == [1, 0, 0, 0, 0]


def test_encoder_api():
    words = ["files", "find", "my"]
    enc = OneHotEncoder(words)
    for word in words:
        idx = enc.get_index(word)
        assert word == enc.get_word(idx)


def test_encoder_encoding():
    words = ["files", "find", "my", "mind"]
    enc = OneHotEncoder(words)
    encoding = enc.encode("find", "my", "files")
    encoding = [list(item) for item in encoding]
    assert encoding == [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0]
    ]


def tut_test_1():
    enc = OneHotEncoder(["files", "find", "my"])
    assert [1, 0, 0] == enc.encode("find", "my", "files")
