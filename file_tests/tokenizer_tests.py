from tokenization import SimpleTokenizer


def test_word_tokenizer_basic_encode_decode():
    vocab = ["the", "cat", "cats", "sleeps", "sleep"]
    tok = SimpleTokenizer(vocab=vocab, mode="word")

    text = "the cat sleeps"
    ids = tok.encode(text)
    decoded = tok.decode(ids)

    assert decoded == text
    assert len(ids) == 3


def test_word_tokenizer_unknown_token():
    vocab = ["the", "cat", "sleeps"]
    tok = SimpleTokenizer(vocab=vocab, mode="word")

    text = "the dog sleeps"  # "dog" not in vocab
    ids = tok.encode(text)

    unk_id = tok.stoi[tok.unk_token]
    assert ids[1] == unk_id
    assert tok.decode(ids) == "the <unk> sleeps"


def test_word_tokenizer_eos():
    vocab = ["the", "cat", "sleeps"]
    tok = SimpleTokenizer(vocab=vocab, mode="word")

    ids = tok.encode("the cat sleeps", add_eos=True)

    assert ids[-1] == tok.stoi[tok.eos_token]
    assert tok.decode(ids) == "the cat sleeps <eos>"


def test_vocab_contains_special_tokens():
    vocab = ["the", "cat"]
    tok = SimpleTokenizer(vocab=vocab, mode="word")

    assert tok.unk_token in tok.vocab
    assert tok.eos_token in tok.vocab
    assert tok.vocab_size == len(tok.vocab)


def test_char_tokenizer_roundtrip():
    vocab = list("abc ")  # include space explicitly
    tok = SimpleTokenizer(vocab=vocab, mode="char")

    text = "a b"
    ids = tok.encode(text)
    decoded = tok.decode(ids)

    assert decoded == text


if __name__ == "__main__":
    test_word_tokenizer_basic_encode_decode()
    test_word_tokenizer_unknown_token()
    test_word_tokenizer_eos()
    test_vocab_contains_special_tokens()
    test_char_tokenizer_roundtrip()

    print("All tokenizer tests passed.")