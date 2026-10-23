from prompt_scoring import compress_prompt, split_words


def test_split_words_roundtrip():
    text = "Fix bug: `get_user(id=42)` must NOT return None."
    words = split_words(text)
    assert all(text[start:end] == word for word, start, end in words)
    assert any(word == "`get_user(id=42)`" for word, _, _ in words)


def test_compress_preserves_whitespace():
    result = {"words": split_words("foo bar baz"), "scores": [0.9, 0.1, 0.5]}
    assert compress_prompt("foo bar baz", result, 0.5) == "foo baz"


def test_protected_negation_survives():
    text = "must not delete this"
    result = {"words": split_words(text), "scores": [0.1, 0.1, 0.1, 0.9]}
    assert "must" in compress_prompt(text, result, 0.25)
    assert "not" in compress_prompt(text, result, 0.25)
