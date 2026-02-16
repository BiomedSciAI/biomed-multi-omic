"""Tests for MultiFieldInstance class."""

from bmfm_targets.tokenization import MultiFieldInstance


def test_can_instantiate():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={"genes": "token1", "expressions": "token2"},
    )
    assert mfi["genes"] == "token1"
    assert mfi["expressions"] == "token2"


def test_can_instantiate_from_str_lists():
    mfi = MultiFieldInstance(
        metadata={"cell_name": "test"},
        data={"genes": ["token1", "token2"], "expressions": ["token3", "token4"]},
    )
    assert mfi["genes"] == ["token1", "token2"]
    assert mfi["expressions"] == ["token3", "token4"]


def test_seq_length():
    """Test seq_length property."""
    assert MultiFieldInstance(data={"genes": ["token1"]}).seq_length == 1
    assert MultiFieldInstance(data={"genes": ["t1", "t2", "t3"]}).seq_length == 3
    assert MultiFieldInstance(data={"genes": []}).seq_length == 0
