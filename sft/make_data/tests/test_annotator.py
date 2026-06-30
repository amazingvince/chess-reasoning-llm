import pytest


def test_batch_annotator_returns_empty_annotation_for_malformed_fen():
    from pool.annotator import BatchAnnotator

    class UnexpectedStockfish:
        def evaluate(self, _board, _depth=None):
            raise AssertionError("invalid FEN should not reach Stockfish")

    annotator = BatchAnnotator(stockfish=UnexpectedStockfish())

    assert annotator.annotate("not a fen") == {
        "cp": None,
        "mate": None,
        "best_move": None,
        "pv_line": "",
    }


def test_batch_annotator_returns_empty_annotation_for_parseable_invalid_fen():
    from pool.annotator import BatchAnnotator

    class UnexpectedStockfish:
        def evaluate(self, _board, _depth=None):
            raise AssertionError("invalid board should not reach Stockfish")

    annotator = BatchAnnotator(stockfish=UnexpectedStockfish())

    assert annotator.annotate("8/8/8/8/8/8/8/8 w - - 0 1") == {
        "cp": None,
        "mate": None,
        "best_move": None,
        "pv_line": "",
    }
