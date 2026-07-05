import pytest


class FakeTokenizer:
    def __init__(self):
        self.additional_special_tokens = ["<think>", "</think>"]
        self.calls = []

    def add_special_tokens(self, special_tokens_dict, **kwargs):
        self.calls.append((special_tokens_dict, kwargs))
        self.additional_special_tokens.extend(
            special_tokens_dict.get("additional_special_tokens", [])
        )
        return len(special_tokens_dict.get("additional_special_tokens", []))


class FakeLegacyTokenizer:
    def __init__(self):
        self.additional_special_tokens = ["<think>", "</think>"]
        self.calls = []
        self.ids = {"<think>": 10, "</think>": 11}
        self.unk_token_id = None

    def add_special_tokens(self, special_tokens_dict):
        self.calls.append(special_tokens_dict)
        added = 0
        for token in special_tokens_dict.get("additional_special_tokens", []):
            if token not in self.ids:
                self.ids[token] = 100 + added
                added += 1
        self.additional_special_tokens = special_tokens_dict.get(
            "additional_special_tokens", []
        )
        return added

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token)


def test_add_move_special_tokens_preserves_existing_additional_tokens():
    from chess_llm.training.tokenizer import MOVE_SPECIAL_TOKENS, add_move_special_tokens

    tokenizer = FakeTokenizer()

    added = add_move_special_tokens(tokenizer)

    assert added == 2
    assert tokenizer.calls == [
        (
            {"additional_special_tokens": list(MOVE_SPECIAL_TOKENS)},
            {"replace_additional_special_tokens": False},
        )
    ]


def test_add_move_special_tokens_falls_back_when_replace_flag_is_unsupported():
    from chess_llm.training.tokenizer import MOVE_SPECIAL_TOKENS, add_move_special_tokens

    tokenizer = FakeLegacyTokenizer()

    added = add_move_special_tokens(tokenizer)

    assert added == 2
    assert tokenizer.calls == [
        {
            "additional_special_tokens": [
                "<think>",
                "</think>",
                *MOVE_SPECIAL_TOKENS,
            ]
        }
    ]


def test_initialize_added_move_token_embeddings_uses_existing_fragments():
    torch = pytest.importorskip("torch")
    from chess_llm.training.tokenizer import initialize_added_move_token_embeddings

    class TokenizerWithIds:
        def __init__(self):
            self.ids = {
                "<": 0,
                "move": 1,
                ">": 2,
                "</": 3,
                "<move>": 6,
                "</move>": 7,
            }

        def convert_tokens_to_ids(self, token):
            return self.ids[token]

    class FakeModel:
        def __init__(self):
            self.input_embeddings = torch.nn.Embedding(8, 2)
            self.output_embeddings = torch.nn.Embedding(8, 2)
            with torch.no_grad():
                values = torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 2.0],
                        [3.0, 0.0],
                        [0.0, 4.0],
                        [9.0, 9.0],
                        [8.0, 8.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                )
                self.input_embeddings.weight.copy_(values)
                self.output_embeddings.weight.copy_(values * 2)

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    model = FakeModel()

    initialize_added_move_token_embeddings(
        model,
        TokenizerWithIds(),
        original_vocab_size=6,
    )

    expected_start = torch.tensor([4.0 / 3.0, 2.0 / 3.0])
    expected_end = torch.tensor([1.0, 2.0])
    assert torch.allclose(model.input_embeddings.weight[6], expected_start)
    assert torch.allclose(model.input_embeddings.weight[7], expected_end)
    assert torch.allclose(model.output_embeddings.weight[6], expected_start * 2)
    assert torch.allclose(model.output_embeddings.weight[7], expected_end * 2)


def test_resize_move_tokens_does_not_shrink_larger_embedding_matrix():
    torch = pytest.importorskip("torch")
    from chess_llm.training.tokenizer import resize_and_initialize_move_special_tokens

    class TokenizerWithIds:
        def __init__(self):
            self.ids = {
                "<": 0,
                "move": 1,
                ">": 2,
                "</": 3,
                "<move>": 6,
                "</move>": 7,
            }

        def __len__(self):
            return 8

        def convert_tokens_to_ids(self, token):
            return self.ids[token]

    class FakeModel:
        def __init__(self):
            self.input_embeddings = torch.nn.Embedding(10, 2)
            self.resized_to = None

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return None

        def resize_token_embeddings(self, size):
            self.resized_to = size
            self.input_embeddings = torch.nn.Embedding(size, 2)

    model = FakeModel()

    resize_and_initialize_move_special_tokens(
        model,
        TokenizerWithIds(),
        original_vocab_size=6,
        added_token_count=2,
    )

    assert model.resized_to is None
    assert model.input_embeddings.weight.shape[0] == 10
