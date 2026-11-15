from src.reconstruction import reconstruct_text


def test_reconstruct_text_joins_characters_words_and_lines():
    recognized_lines = [
        [
            {"text": "J", "space_before": False},
            {"text": "e", "space_before": False},
            {"text": "w", "space_before": False},
            {"text": "e", "space_before": False},
            {"text": "l", "space_before": False},
            {"text": "C", "space_before": True},
            {"text": "i", "space_before": False},
            {"text": "t", "space_before": False},
            {"text": "y", "space_before": False},
        ],
        [
            {"text": "O", "space_before": False},
            {"text": "K", "space_before": False},
        ],
    ]

    assert reconstruct_text(recognized_lines) == "Jewel City\nOK"


def test_reconstruct_text_returns_empty_string_for_no_lines():
    assert reconstruct_text([]) == ""


def test_reconstruct_text_does_not_add_leading_space():
    recognized_lines = [[{"text": "A", "space_before": True}]]

    assert reconstruct_text(recognized_lines) == "A"
