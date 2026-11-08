from preprocess import clean_text


def test_removes_br_tags():
    assert clean_text("Great movie.<br /><br />Loved it.") == "great movie. loved it."


def test_removes_other_html_tags():
    assert clean_text("<i>Amazing</i> film") == "amazing film"


def test_decodes_html_entities():
    assert clean_text("Tom &amp; Jerry") == "tom & jerry"


def test_lowercases_and_collapses_whitespace():
    assert clean_text("  GOOD    Movie \n ") == "good movie"


def test_keeps_negation_words():
    assert "not" in clean_text("This was NOT good")