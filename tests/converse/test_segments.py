import pytest

from legible.converse.segments import (
    Segment,
    SegmentParser,
    echo_text,
    parse_tagged,
    plain_text,
    tagged_text,
)

ANSWER = (
    "[excited] Great question! [content] Octopuses have three hearts. "
    "[surprised] Their blood is blue, because it uses copper!"
)
EXPECTED = [
    ("excited", "Great question!"),
    ("content", "Octopuses have three hearts."),
    ("surprised", "Their blood is blue, because it uses copper!"),
]


def pairs(segments: list[Segment]) -> list[tuple[str, str]]:
    return [(segment.emotion, segment.text) for segment in segments]


def stream(chunks: list[str]) -> list[Segment]:
    parser = SegmentParser()
    out: list[Segment] = []
    for chunk in chunks:
        out += parser.feed(chunk)
    return out + parser.finish()


def test_tagged_answer_becomes_one_segment_per_sentence():
    segments = parse_tagged(ANSWER)
    assert pairs(segments) == EXPECTED
    assert [segment.index for segment in segments] == [0, 1, 2]


@pytest.mark.parametrize("size", [1, 2, 3, 5, 8, 13])
def test_any_chunk_size_gives_the_same_segments(size):
    chunks = [ANSWER[i : i + size] for i in range(0, len(ANSWER), size)]
    assert pairs(stream(chunks)) == EXPECTED


def test_every_two_way_split_gives_the_same_segments():
    for cut in range(len(ANSWER) + 1):
        assert pairs(stream([ANSWER[:cut], ANSWER[cut:]])) == EXPECTED, cut


def test_segments_arrive_as_soon_as_they_are_complete():
    parser = SegmentParser()
    assert parser.feed("[excited] Great question! [con") == []  # the tag may still be "[content]" or text
    first = parser.feed("tent] Octopuses have")
    assert pairs(first) == [("excited", "Great question!")]
    assert parser.feed(" three hearts.") == []  # could still continue
    second = parser.feed(" Their")  # an untagged sentence starts, so the previous one is done
    assert pairs(second) == [("content", "Octopuses have three hearts.")]
    assert pairs(parser.finish()) == [("neutral", "Their")]


def test_missing_and_unknown_tags_are_neutral():
    segments = parse_tagged("Sure! [happy] Hello there. [EXCITED] Wow. [ sad ] Oh no.")
    assert pairs(segments) == [
        ("neutral", "Sure!"),
        ("neutral", "Hello there."),
        ("excited", "Wow."),
        ("sad", "Oh no."),
    ]


def test_untagged_sentences_are_split_and_neutral():
    segments = parse_tagged("[excited] Great question! Octopuses have three hearts. Isn't that wild?")
    assert pairs(segments) == [
        ("excited", "Great question!"),
        ("neutral", "Octopuses have three hearts."),
        ("neutral", "Isn't that wild?"),
    ]


def test_plain_text_without_any_tags():
    assert pairs(parse_tagged("No tags at all. Just plain sentences!")) == [
        ("neutral", "No tags at all."),
        ("neutral", "Just plain sentences!"),
    ]


def test_whitespace_markdown_and_list_markers_are_tidied():
    segments = parse_tagged("  \n[content]   Line one\n\nline **two**.  [neutral]\n- `First` point")
    assert pairs(segments) == [("content", "Line one line two."), ("neutral", "First point")]


def test_brackets_that_cannot_be_tags_stay_in_the_text():
    segments = parse_tagged("[excited] The answer [in short] is yes. [content] See note [1] here.")
    assert pairs(segments) == [("excited", "The answer [in short] is yes."), ("content", "See note [1] here.")]


def test_abbreviations_numbers_and_ellipses_do_not_split_sentences():
    segments = parse_tagged(
        "[content] Dr. Smith climbed Mt. Everest in the U.S. Army. Pi is about 3.14 and that is fun. "
        "Hmm... let me think."
    )
    assert pairs(segments) == [
        ("content", "Dr. Smith climbed Mt. Everest in the U.S. Army."),
        ("neutral", "Pi is about 3.14 and that is fun."),
        ("neutral", "Hmm... let me think."),
    ]


def test_double_tags_empty_tags_and_a_cut_off_final_tag():
    assert pairs(parse_tagged("[excited] [content] Calm now. [surprised]")) == [("content", "Calm now.")]
    assert pairs(stream(["[content] Ends here. [exci"])) == [("content", "Ends here.")]
    assert parse_tagged("") == [] and parse_tagged("[excited]   ") == [] and parse_tagged("[content] ...") == []


def test_unclosed_bracket_is_text_once_it_cannot_be_a_tag():
    assert pairs(parse_tagged("[content] Arrays look like [1, 2, 3 and so on.")) == [
        ("content", "Arrays look like [1, 2, 3 and so on.")
    ]


def test_echo_text_tags_every_segment_and_speaks_markup_symbols():
    segments = parse_tagged("[content] 5 < 7 & 9 > 2. [neutral] Done.")
    assert echo_text(segments) == (
        '<emotion value="content"/> 5 less than 7 and 9 greater than 2. <emotion value="neutral"/> Done.'
    )


def test_text_helpers_and_storage_round_trip():
    segments = parse_tagged(ANSWER)
    assert plain_text(segments) == " ".join(text for _, text in EXPECTED)
    assert pairs(parse_tagged(tagged_text(segments))) == EXPECTED
    assert [Segment.from_dict(segment.as_dict()) for segment in segments] == segments
    assert Segment.from_dict({"index": 0, "emotion": "bogus", "text": "Hi"}).emotion == "neutral"
