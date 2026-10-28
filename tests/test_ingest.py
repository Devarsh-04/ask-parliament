from ingest import chunk_speech, parse_day, parse_members

DAY = b"""<?xml version="1.0" encoding="UTF-8"?>
<debates>
 <major-heading id="d.1" url="http://x/1">
HOUSING </major-heading>
 <minor-heading id="d.2" url="http://x/2">
Supply Bill; Second Reading </minor-heading>
 <speech id="d.3" speakerid="uk.org.publicwhip/member/815" speakername="Milton Dick" talktype="speech" time="12:00" url="http://x/3">
<p>First <i>para</i>.</p><p>Second para.</p>
 </speech>
 <speech id="d.4" speakerid="unknown" speakername="Honourable members" talktype="interjection" time="12:01" url="http://x/4">
<p>Hear, hear!</p>
 </speech>
 <division id="d.5"><p>ignored</p></division>
 <major-heading id="d.6" url="http://x/6">QUESTIONS WITHOUT NOTICE</major-heading>
 <speech id="d.7" speakerid="uk.org.publicwhip/lord/100974" speakername="Vanessa Bleyer" talktype="speech" time="14:00" url="http://x/7">
<p>A question.</p>
 </speech>
</debates>"""

MEMBERS = b"""<members>
 <member id="uk.org.publicwhip/member/815" firstname="Milton" lastname="Dick" party="Australian Labor Party" fromdate="2016-07-02" todate="9999-12-31"/>
 <member id="uk.org.publicwhip/lord/100974" firstname="Vanessa" lastname="Bleyer" party="Australian Greens" fromdate="2026-08-20" todate="9999-12-31"/>
</members>"""


def test_parse_day_keeps_speeches_with_headings():
    rows = parse_day(DAY, "representatives", "2025-02-04")
    assert [r["id"] for r in rows] == ["d.3", "d.7"]
    first = rows[0]
    assert first["text"] == "First para.\n\nSecond para."
    assert (first["major"], first["minor"]) == ("HOUSING", "Supply Bill; Second Reading")
    assert (first["speaker"], first["speaker_id"], first["house"], first["date"]) == (
        "Milton Dick", "uk.org.publicwhip/member/815", "representatives", "2025-02-04")
    assert first["url"] == "http://x/3"
    assert (rows[1]["major"], rows[1]["minor"]) == ("QUESTIONS WITHOUT NOTICE", None)


def test_parse_day_merges_continuation_into_speech():
    xml = b"""<debates>
     <speech id="a" speakerid="m/1" speakername="A" talktype="speech" url="u"><p>Start.</p></speech>
     <speech id="b" speakerid="m/2" speakername="Speaker" talktype="interjection" url="u"><p>Order!</p></speech>
     <speech id="c" speakerid="m/1" speakername="A" talktype="continuation" url="u"><p>Finish.</p></speech>
    </debates>"""
    rows = parse_day(xml, "senate", "2025-01-01")
    assert [(r["id"], r["text"]) for r in rows] == [("a", "Start.\n\nFinish.")]


def test_parse_members_maps_id_to_party():
    assert parse_members(MEMBERS) == {
        "uk.org.publicwhip/member/815": "Australian Labor Party",
        "uk.org.publicwhip/lord/100974": "Australian Greens",
    }


def test_chunk_speech_splits_on_paragraphs():
    paras = [" ".join(["w"] * 120)] * 5
    chunks = chunk_speech("\n\n".join(paras), max_words=300)
    assert [len(c.split()) for c in chunks] == [240, 240, 120]


def test_chunk_speech_splits_giant_paragraph():
    chunks = chunk_speech(" ".join(["w"] * 650), max_words=300)
    assert [len(c.split()) for c in chunks] == [300, 300, 50]
