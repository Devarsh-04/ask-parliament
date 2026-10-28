from pathlib import Path

import pandas as pd
from lxml import etree

DATA = Path("data")
HOUSES = {"representatives": "representatives_debates", "senate": "senate_debates"}


def _clean(text: str) -> str:
    return " ".join(text.split())


def parse_day(xml: bytes, house: str, date: str) -> list[dict]:
    rows, major, minor = [], None, None
    for el in etree.fromstring(xml):
        if el.tag == "major-heading":
            major, minor = _clean(el.xpath("string()")), None
        elif el.tag == "minor-heading":
            minor = _clean(el.xpath("string()"))
        elif el.tag == "speech" and el.get("talktype") != "interjection" and el.get("speakerid") != "unknown":
            text = "\n\n".join(_clean(p.xpath("string()")) for p in el.iter("p"))
            if not text:
                continue
            if el.get("talktype") == "continuation" and rows and rows[-1]["speaker_id"] == el.get("speakerid"):
                rows[-1]["text"] += "\n\n" + text
            else:
                rows.append({
                    "id": el.get("id"), "house": house, "date": date, "time": el.get("time"),
                    "major": major, "minor": minor, "speaker": el.get("speakername"),
                    "speaker_id": el.get("speakerid"), "url": el.get("url"), "text": text,
                })
    return rows


def parse_members(xml: bytes) -> dict[str, str]:
    return {m.get("id"): m.get("party") for m in etree.fromstring(xml).iter("member")}


def chunk_speech(text: str, max_words: int = 300) -> list[str]:
    pieces = []
    for para in text.split("\n\n"):
        words = para.split()
        pieces += [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    chunks, current = [], []
    for piece in pieces:
        if current and sum(len(p.split()) for p in current) + len(piece.split()) > max_words:
            chunks.append("\n\n".join(current))
            current = []
        current.append(piece)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def main() -> None:
    party = {}
    for f in ("representatives.xml", "senators.xml"):
        party |= parse_members((DATA / f).read_bytes())
    speeches = []
    for house, folder in HOUSES.items():
        for f in sorted((DATA / "raw" / folder).glob("*.xml")):
            speeches += parse_day(f.read_bytes(), house, f.stem[:10])
    speeches = pd.DataFrame(speeches)
    speeches["party"] = speeches.speaker_id.map(party)
    chunks = speeches.assign(text=speeches.text.map(chunk_speech)).explode("text", ignore_index=True)
    chunks.insert(0, "chunk_id", chunks.id + "#" + chunks.groupby("id").cumcount().astype(str))
    speeches.to_parquet(DATA / "speeches.parquet", index=False)
    chunks.to_parquet(DATA / "chunks.parquet", index=False)
    print(f"{len(speeches)} speeches, {len(chunks)} chunks, {chunks.text.str.split().str.len().sum():,} words")


if __name__ == "__main__":
    main()
