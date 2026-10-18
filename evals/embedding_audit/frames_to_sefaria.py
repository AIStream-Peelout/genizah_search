"""Map KTIV scholarly frames to Sefaria refs (for canonical-text enrichment and training pairs).

Handles the identified works that Sefaria carries with compatible addressing: Talmud Bavli (daf/amud),
Mishnah (chapter:mishnah), Bible (chapter:verse) and Mishneh Torah (hilkhot chapter:halakhah). Rif uses its own
pagination and the Yerushalmi frames are irregular, so both are left out. Writes ``frame_refs.jsonl``
({doc_id, frame, ref}) and ``frame_refs.txt`` (unique refs, for ``sefaria_pull.py texts``).
"""

import json
import re
from collections import Counter

from embed_utils import AUDIT_ROOT

BAVLI_NAMES = {
    "Bava Metzi'a": "Bava Metzia", "Bava Qamma": "Bava Kamma", "Rosh ha-Shanah": "Rosh Hashanah", "Qiddushin": "Kiddushin",
    "Hullin": "Chullin", "Ketubbot": "Ketubot", "Pesahim": "Pesachim", "Betzah": "Beitzah", "Ta'anit": "Taanit",
    "Mo'ed Qatan": "Moed Katan", "Hagigah": "Chagigah", "Menahot": "Menachot", "Zevahim": "Zevachim",
    "Me'ilah": "Meilah", "Bekhorot": "Bekhorot", "Arakhin": "Arakhin", "Keritot": "Keritot", "Shevu'ot": "Shevuot",
    "Avodah Zarah": "Avodah Zarah", "Pe'ah": "Peah", "Ma'aserot": "Maasrot", "Ma'aser Sheni": "Maaser Sheni",
    "Kil'ayim": "Kilayim", "Shevi'it": "Sheviit", "Orlah": "Orlah", "Bikkurim": "Bikkurim", "Terumot": "Terumot",
    "Demai": "Demai", "Hallah": "Challah", "Eduyot": "Eduyot", "Ohalot": "Oholot", "Kelim": "Kelim",
    "Negaim": "Negaim", "Parah": "Parah", "Tohorot": "Tahorot", "Mikva'ot": "Mikvaot", "Makhshirin": "Makhshirin",
    "Zavim": "Zavim", "Tevul Yom": "Tevul Yom", "Yadayim": "Yadayim", "Uqtzin": "Oktzin", "Sheqalim": "Shekalim",
}
MT_NAMES = {
    "Shabbat": "Sabbath", "Tefillah ve-Birkat Kohanim": "Prayer and the Priestly Blessing", "Shehitah": "Ritual Slaughter",
    "Qeri'at Shema": "Reading the Shema", "Tefillin Mezuzah ve-Sefer Torah": "Tefillin, Mezuzah and the Torah Scroll",
    "Berakhot": "Blessings", "Eruvin": "Eruvin", "Qiddush ha-Hodesh": "Sanctification of the New Month",
    "Hametz u-Matztzah": "Leavened and Unleavened Bread", "Shofar ve-Sukkah ve-Lulav": "Shofar, Sukkah and Lulav",
    "Ishut": "Marriage", "Gerushin": "Divorce", "Teshuvah": "Repentance", "Yesodei ha-Torah": "Foundations of the Torah",
    "Talmud torah": "Torah Study", "Talmud Torah": "Torah Study", "De'ot": "Human Dispositions",
    "Avodah Zarah": "Foreign Worship and Customs of the Nations", "Terumot": "Heave Offerings",
    "Ma'akhalot Asurot": "Forbidden Foods", "Issurei Bi'ah": "Forbidden Intercourse",
    "Shevitat Asor": "Rest on the Tenth of Tishrei", "Shevitat Yom Tov": "Rest on a Holiday",
    "Megillah va-Hanukkah": "Scroll of Esther and Hanukkah", "Ta'aniyot": "Fasts", "Nedarim": "Vows", "Shevu'ot": "Oaths",
    "Mamrim": "Rebels", "Edut": "Testimony", "Malveh ve-Loveh": "Creditor and Debtor", "Mekhirah": "Sales",
    "To'en ve-Nit'an": "Plaintiff and Defendant", "Nahalot": "Inheritances", "Shekhenim": "Neighbors",
    "Sheluhin ve-Shutafin": "Agents and Partners", "Avadim": "Slaves", "Gezelah va-Avedah": "Robbery and Lost Property",
    "Nizkei Mamon": "Damages to Property", "Mikva'ot": "Immersion Pools", "Tum'at Met": "Defilement by a Corpse",
    "Ma'aser": "Tithes", "Kil'ayim": "Diverse Species", "Shemittah ve-Yovel": "Sabbatical Year and the Jubilee",
    "Matanot Aniyim": "Gifts to the Poor", "Evel": "Mourning", "Melakhim u-Milhamot": "Kings and Wars",
    "Beit ha-Behirah": "The Chosen Temple", "Avodat Yom ha-Kippurim": "Service on the Day of Atonement",
    "Ma'aseh ha-Qorbanot": "Sacrificial Procedure", "Temidin u-Musafin": "Daily Offerings and Additional Offerings",
    "Qorban Pesah": "Paschal Offering", "Hagigah": "Festival Offering", "Bekhorot": "Firstlings",
    "Mehusarei Kapparah": "Those with Incomplete Atonement", "Temurah": "Substitution", "Nezirut": "Nazariteship",
    "Tzitzit": "Fringes", "Milah": "Circumcision", "Sanhedrin": "The Sanhedrin and the Penalties within their Jurisdiction",
    "Bikkurim": "First Fruits and other Gifts to Priests Outside the Sanctuary", "Ma'aser Sheni ve-Neta Revai":
    "Second Tithes and Fourth Year's Fruit", "Sekhirut": "Hiring", "She'elah u-Fiqadon": "Borrowing and Deposit",
    "Zekhiyah u-Mattanah": "Ownerless Property and Gifts", "Genevah": "Theft", "Hovel u-Maziq": "One Who Injures a Person or Property",
    "Rotze'ah u-Shmirat Nefesh": "Murderer and the Preservation of Life", "Na'arah Betulah": "Virgin Maiden",
    "Yibbum va-Halitzah": "Levirate Marriage and Release", "Sotah": "Woman Suspected of Infidelity",
    "Kelei ha-Miqdash": "Vessels of the Sanctuary and Those who Serve Therein", "Pesulei ha-Muqdashin": "Disqualified Offerings",
    "Shegagot": "Offerings for Unintentional Transgressions", "Arakhin va-Haramin": "Appraisals and Devoted Property",
}
DASH = r"\s*[–-]\s*"


def clean(name: str) -> str:
    """Strip catalogue noise and apostrophes.

    :param name: Raw work/section name.
    :returns: Cleaned name.
    :rtype: str
    """
    return name.strip(" ,;:")


def map_frame(frame: str):
    """Map one frame to a Sefaria ref, or None.

    :param frame: KTIV frame string.
    :returns: Sefaria ref or None.
    """
    frame = re.sub(r"\[[^\]]*$|\[(?!Talmud|Mishnah|Bible|Moses)[^\]]*\]", "", frame).strip()
    m = re.match(r"\[Talmud Bavli\]:\s*([A-Za-z' \-]+?)\s+(\d+)\s*([ab])(?:" + DASH + r"(?:(\d+)\s*)?([ab]))?\s*$", frame)
    if m:
        t, f0, s0, f1, s1 = m.groups()
        t = BAVLI_NAMES.get(clean(t), clean(t))
        start = f"{f0}{s0}"
        end = f"{f1 or f0}{s1}" if s1 else None
        return f"{t} {start}-{end}" if end and end != start else f"{t} {start}"
    m = re.match(r"\[Mishnah\]:\s*([A-Za-z' \-]+?)\s+(\d+(?::\d+)?)(?:" + DASH + r"(\d+(?::\d+)?))?\s*$", frame)
    if m:
        t, a, b = m.groups()
        t = BAVLI_NAMES.get(clean(t), clean(t)).replace("'", "")
        return f"Mishnah {t} {a}" + (f"-{b}" if b else "")
    m = re.match(r"\[Bible\]:\s*((?:I{1,2} )?[A-Za-z ]+?)\s+(\d+(?::\d+)?)(?:" + DASH + r"(\d+(?::\d+)?))?\s*$", frame)
    if m:
        book, a, b = m.groups()
        return f"{book.strip()} {a}" + (f"-{b}" if b else "")
    m = re.match(r"\[Moses b\. Maimon, Rambam, Mishneh Torah\]:\s*([A-Za-z' \-]+?)\s+(\d+(?::\d+)?)(?:" + DASH + r"(\d+(?::\d+)?))?\s*$", frame)
    if m:
        sec, a, b = m.groups()
        name = MT_NAMES.get(clean(sec))
        if name:
            return f"Mishneh Torah, {name} {a}" + (f"-{b}" if b else "")
    return None


def main() -> None:
    """Map all framed records."""
    stats, refs = Counter(), Counter()
    out = open(AUDIT_ROOT / "frame_refs.jsonl", "w", encoding="utf-8")
    for line in open(AUDIT_ROOT / "corpus_v1.jsonl", encoding="utf-8"):
        row = json.loads(line)
        for fr in row.get("frames") or []:
            work = (re.match(r"\[([^\]]+)\]", fr) or [None, "other"])[1]
            ref = map_frame(fr)
            stats[f"{work[:40]}:{'ok' if ref else 'no'}"] += 1
            if ref:
                refs[ref] += 1
                out.write(json.dumps({"doc_id": row["doc_id"], "frame": fr, "ref": ref}, ensure_ascii=False) + "\n")
    out.close()
    (AUDIT_ROOT / "frame_refs.txt").write_text("\n".join(sorted(refs)) + "\n")
    print(json.dumps({"unique_refs": len(refs), **{k: v for k, v in stats.most_common(14)}}, indent=1))


if __name__ == "__main__":
    main()
