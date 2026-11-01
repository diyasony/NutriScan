"""
NutriScan pipeline: OCR -> text cleaning -> KB/ML matching -> warnings -> health score

Flow
  extract_text_from_image  EasyOCR (reader cached at module level)
  clean_ingredient_text    regex cleaner; returns ParsedIngredient items (a str subclass that
                           also carries .ins_codes and .trace, so old code treating the
                           result as a plain list of strings keeps working)
  load_knowledge_base      data/nutriscan_kb_v2.csv, indexed by name AND aliases
  match_ingredient         exact name -> alias -> INS code -> difflib fuzzy (0.85) -> ML classifier
  check_warnings           KB tags + allergen keyword safety net, per the user's profile
  compute_health_score     rule-based, formula in its docstring
"""

import difflib
import os
import re

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_KB_PATH = os.path.join(BASE_DIR, "data", "nutriscan_kb_v2.csv")

RISK_WEIGHTS = {"low": 0, "medium": 5, "high": 15, "unknown": 3}
CONFIRMED_ALLERGEN_PENALTY = 20
SOFT_ALLERGEN_PENALTY = 10   # possible or trace
CONDITION_PENALTY = 15
FUZZY_CUTOFF = 0.85
JUNK_FUZZY_CUTOFF = 0.45     # ML-only items with no KB key this similar are treated as OCR noise

# ----------------------------------------------------------------------------- OCR

_OCR_READER = None


def _get_ocr_reader():
    """Create the EasyOCR reader once per process (loading it takes seconds)."""
    global _OCR_READER
    if _OCR_READER is None:
        import easyocr
        _OCR_READER = easyocr.Reader(["en"], gpu=False)
    return _OCR_READER


def extract_text_from_image(image_path):
    """Extract text from a food label while preserving line breaks."""
    reader = _get_ocr_reader()
    results = reader.readtext(image_path, detail=0, paragraph=False)
    return "\n".join(results)


# ----------------------------------------------------------------------------- cleaner

class ParsedIngredient(str):
    """An ingredient name (plain str) plus parsed extras.

    .ins_codes : INS codes read from the label text, e.g. ["211"] or ["322(i)"]
    .trace     : True when the item came from "may contain ..." / shared-facility wording
    .declared  : True when the item came only from a "contains ..." allergen declaration
    """

    def __new__(cls, name, ins_codes=(), trace=False, declared=False):
        obj = super().__new__(cls, name)
        obj.ins_codes = list(ins_codes)
        obj.trace = bool(trace)
        obj.declared = bool(declared)
        return obj


class ParsedLabel(list):
    """List of ParsedIngredient plus the label's allergen statements (not ingredients).

    .declared_allergens : allergen keys named in "Contains ..." (confirmed), e.g. ["dairy", "peanut", "soy"]
    .declared_possible  : ambiguous ones named there (bare "nuts"): peanut / tree_nut as POSSIBLE
    """
    declared_allergens = ()
    declared_possible = ()


_IGNORE_CONTAINS = re.compile(r"\bvegetable\s+fats?\s+in\s+addition\s+to\s+cocoa\s+butter\b")


def _allergen_pieces(part):
    """'milk peanut' (comma lost by OCR) -> ['milk', 'peanut']; leaves 'tree nut' / 'wheat flour' alone."""
    spans = sorted({(m.start(), m.end()) for pats in ALLERGEN_KEYWORDS.values()
                    for pat in pats for m in re.finditer(pat, part)})
    if len(spans) < 2 or any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        return [part]
    if re.sub(r"\s+", "", "".join(c if not any(a <= i < b for a, b in spans) else ""
                                   for i, c in enumerate(part))):
        return [part]
    return [part[a:b] for a, b in spans]


# Allergen words used ONLY to repair OCR typos inside "contains ..." / "may contain ..." statements
ALLERGEN_WORDS = [
    "milk", "whey", "casein", "lactose", "cheese", "peanut", "groundnut", "soy", "soya", "soybean",
    "wheat", "gluten", "barley", "rye", "egg", "fish", "nut", "almond", "cashew", "walnut",
    "pistachio", "hazelnut", "prawn", "shrimp", "crab", "sulphite", "sulfite",
]
_ALLERGEN_FORMS = {w + s for w in ALLERGEN_WORDS for s in ("", "s", "es")}
_NOT_ALLERGEN = {
    "and", "or", "may", "the", "also", "not", "nor", "for", "with", "when", "what", "that", "this",
    "than", "less", "trace", "traces", "tree", "from", "made", "free", "such", "some", "other",
    "only", "can", "all", "any", "ins", "contain", "contains",
}
ALLERGEN_FUZZY_CUTOFF = 0.75


def _allergen_word(tok):
    """Known allergen word this token is (or is an OCR typo of), else None.
    Typos: difflib ratio >= 0.75, or - for 3-4 letter words such as soy/egg/nut/fish, where one
    wrong letter already drops the ratio below 0.75 - same length, same first letter, one letter off
    ('say', 'sov', 's0y' -> soy)."""
    if len(tok) < 3 or not re.search(r"[a-z]", tok) or tok in _NOT_ALLERGEN:
        return None
    if tok in _ALLERGEN_FORMS:
        return tok
    close = difflib.get_close_matches(tok, ALLERGEN_WORDS, n=1, cutoff=ALLERGEN_FUZZY_CUTOFF)
    if close:
        return close[0]
    for w in ALLERGEN_WORDS:
        if (len(w) <= 4 and len(tok) == len(w) and tok[0] == w[0]
                and sum(a != b for a, b in zip(tok, w)) == 1):
            return w
    return None


def _correct_allergen_words(seg):
    """Fix OCR typos of allergen words inside one contains / may-contain statement
    ('say' -> 'soy', 'peanul' -> 'peanut'). Never used on ordinary ingredient text."""
    def fix(m):
        tok = m.group(0)
        w = _allergen_word(tok)
        return tok if (w is None or tok in _ALLERGEN_FORMS) else w
    return re.sub(r"[a-z0-9]+", fix, seg)


_STATEMENT = re.compile(r"\b(?:may\s+contain|contains?)\b")


def _join_statement_lines(text):
    """OCR often breaks the line inside an allergen statement ('Contains Milk, Peanut and' / 'Soy.').
    Join the next line onto an unfinished 'contains ...' / 'may contain ...' line when that line ends
    with a comma, 'and' or 'or', or the next line starts with an allergen word."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        cur = lines[i]
        while i + 1 < len(lines):
            found = list(_STATEMENT.finditer(cur))
            nxt = lines[i + 1].strip()
            if not found or not nxt:
                break
            tail = cur[found[-1].end():]
            if "." in tail:                                  # statement already finished
                break
            first = re.match(r"[a-z0-9]+", nxt)
            if (re.search(r"(?:,|\band|\bor)\s*$", tail)
                    or (first and "less than" not in tail and _allergen_word(first.group()))):
                cur = cur.rstrip() + " " + nxt
                i += 1
            else:
                break
        out.append(cur)
        i += 1
    return "\n".join(out)


CLASS_WORDS = {
    "preservative", "preservatives", "colour", "colours", "color", "colors",
    "emulsifier", "emulsifiers", "stabilizer", "stabilizers", "stabiliser", "stabilisers",
    "antioxidant", "antioxidants", "acidity regulator", "acidity regulators",
    "sweetener", "sweeteners", "flavour enhancer", "flavour enhancers",
    "flavor enhancer", "flavor enhancers", "thickener", "thickeners",
    "raising agent", "raising agents", "anticaking agent", "anti-caking agent",
    "humectant", "gelling agent", "firming agent", "bulking agent", "improver",
    "flour treatment agent", "permitted natural colour", "permitted natural colours",
    "permitted synthetic food colour", "permitted synthetic food colours",
    "permitted class ii preservative", "class ii preservative", "leavening agent",
    "leavening agents", "mineral", "minerals", "vitamins", "flavour", "flavours",
}
DESCRIPTORS = {"natural", "artificial", "nature identical", "permitted", "synthetic",
               "approved", "added", "and", "or", "etc", "others", "other"}

_NUTRIENT_WORDS = (
    r"energy|calories|protein|proteins|total fat|saturated fat|trans fat|fat|"
    r"carbohydrates?|total carbohydrates?|total sugars?|added sugars?|sugars?|"
    r"dietary fib(?:re|er)|fib(?:re|er)|cholesterol|sodium|potassium|calcium|iron|"
    r"vitamin [a-z0-9]+|salt"
)
_UNIT = r"(?:g|mg|mcg|µg|ug|kcal|kj|cal)\b"
# a nutrient word followed by a number and a unit (a nutrition-table row)
_NUTR_ROW = re.compile(
    rf"\b(?:{_NUTRIENT_WORDS})\b\s*(?:\([^)]*\))?\s*[:\-–]?\s*[<>~]?\s*\d+(?:[.,]\d+)?\s*{_UNIT}")
# "energy (kcal) 480" - unit sits in brackets, number follows
_NUTR_ROW_BRACKET = re.compile(
    rf"\b(?:{_NUTRIENT_WORDS})\b\s*\([^)]*(?:g|mg|kcal|kj)[^)]*\)\s*[:\-–]?\s*\d")
_NUTR_HEADING = re.compile(
    r"^\W*(nutrition(al)?\s+(information|facts|value)s?|per\s+100\s?(g|ml)|per\s+serv\w*|"
    r"amount\s+per|serving\s+size|servings?\s+per|%\s*(rda|dv|daily)|approx\w*\s+values?|"
    r"typical\s+values?|values?\s+per)\b")
_NUMERIC_ONLY = re.compile(r"^\W*[<>~]?\d+(?:[.,]\d+)?\s*(?:g|mg|mcg|µg|kcal|kj|cal|%|ml)?\W*$")
_BARE_NUTRIENT = {
    "energy", "calories", "protein", "proteins", "total fat", "saturated fat", "trans fat",
    "carbohydrate", "carbohydrates", "total carbohydrate", "total carbohydrates",
    "total sugar", "total sugars", "added sugar", "added sugars", "dietary fibre",
    "dietary fiber", "cholesterol", "sodium", "per 100g", "per 100 g", "serving size",
}

# stop cutting the ingredient list where non-ingredient sections begin
STOP_PATTERNS = [
    r"\bnutritional information\b", r"\bnutrition facts\b", r"\bnutrition information\b",
    r"\bnutritional value\b", r"\bnutrition value\b", r"\bamount per\b", r"\bper 100\s?g\b",
    r"\bstorage\b", r"\bstore (?:in|at|below|away)\b", r"\bkeep (?:in|away|refrigerated)\b",
    r"\bnet (?:weight|wt|qty|quantity|content)\b", r"\bmanufactured by\b", r"\bmarketed by\b",
    r"\bpacked by\b", r"\bdistributed by\b", r"\bimported by\b", r"\bbest before\b",
    r"\buse before\b", r"\bcustomer care\b", r"\bcustomer service\b", r"\bexpiry date\b",
    r"\bmanufacturing date\b", r"\bmfg\.? date\b", r"\bbatch (?:number|no)\b", r"\bmrp\b",
    r"\bfssai\b", r"\bonce opened\b", r"\bconsume within\b", r"\bdirections? (?:for|to)\b",
    r"\bhow to (?:use|prepare)\b",
]
# headings that are removed but do NOT end the list (the allergen block carries "contains ...")
_ALLERGEN_HEADING = re.compile(r"\b(?:allergen|allergy)\s+(?:information|advice|info)\s*[:\-]?", re.I)

# "and" inside a real ingredient name must not split it
_AND_PROTECT = [
    r"mono[\s\-]*and[\s\-]*di[\s\-]*glycerides?(?: of fatty acids?)?",
    r"diacetyl\w*\s+and\s+fatty\s+acids?\s+esters?\s+of\s+glycerol",
    r"fd\s*and\s*c\b",
    r"(?:sorbitol|maltitol)\s+and\s+(?:sorbitol|maltitol|matitol)\s+syrup",
    r"vitamins?\s+and\s+minerals?",
]
_KNOWN_AND_PHRASES = []      # filled from the KB by load_knowledge_base
_KNOWN_KEYS = set()          # all KB names + aliases, filled by load_knowledge_base

_SUFFIX = r"(\(\s*[ivx]+\s*\)|_[ivx]+|[a-z])"
_INS_DIGITS = r"1[0-5]\d\d|[1-9]\d\d"            # valid INS numbers only: 100-1599
_INS_NUM = r"(" + _INS_DIGITS + r")(?!\d)"
_INS_ONE = re.compile(r"^(?:ins|e)?\s*[-:]?\s*" + _INS_NUM + r"\s*" + _SUFFIX + r"?$")
_INS_INLINE = re.compile(r"\b(?:ins|e)\s?[-:]?\s?" + _INS_NUM + r"(?:\s*(\(\s*[ivx]+\s*\)|_[ivx]+)|([a-z])\b)?", re.I)
_BAD_INS_NAME = re.compile(r"^(?:ins|e)[\s\-:]*\d+\s*(?:\(?[ivx]+\)?|[a-z])?$")   # e.g. "ins3593"


def _norm_code(num, suffix=None):
    """'322' + '_i' / '(i)' / 'a'  ->  '322(i)' / '322a'."""
    s = (suffix or "").replace(" ", "").lower()
    if s.startswith("_"):
        s = "(" + s[1:] + ")"
    return str(num) + s


def _inline_code(m):
    return _norm_code(m.group(1), m.group(2) or m.group(3))


def _codes_from_list(text, allow_bare=True):
    """If `text` is only INS codes (e.g. '211', 'ins 322(i)', '330, 331 and 332'), return them.
    A bare number (no INS/E prefix) only counts when allow_bare (it follows an additive word)."""
    pieces = [p.strip() for p in re.split(r",|;|&|/|\band\b|\bor\b", text.lower()) if p.strip()]
    if not pieces:
        return None
    codes = []
    for p in pieces:
        m = _INS_ONE.match(p)
        if not m or not (allow_bare or re.match(r"(?:ins|e)\b|e\d", p)):
            return None
        codes.append(_norm_code(m.group(1), m.group(2)))
    return codes


def _protect_and(text):
    pats = list(_AND_PROTECT) + [re.escape(p).replace(r"\ ", r"\s+")
                                 for p in _KNOWN_AND_PHRASES]
    for p in pats:
        text = re.sub(p, lambda m: re.sub(r"\band\b", "\x01", m.group(0)), text, flags=re.I)
    return text


def _split_top(text):
    """Split on , ; : sentence-periods and ' and ' at bracket depth 0."""
    parts, buf, depth, i = [], [], 0, 0
    while i < len(text):
        ch = text[i]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        if depth == 0 and ch == ":":
            label = "".join(buf)
            if re.search(r"\(\s*[<>~]?\s*\d+(?:[.,]\d+)?\s*%\s*\)\s*$", label):
                pass                                         # "centre filling (65%):" is a heading
            elif len(label.split()) > 3:                     # "noodles:" / "colour:" are headings
                parts.append(label)
            buf = []
        elif depth == 0 and ch in ",;":
            parts.append("".join(buf)); buf = []
        elif depth == 0 and ch == "." and (i + 1 == len(text) or text[i + 1].isspace()):
            parts.append("".join(buf)); buf = []
        elif depth == 0 and text.startswith(" and ", i):
            parts.append("".join(buf)); buf = []
            i += 5
            continue
        else:
            buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _is_table_row(text):
    t = text.strip()
    return bool(_NUTR_ROW.search(t) or _NUTR_ROW_BRACKET.search(t)
                or _NUTR_HEADING.match(t) or _NUMERIC_ONLY.match(t))


def _first_bracket(text):
    """Return (head, inner, tail) for the first top-level (...) / [...] group, or None."""
    start = None
    depth = 0
    for i, ch in enumerate(text):
        if ch in "([":
            if depth == 0:
                start = i
            depth += 1
        elif ch in ")]" and depth > 0:
            depth -= 1
            if depth == 0:
                return text[:start], text[start + 1:i], text[i + 1:]
    if depth > 0 and start is not None:          # unclosed bracket (OCR): take the rest
        return text[:start], text[start + 1:], ""
    return None


def _tidy_name(text):
    t = text.lower().replace("\x01", "and")
    t = re.sub(r"[<>~]?\s*\b\d+(?:[.,]\d+)?\s*%", " ", t)               # percentages, "<5%" 
    t = re.sub(r"\b(?:min|max|approx)\.?\s*$", " ", t)
    t = re.sub(r"^\s*(?:and|or|with|of|from|as|for|to|contains?)\s+", "", t)
    t = re.sub(r"[\u2022*_~|]", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" \t.-–:;,'\"")
    return t


_HEADING_END = re.compile(r"\b(?:coating|filling|layer|base|centre|center)(?:\s*\(?\s*[<>~]?\s*\d+(?:[.,]\d+)?\s*%\s*\)?)?$")


def _emit(out, name, codes, trace):
    name = _tidy_name(name)
    codes = list(codes)
    if _HEADING_END.search(name) and name not in _KNOWN_KEYS:   # "milk chocolate coating (35%)" is a heading
        return
    if not name and codes:
        name = "ins " + codes[0]
    if len(name) < 3 or not re.search(r"[a-z]", name):
        return
    if not codes and _BAD_INS_NAME.match(name):               # "ins3593": outside 100-1599
        return
    if name in DESCRIPTORS or name in _BARE_NUTRIENT:
        return
    if _is_table_row(name):
        return
    if len(name.split()) > 8 and name not in _KNOWN_KEYS:
        return
    out.append(ParsedIngredient(name, codes, trace))


def _process_item(raw, out, trace):
    item = raw.strip()
    if not item:
        return
    br = _first_bracket(item)
    if br is not None:
        head, inner, tail = br
        if re.match(r"\s*[a-z]", tail) and re.search(r"[,;]", inner) and not _codes_from_list(inner):
            # sub-ingredient list followed directly by another item (lost comma): split there
            _process_item(item[:len(item) - len(tail)], out, trace)
            _process_item(tail, out, trace)
            return
        parent = _tidy_name((head + " " + tail))
        inner_c = inner.strip()
        inner_nopct = _tidy_name(inner_c)
        if not inner_nopct:                                  # "(15%)" etc.
            _process_item(parent, out, trace)
            return
        codes = _codes_from_list(inner_c, allow_bare=(not parent or parent in CLASS_WORDS
                                                      or parent in _KNOWN_KEYS))
        if codes:
            if not parent or parent in CLASS_WORDS or len(codes) > 1:
                label = parent or "ins"
                for c in codes:
                    _emit(out, f"{label} ({c})" if parent else f"ins {c}", [c], trace)
            else:
                _emit(out, parent, codes, trace)
            return
        full = _tidy_name(f"{parent} ({inner_c})")
        if full in _KNOWN_KEYS:                              # e.g. "sugar (sucrose)"
            _emit(out, full, [], trace)
            return
        if (parent in _KNOWN_KEYS and "," not in inner_c and ";" not in inner_c
                and _tidy_name(inner_c) not in _KNOWN_KEYS):
            _emit(out, full, [], trace)                     # e.g. "edible vegetable oil (palm)"
            return
        # sub-ingredient list: keep the parent (unless it is a generic class word) + children
        if parent and parent not in CLASS_WORDS:
            _process_item(parent, out, trace)
        for sub in _split_top(_protect_and(inner_c)):
            _process_item(sub, out, trace)
        return

    # no brackets: INS codes written inline ("e211", "ins 322(i)", "preservative 211")
    low = item.lower()
    m = re.match(r"^\s*(" + "|".join(sorted(map(re.escape, CLASS_WORDS), key=len, reverse=True))
                 + r")\s+((?:" + _INS_DIGITS + r")[a-z]?)(?!\d)\s*$", low)
    if m:
        _emit(out, f"{m.group(1)} ({m.group(2)})", [m.group(2)], trace)
        return
    found = [_inline_code(mm) for mm in _INS_INLINE.finditer(low)]
    if found:
        rest = _INS_INLINE.sub(" ", low)
        rest = re.sub(r"\s+", " ", rest).strip(" -:,")
        if not rest or rest in CLASS_WORDS:
            for c in found:
                _emit(out, f"{rest or 'ins'} ({c})" if rest else f"ins {c}", [c], trace)
        else:
            _emit(out, rest, found, trace)
        return
    _emit(out, item, [], trace)


def _segment_pattern(lead):
    # segment runs to a period / newline, but continues onto the next line after a trailing comma
    return re.compile(lead + r"((?:[^.\n]*,[ \t]*\n)*[^.\n]*)", re.I)


_MAY_CONTAIN = _segment_pattern(
    r"\bmay\s+contain(?:\s+(?:traces?|trace\s+amounts?)(?:\s+of)?)?\s*[:\-]?\s*")
_FACILITY = _segment_pattern(
    r"(?:\b(?:produced|manufactured|packed|processed|made)\s+(?:in|at)\s+(?:a|an|the)?\s*)?"
    r"\b(?:facility|factory|unit|plant|premises|line)\b[^.\n]{0,60}?"
    r"\b(?:processes|processing|handles|handling|uses|using|manufactures)\b\s*[:\-]?\s*")
_CONTAINS = _segment_pattern(r"\bcontains?\b\s*[:\-]?\s*(?:less\s+than\s+\d+%\s+of\s*:?\s*)?")


def _cut_at_stops(text):
    for pat in STOP_PATTERNS:
        text = re.split(pat, text, maxsplit=1, flags=re.I)[0]
    return text


def _pull_segments(text, pattern):
    """Remove every match of `pattern` from text; return (text_without, [segment strings])."""
    segs = []

    def grab(m):
        seg = _cut_at_stops(m.group(1).replace("\n", " "))
        seg = re.sub(r"\b(?:and|or)\s*$", "", seg.strip())
        segs.append(seg)
        return " . "
    return pattern.sub(grab, text), segs


def parse_label_text(raw_text):
    """Full parser; returns a list of ParsedIngredient (name, .ins_codes, .trace)."""
    if not raw_text or not raw_text.strip():
        return ParsedLabel()
    text = raw_text.lower().replace("\r", "\n")
    text = re.sub(r"(\d{3,4})\s*\(\s*([ivx]+)\s*\)", r"\1_\2", text)   # 322(i) -> 322_i
    text = re.sub(r"\s&\s", " and ", text)

    # 1. trace statements and "contains ..." statements, wherever they appear on the label
    text = _join_statement_lines(text)
    text, trace_segs = _pull_segments(text, _MAY_CONTAIN)
    text, fac_segs = _pull_segments(text, _FACILITY)
    text = _ALLERGEN_HEADING.sub(" . ", text)
    text, contains_segs = _pull_segments(text, _CONTAINS)

    # 2. ingredient section
    heading = re.search(r"\b(?:ingr\w{3,8}|incredients?)\b\s*[:\-–]?", text)   # also OCR "incredients"
    if heading:
        text = text[heading.end():]
    text = _cut_at_stops(text)

    # 3. drop nutrition-table rows line by line, then re-join wrapped lines
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln and not _is_table_row(ln)]
    body = " ".join(lines) if re.search(r"[,;]", " ".join(lines)) or len(lines) <= 1 \
        else ", ".join(lines)
    # OCR garble of a percentage: "(21%6)" -> "(21%)"
    body = re.sub(r"(\d)\s*%\s*\d\s*\)", r"\1%)", body)
    # a percentage normally ends an item: "peanuts (21%) sugar" -> "peanuts (21%), sugar" (lost comma)
    body = re.sub(r"(\d\s*%\s*\)?)[ \t]+(?=[a-z])(?!(?:of|and|or|w/w|w/v|by|added|approx|min|max|dry|per)\b)",
                  r"\1, ", body)
    # a closing bracket followed directly by a word is also a lost comma: "(palm oil) milk solids"
    body = re.sub(r"(\))[ \t]+(?=[a-z])(?!(?:of|and|or|w/w|w/v|by|per)\b)", r"\1, ", body)

    out = []
    for part in _split_top(_protect_and(body)):
        _process_item(part, out, trace=False)
    # "Contains milk, peanut and soy" is a label allergen statement, not a list of ingredients
    decl_conf, decl_poss = set(), set()
    for seg in contains_segs:
        seg = _correct_allergen_words(_IGNORE_CONTAINS.sub(" ", seg))
        seg = re.sub(r"[:;]", ",", seg)                    # OCR often reads the final "." as ":"
        for part in _split_top(seg):                       # commas and "and"
            conf, poss = keyword_allergens(part)
            decl_conf |= conf
            decl_poss |= poss
    for seg in trace_segs + fac_segs:
        seg = re.sub(r"[:;]", ",", _correct_allergen_words(seg))
        for part in _split_top(_protect_and(seg)):
            for piece in _allergen_pieces(part):
                _process_item(piece, out, trace=True)

    # 4. de-duplicate (a confirmed mention beats a trace mention of the same thing)
    merged = {}
    for it in out:
        key = str(it)
        if key not in merged:
            merged[key] = it
        else:
            prev = merged[key]
            if prev.trace and not it.trace:
                it.ins_codes = sorted(set(it.ins_codes) | set(prev.ins_codes))
                merged[key] = it
            elif prev.declared and not it.declared:
                it.ins_codes = sorted(set(it.ins_codes) | set(prev.ins_codes))
                merged[key] = it
    label = ParsedLabel(merged.values())
    label.declared_allergens = sorted(decl_conf)
    label.declared_possible = sorted(decl_poss - decl_conf)
    return label


def clean_ingredient_text(raw_text):
    """Extract likely ingredient names from OCR text (see parse_label_text)."""
    return parse_label_text(raw_text)


# ----------------------------------------------------------------------------- KB

def _clean_cell(v):
    return "" if pd.isna(v) else v


# "sal fat" (a cocoa-butter equivalent, oil_fat category) is not in the KB: alias it to cocoa butter
EXTRA_ALIASES = {"sal fat": "cocoa butter"}


def load_knowledge_base(path=None):
    """Load the KB as a dict keyed by lowercase ingredient name AND by every alias.

    Alias keys point at the same record as the canonical name; record["canonical"] is the
    canonical ingredient name, so `key == record["canonical"]` means an exact-name hit.
    """
    df = pd.read_csv(path or DEFAULT_KB_PATH)
    df["ingredient_name"] = df["ingredient_name"].str.lower().str.strip()
    kb = {}
    records = df.to_dict("records")
    for rec in records:
        info = {k: _clean_cell(v) for k, v in rec.items()}
        info["canonical"] = rec["ingredient_name"]
        info["ins_code"] = info["ins_code"] or None
        kb[rec["ingredient_name"]] = info
    for rec in records:
        info = kb[rec["ingredient_name"]]
        for a in str(_clean_cell(rec.get("aliases", ""))).split(";"):
            a = re.sub(r"\s+", " ", a.strip().lower())
            if a and a not in kb:
                kb[a] = info
    for alias, canonical in EXTRA_ALIASES.items():           # code-level aliases; the CSV is not edited
        if alias not in kb and canonical in kb:
            kb[alias] = kb[canonical]
    _KNOWN_KEYS.clear()
    _KNOWN_KEYS.update(kb.keys())
    _KNOWN_AND_PHRASES[:] = [k for k in kb if re.search(r"\band\b", k) and "(" not in k]
    _INS_CACHE.pop(id(kb), None)
    return kb


_INS_CACHE = {}


def _ins_index(kb):
    """code -> record, built from ins_code columns and INS-looking aliases."""
    idx = _INS_CACHE.get(id(kb))
    if idx is not None and idx[0] is kb:
        return idx[1]
    index = {}
    for key, info in kb.items():
        if key == info.get("canonical") and info.get("ins_code"):
            m = re.match(r"^(?:ins)?\s*(\d{3,4})\s*" + _SUFFIX + r"?$",
                         str(info["ins_code"]).lower())
            if m:
                index.setdefault(_norm_code(m.group(1), m.group(2)), info)
        else:
            m = _INS_ONE.match(key)
            if m and key.startswith(("ins", "e")):
                index.setdefault(_norm_code(m.group(1), m.group(2)), info)
    _INS_CACHE[id(kb)] = (kb, index)
    return index


def _lookup_ins(code, kb):
    index = _ins_index(kb)
    code = _norm_code(code)
    if code in index:
        return index[code]
    base = re.sub(r"\([ivx]+\)$", "", code)
    if base in index:
        return index[base]
    cands = {id(v): v for k, v in index.items() if k.startswith(base + "(")}
    if len(cands) == 1:                       # "460" when only "460(i)" is in the KB
        return next(iter(cands.values()))
    return None


def _name_variants(name):
    n = re.sub(r"\s+", " ", name.lower().strip())
    variants = [n]
    no_br = re.sub(r"\s*[(\[][^)\]]*[)\]]", "", n).strip()
    if no_br and no_br not in variants:
        variants.append(no_br)
    for v in list(variants):
        if v.endswith("s") and len(v) > 4:
            variants.append(v[:-1])
        if v.endswith("es") and len(v) > 5:
            variants.append(v[:-2])
        if "color" in v:
            variants.append(v.replace("color", "colour"))
    return list(dict.fromkeys(variants))


_NUM_TOKEN = re.compile(r"^(?:\d+|[ivx]+)$")


def _fuzzy_compatible(name, cand):
    """Fuzzy matches are only for OCR typos: same token count, each token (nearly) equal, and
    numbers / roman numerals identical. Rejects 'coconut milk' -> 'coconut oil',
    'polysorbate 60' -> 'polysorbate 20', 'caramel ii' -> 'caramel iii'."""
    ta, tb = name.split(), cand.split()
    if len(ta) != len(tb):
        return False
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if _NUM_TOKEN.match(x) or _NUM_TOKEN.match(y):
            return False
        if difflib.SequenceMatcher(None, x, y).ratio() < 0.8:
            return False
    return True


def _info_from_kb(info, source, match_type, note=None):
    out = dict(info)
    out["source"] = source
    out["match_type"] = match_type
    if note:
        out["match_note"] = note
    return out


def match_ingredient(name, kb, classifier=None, vectorizer=None, ins_codes=None):
    """Resolve an ingredient. Order: exact name -> alias -> INS code -> fuzzy (0.85) -> ML.

    ML predictions give a category only: risk_level "unknown", no allergen tags,
    source "ml_prediction".
    """
    name = re.sub(r"\s+", " ", str(name).lower().strip())
    codes = list(ins_codes or [])
    for m in _INS_INLINE.finditer(name):
        codes.append(_inline_code(m))
    variants = _name_variants(name)

    for v in variants:                                       # 1. exact name, 2. alias
        if v in kb:
            info = kb[v]
            if v == info.get("canonical"):
                return _info_from_kb(info, "knowledge_base", "exact")
            return _info_from_kb(info, "knowledge_base", "alias",
                                 f"alias of '{info['canonical']}'")
    for c in codes:                                          # 3. INS code
        info = _lookup_ins(c, kb)
        if info:
            return _info_from_kb(info, "knowledge_base", "ins_code",
                                 f"matched by INS {c} -> '{info['canonical']}'")
    if len(name) >= 5:                                       # 4. fuzzy
        for cand in difflib.get_close_matches(name, list(kb.keys()), n=5, cutoff=FUZZY_CUTOFF):
            if _fuzzy_compatible(name, cand):
                info = kb[cand]
                return _info_from_kb(info, "knowledge_base_fuzzy", "fuzzy",
                                     f"fuzzy match to '{cand}' - please check")
    base = {"ins_code": None, "risk_level": "unknown", "allergen_tags": "",
            "condition_tags": "", "possible_allergen": "no", "canonical": None,
            "risk_reason": ""}
    if classifier is not None and vectorizer is not None:    # 5. ML (category only)
        pred = classifier.predict(vectorizer.transform([name]))[0]
        return {**base, "category": pred, "source": "ml_prediction", "match_type": "ml",
                "notes": "Not in knowledge base; category predicted by ML classifier",
                "match_note": "category predicted by ML (risk unknown)"}
    return {**base, "category": "unknown", "source": "unmatched", "match_type": "none",
            "notes": "Not in knowledge base", "match_note": "not in knowledge base"}


# ----------------------------------------------------------------------------- allergens

def _split_tags(tag_string):
    if pd.isna(tag_string) or not tag_string:
        return set()
    return set(t.strip().lower() for t in str(tag_string).split(";") if t.strip())


_FREE = r"(?!\s*[- ]?(?:free|less)\b)"
_NO_PLANT_MILK = r"(?<!coconut )(?<!soy )(?<!soya )(?<!almond )(?<!oat )(?<!rice )(?<!cashew )(?<!nut )"
_NO_NUT_BUTTER = (r"(?<!cocoa )(?<!shea )(?<!peanut )(?<!almond )(?<!cashew )(?<!nut )"
                  r"(?<!mango )(?<!kokum )(?<!groundnut )(?<!seed )(?<!sal )")
# Allergen keyword safety net. Applied to every ingredient, even ones not in the KB.
ALLERGEN_KEYWORDS = {
    "peanut": [r"\bpeanuts?\b", r"\bground\s?nuts?\b"],
    "dairy": [_NO_PLANT_MILK + r"\bmilk\b", r"\bwhey\b", r"\bcasein\w*", _NO_NUT_BUTTER + r"\bbutter\b",
              r"\bbuttermilk\b", r"\bghee\b", r"\bcurd\b", r"\bcream\b(?! of tartar)(?<!coconut cream)",
              r"\bpaneer\b",
              # beyond the brief: obvious dairy words
              r"\bcheese\b", r"\byogh?urt\b", r"\blactose\b", r"\bdahi\b"],
    "gluten": [r"\bgluten\b", r"\bwheat\b", r"\bmaida\b", r"\batta\b", r"\bbarley\b", r"\brye\b",
               r"\bsemolina\b", r"\brava\b"],
    "soy": [r"\bsoy(?:a|abean|bean)?s?\b"],
    "egg": [r"\beggs?\b"],
    "tree_nut": [r"\balmonds?\b", r"\bcashews?\b", r"\bwalnuts?\b", r"\bpistachios?\b",
                 r"\bhazelnuts?\b", r"\btree\s+nuts?\b"],
    "fish": [r"\bfish\b"],
    "shellfish": [r"\bprawns?\b", r"\bshrimps?\b", r"\bcrabs?\b"],
    "sulphites": [r"sul(?:ph|f)ites?\b", r"\bsul(?:ph|f)ur dioxide\b"],
    "phenylketonuria": [r"\baspartame\b"],
}
# Bare "nuts" is ambiguous on Indian labels (peanuts or tree nuts): flag both, as POSSIBLE.
_GENERIC_NUTS = re.compile(r"(?<!ground )(?<!tree )(?<!pea)\bnuts?\b" + _FREE)


def keyword_allergens(text):
    """Return (confirmed, possible) allergen sets from the keyword safety net."""
    t = text.lower()
    confirmed = set()
    for allergen, pats in ALLERGEN_KEYWORDS.items():
        if any(re.search(p + _FREE, t) for p in pats):
            confirmed.add(allergen)
    possible = set()
    if _GENERIC_NUTS.search(t):
        possible |= {"peanut", "tree_nut"}
    return confirmed, possible - confirmed


def _looks_like_junk(name, kb):
    """OCR noise: very short, no vowels, or nothing in the KB even loosely similar."""
    letters = re.sub(r"[^a-z]", "", name.lower())
    if len(letters) < 3 or not re.search(r"[aeiou]", letters):
        return True
    return not difflib.get_close_matches(name.lower(), list(kb.keys()), n=1, cutoff=JUNK_FUZZY_CUTOFF)


def _short_notes(info, trace, possible_flag, tags):
    parts = []
    if info.get("risk_reason"):
        parts.append(info["risk_reason"])
    if info.get("match_note"):
        parts.append(info["match_note"])
    if possible_flag and not tags:
        parts.append("allergen depends on source and label may not say (no allergen tag)")
    elif possible_flag:
        parts.append("allergen only POSSIBLE: depends on source")
    if trace:
        parts.append("listed as 'may contain' / trace")
    if info.get("source") == "ml_prediction":
        parts.append("ML-predicted category, risk unknown")
    return "; ".join(parts)


class ResultList(list):
    """check_warnings result rows plus the label-level hits (allergens named in "Contains ...")."""
    label_declares = ()      # selected allergens the label declares (confirmed)
    label_possible = ()      # selected allergens declared only ambiguously (bare "nuts")


def check_warnings(ingredient_list, kb, classifier, vectorizer, user_profile):
    """Check each ingredient against the user's selected allergies/conditions.

    Per result:
      allergen_warning           confirmed allergen hits (user's allergies only)
      possible_allergen_warning  hits that are only possible (KB possible_allergen = yes,
                                 or an ambiguous word such as bare "nuts")
      trace_allergen_warning     hits on "may contain" / shared-facility items
      condition_warning          condition tags that match the user's conditions
    Duplicates resolving to the same KB row are reported once.
    """
    user_allergies = set(a.strip().lower() for a in user_profile.get("allergies", []))
    user_conditions = set(c.strip().lower() for c in user_profile.get("conditions", []))

    results, index_by_key = ResultList(), {}
    declared = set(getattr(ingredient_list, "declared_allergens", ()) or ())
    declared_poss = set(getattr(ingredient_list, "declared_possible", ()) or ())
    results.label_declares = sorted(user_allergies & declared)                    # confirmed
    results.label_possible = sorted((user_allergies & declared_poss) - set(results.label_declares))
    for ing in ingredient_list:
        name = str(ing)
        trace = bool(getattr(ing, "trace", False))
        codes = list(getattr(ing, "ins_codes", []) or [])
        info = match_ingredient(name, kb, classifier, vectorizer, ins_codes=codes)
        unrecognised = (not trace and info.get("source") == "ml_prediction"
                        and _looks_like_junk(name, kb))
        if unrecognised:                                     # listed, but never penalised
            info = {**info, "category": "unrecognised", "source": "unrecognised",
                    "match_type": "none", "notes": "Unrecognised (looks like OCR noise); not scored",
                    "match_note": "unrecognised: looks like OCR noise, not scored"}

        kb_tags = _split_tags(info.get("allergen_tags", ""))
        cond_tags = _split_tags(info.get("condition_tags", ""))
        possible_flag = str(info.get("possible_allergen", "no")).strip().lower() == "yes"
        kw_conf, kw_poss = keyword_allergens(name)

        confirmed = set(kw_conf) | (set() if possible_flag else kb_tags)
        possible = (set(kw_poss) | (kb_tags if possible_flag else set())) - confirmed
        if trace:
            trace_set, confirmed, possible = confirmed | possible, set(), set()
        else:
            trace_set = set()

        allergen_hit = user_allergies & confirmed
        possible_hit = user_allergies & possible
        trace_hit = user_allergies & trace_set
        condition_hit = set() if trace else (user_conditions & cond_tags)

        unresolved = info.get("source") in ("ml_prediction", "unmatched")
        declared = bool(getattr(ing, "declared", False))
        entry = {
            "ingredient": name,
            "category": info.get("category"),
            "risk_level": info.get("risk_level", "unknown") or "unknown",
            "ins_code": info.get("ins_code") or (("INS" + codes[0]) if codes else None),
            "source": info.get("source"),
            "match_type": info.get("match_type"),
            "trace": trace,
            "declared": declared,
            # trace items and bare allergen declarations ("contains soy") that are not in the KB
            # are used for allergen warnings only: they do not add a risk deduction
            "scored": not trace and not unrecognised and not (declared and unresolved),
            "unrecognised": unrecognised,
            "possible_allergen_flag": possible_flag,
            "allergen_warning": sorted(allergen_hit) or None,
            "possible_allergen_warning": sorted(possible_hit) or None,
            "trace_allergen_warning": sorted(trace_hit) or None,
            "condition_warning": sorted(condition_hit) or None,
            "notes": _short_notes(info, trace, possible_flag, kb_tags),
            "kb_notes": info.get("notes", ""),
        }
        key = info.get("canonical") or name
        if key in index_by_key:
            i = index_by_key[key]
            if results[i]["trace"] and not trace:      # a confirmed mention replaces a trace one
                results[i] = entry
            continue
        index_by_key[key] = len(results)
        results.append(entry)

    # report each allergen once: at its strongest level (label statement > confirmed > possible > trace)
    owner = {a: (-1, -1) for a in results.label_declares + results.label_possible}
    fields = ("allergen_warning", "possible_allergen_warning", "trace_allergen_warning")
    for level, field in enumerate(fields):
        for i, r in enumerate(results):
            for a in r[field] or []:
                owner.setdefault(a, (level, i))
    for i, r in enumerate(results):
        for level, field in enumerate(fields):
            r[field] = sorted(a for a in (r[field] or []) if owner[a] == (level, i)) or None
    return results


def compute_health_score(results):
    """Rule-based score, start at 100 and subtract (clamped to 0-100):

        score = 100
              - sum over ingredients of RISK_WEIGHTS[risk_level]       low 0, medium 5, high 15, unknown 3
              - 20 per CONFIRMED allergen hit                          (allergies the user selected;
                                                                         each allergen counted once, incl.
                                                                         "Contains ..." label statements)
              - 10 per POSSIBLE or TRACE allergen hit                  (allergies the user selected)
              - 15 per selected condition that has at least one hit    (each condition counted ONCE,
                                                                         however many ingredients hit it)

    Notes: allergen/condition deductions exist only for what the user selected (check_warnings
    only reports those), and plain foods are never penalised just for being allergens (their
    risk_level is low). Items listed as "may contain" (trace), and bare "contains X"
    declarations that are not in the KB, get no risk deduction (and trace items no condition
    deduction either); a trace item only costs the 10-point allergen deduction when it hits
    the user's selected allergy.
    """
    score = 100
    score -= CONFIRMED_ALLERGEN_PENALTY * len(getattr(results, "label_declares", ()))
    score -= SOFT_ALLERGEN_PENALTY * len(getattr(results, "label_possible", ()))
    conditions_hit = set()
    for r in results:
        if r.get("scored", not r.get("trace")):
            score -= RISK_WEIGHTS.get(r.get("risk_level"), 3)
        score -= CONFIRMED_ALLERGEN_PENALTY * len(r.get("allergen_warning") or [])
        score -= SOFT_ALLERGEN_PENALTY * (len(r.get("possible_allergen_warning") or [])
                                          + len(r.get("trace_allergen_warning") or []))
        conditions_hit |= set(r.get("condition_warning") or [])
    score -= CONDITION_PENALTY * len(conditions_hit)        # once per condition, not per ingredient
    return max(0, min(100, score))
