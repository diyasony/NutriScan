"""
NutriScan pipeline: OCR -> text cleaning -> KB/ML matching -> warnings -> health score
"""

import re
import pandas as pd

RISK_WEIGHTS = {"low": 0, "medium": 5, "high": 15, "unknown": 3}


def extract_text_from_image(image_path):
    """Extract text from a food label while preserving line breaks."""
    import easyocr

    reader = easyocr.Reader(["en"])
    results = reader.readtext(image_path, detail=0, paragraph=False)

    return "\n".join(results)


def clean_ingredient_text(raw_text):
    """
    Extract candidate ingredients from OCR text.
    Stop before allergen statements and packaging information.
    """

    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.lower().replace("\r\n", "\n").replace("\r", "\n")

    # Find the ingredient section.
    ingredient_match = re.search(
        r"\bingredients?\s*[:\-]?\s*",
        text,
        flags=re.IGNORECASE
    )

    if not ingredient_match:
        return []

    # Keep text after the Ingredients heading.
    text = text[ingredient_match.end():]

    # Stop when a non-ingredient section begins.
    stop_patterns = [
        r"\ballergen note\b",
        r"\ballergen information\b",
        r"\ballergy information\b",
        r"\bcontains\s+(?:milk|soy|soya|wheat|gluten|peanut|egg|fish|shellfish|nuts?)\b",
        r"\bnutritional information\b",
        r"\bnutrition facts\b",
        r"\bnutrition information\b",
        r"\bstorage instructions\b",
        r"\bserving suggestion\b",
        r"\brefer pack for details\b",
        r"\bnet weight\b",
        r"\bmanufactured by\b",
        r"\bmarketed by\b",
        r"\bbest before\b",
        r"\bexpiry date\b",
        r"\bbatch number\b",
        r"\bmrp\b",
    ]

    stop_positions = []

    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            stop_positions.append(match.start())

    if stop_positions:
        text = text[:min(stop_positions)]

    # Remove percentages, such as (7.7%) and (5.3%).
    text = re.sub(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)", " ", text)

    # Remove parenthesized Roman numeral additive notation,
    # such as (iii), while preserving ordinary ingredient names.
    text = re.sub(
        r"\(\s*(?:i{1,3}|iv|v|vi{0,3}|ix|x)\s*\)",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Normalize the conjunction in ingredient lists.
    text = re.sub(r"\s+\band\b\s+", ", ", text)

    # Split on commas and semicolons.
    parts = re.split(r"[,;]", text)

    ingredients = []

    for part in parts:
        item = part.strip(" \t.-•*:")

        # Remove leftover percentages and normalize spaces.
        item = re.sub(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)", "", item)
        item = re.sub(r"\s+", " ", item).strip()

        if len(item) < 3:
            continue

        if not re.search(r"[a-z]", item):
            continue

        # Reject fragments that are clearly not ingredient names.
        if re.fullmatch(r"(iii|ii|iv|i)", item):
            continue

        if item not in ingredients:
            ingredients.append(item)

    return ingredients


def load_knowledge_base(path="data/ingredient_knowledge_base.csv"):
    """Load the knowledge base and normalize empty values."""

    # Read the CSV and interpret empty cells as empty strings.
    df = pd.read_csv(path, keep_default_na=False)

    # Normalize ingredient names.
    df["ingredient_name"] = (
        df["ingredient_name"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # Remove rows with missing ingredient names.
    df = df[df["ingredient_name"] != ""]

    # Convert the table into a dictionary keyed by ingredient name.
    return df.set_index("ingredient_name").to_dict("index")


def match_ingredient(name, kb, classifier=None, vectorizer=None):
    """
    Match an ingredient against the knowledge base.
    ML predictions are suggestions only and never establish ingredient identity or safety.
    """

    # Handle empty or invalid ingredient names.
    if not isinstance(name, str) or not name.strip():
        return {
            "ingredient_name": name,
            "category": "unknown",
            "ins_code": "",
            "risk_level": "unknown",
            "allergen_tags": "",
            "condition_tags": "",
            "notes": "Empty or invalid ingredient name; manual review required.",
            "source": "unmatched",
            "match_status": "unmatched",
            "ingredient_matched": False,
            "review_required": True,
        }

    # Normalize the ingredient name.
    name = " ".join(name.lower().strip().split())

    # Normalize knowledge-base keys.
    normalized_kb = {
        " ".join(str(key).lower().strip().split()): value
        for key, value in kb.items()
        if key is not None
    }

       # 1. Match by ingredient name or INS code.
    matched_name = None

    if name in normalized_kb:
        matched_name = name
    else:
        # Normalize INS codes, accepting formats such as INS330,
        # INS 330, and 330.
        code_match = re.fullmatch(
            r"(?:ins\s*)?(\d{3,4})",
            name,
            flags=re.IGNORECASE
        )

        if code_match:
            requested_code = code_match.group(1)

            for kb_name, kb_info in normalized_kb.items():
                stored_code = str(kb_info.get("ins_code", "")).strip()

                stored_match = re.fullmatch(
                    r"(?:ins\s*)?(\d{3,4})",
                    stored_code,
                    flags=re.IGNORECASE
                )

                if stored_match and stored_match.group(1) == requested_code:
                    matched_name = kb_name
                    break

    if matched_name is not None:
        info = dict(normalized_kb[matched_name])

        # Convert missing CSV values to empty strings.
        for field in (
            "ins_code",
            "risk_level",
            "allergen_tags",
            "condition_tags",
            "notes",
        ):
            value = info.get(field, "")
            if pd.isna(value):
                info[field] = ""

        missing_info = (
            not info.get("category")
            or not info.get("risk_level")
            or info.get("risk_level") == "unknown"
        )

        info.update({
            "ingredient_name": matched_name,
            "source": "knowledge_base",
            "match_status": (
                "exact_match" if name == matched_name else "ins_code_match"
            ),
            "ingredient_matched": True,
            "review_required": missing_info,
        })

        return info
def _split_tags(tag_string):
    if pd.isna(tag_string) or not tag_string:
        return set()
    return set(t.strip().lower() for t in str(tag_string).split(";") if t.strip())


def check_warnings(ingredient_list, kb, classifier, vectorizer, user_profile):
    """Check ingredient information against the user's health profile."""

    user_allergies = {
        str(a).strip().lower()
        for a in user_profile.get("allergies", [])
        if str(a).strip()
    }

    user_conditions = {
        str(c).strip().lower()
        for c in user_profile.get("conditions", [])
        if str(c).strip()
    }

    results = []

    for ing in ingredient_list:

        # Match the ingredient against the knowledge base.
        info = match_ingredient(ing, kb, classifier, vectorizer)

        # Read available tags safely.
        allergen_tags = _split_tags(info.get("allergen_tags", ""))
        condition_tags = _split_tags(info.get("condition_tags", ""))

        # Compare database tags with the user's selections.
        allergen_hit = user_allergies & allergen_tags
        condition_hit = user_conditions & condition_tags

        # Determine whether the ingredient was matched.
        match_status = info.get("match_status", "unknown")
        matched = info.get("source") == "knowledge_base"

        # Missing allergen information must not be interpreted as safety.
        allergen_info_available = bool(allergen_tags)

        results.append({
            "ingredient": ing,
            "category": info.get("category", "unknown"),
            "risk_level": info.get("risk_level", "unknown"),
            "ins_code": info.get("ins_code", ""),
            "source": info.get("source", "unknown"),
            "match_status": match_status,

            "allergen_warning": (
                sorted(allergen_hit) if allergen_hit else None
            ),

            "condition_warning": (
                sorted(condition_hit) if condition_hit else None
            ),

            "ingredient_matched": matched,

            "allergen_information_available": allergen_info_available,

            "review_required": (
                info.get("review_required", False)
                or not matched
                or not allergen_info_available
            ),
        })

    return results


def compute_health_score(results):
    """Rule-based weighted score, starting at 100 and deducting for risk/warnings."""
    score = 100
    for r in results:
        score -= RISK_WEIGHTS.get(r["risk_level"], 3)
        if r["allergen_warning"]:
            score -= 20
        if r["condition_warning"]:
            score -= 15
    return max(0, min(100, score))
