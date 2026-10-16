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
    """Extract likely ingredient names from OCR text."""

    if not raw_text or not raw_text.strip():
        return []

    # Convert text to lowercase and normalize line endings.
    text = raw_text.lower().replace("\r", "\n")

    # Find the beginning of the ingredient section.
    ingredient_match = re.search(
        r"\bingredients?\s*[:\-]?\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # If an ingredient heading is found, use the text after it.
    if ingredient_match:
        text = ingredient_match.group(1)

    # Stop processing when common non-ingredient sections begin.
    stop_patterns = [
        r"\bnutritional information\b",
        r"\bnutrition facts\b",
        r"\bnutrition information\b",
        r"\bamount per\b",
        r"\ballergen information\b",
        r"\ballergy information\b",
        r"\bstorage instructions\b",
        r"\bstorage\b",
        r"\bnet weight\b",
        r"\bmanufactured by\b",
        r"\bmarketed by\b",
        r"\bbest before\b",
        r"\buse before\b",
        r"\bcustomer care\b",
        r"\bcustomer service\b",
        r"\bexpiry date\b",
        r"\bmanufacturing date\b",
        r"\bbatch number\b",
        r"\bbatch no\b",
        r"\bmrp\b",
    ]

    for pattern in stop_patterns:
        text = re.split(pattern, text, maxsplit=1)[0]

    # Remove text inside parentheses.
    text = re.sub(r"\([^)]*\)", " ", text)

    # Split text into possible ingredient entries.
    parts = re.split(r"[,;:\n]", text)

    ingredients = []

    for part in parts:
        # Remove unnecessary punctuation around each entry.
        item = part.strip(" \t.-•*")

        # Ignore empty or very short entries.
        if len(item) < 3:
            continue

        # Ignore entries that contain no letters.
        if not re.search(r"[a-z]", item):
            continue

        # Ignore common nutrition labels and packaging information.
        # Do not exclude the standalone word "sugar":
        # sugar can be a genuine ingredient.
        if re.search(
            r"\b(energy|protein|total fat|saturated fat|trans fat|"
            r"sodium|calcium|total carbohydrates|carbohydrates|"
            r"cholesterol|kcal|kj|mg|serving|weight|volume|"
            r"storage|expiry|manufactured|marketed|customer care)\b",
            item
        ):
            continue

        # Normalize spaces.
        item = re.sub(r"\s+", " ", item).strip()

        # Add each unique entry once.
        if item and item not in ingredients:
            ingredients.append(item)

    return ingredients


def load_knowledge_base(path="data/ingredient_knowledge_base.csv"):
    """Load the KB as a dict keyed by lowercase ingredient name."""
    df = pd.read_csv(path)
    df["ingredient_name"] = df["ingredient_name"].str.lower().str.strip()
    return df.set_index("ingredient_name").to_dict("index")


def match_ingredient(name, kb, classifier=None, vectorizer=None):
    """Look up an ingredient in the KB; fall back to the ML classifier for category only."""
    name = name.lower().strip()
    if name in kb:
        info = dict(kb[name])
        info["source"] = "knowledge_base"
        return info
    if classifier is not None and vectorizer is not None:
        pred_category = classifier.predict(vectorizer.transform([name]))[0]
        return {
            "category": pred_category,
            "ins_code": None,
            "risk_level": "unknown",
            "allergen_tags": "",
            "condition_tags": "",
            "notes": "Not in knowledge base; category predicted by ML classifier",
            "source": "ml_prediction",
        }
    return {
        "category": "unknown",
        "ins_code": None,
        "risk_level": "unknown",
        "allergen_tags": "",
        "condition_tags": "",
        "notes": "Not in knowledge base",
        "source": "unmatched",
    }


def _split_tags(tag_string):
    if pd.isna(tag_string) or not tag_string:
        return set()
    return set(t.strip().lower() for t in str(tag_string).split(";") if t.strip())


def check_warnings(ingredient_list, kb, classifier, vectorizer, user_profile):
    """Check each ingredient against the user's allergies/conditions."""
    user_allergies = set(a.strip().lower() for a in user_profile.get("allergies", []))
    user_conditions = set(c.strip().lower() for c in user_profile.get("conditions", []))

    results = []
    for ing in ingredient_list:
        info = match_ingredient(ing, kb, classifier, vectorizer)
        allergen_tags = _split_tags(info.get("allergen_tags", ""))
        condition_tags = _split_tags(info.get("condition_tags", ""))

        allergen_hit = user_allergies & allergen_tags
        condition_hit = user_conditions & condition_tags

        results.append({
            "ingredient": ing,
            "category": info.get("category"),
            "risk_level": info.get("risk_level", "unknown"),
            "ins_code": info.get("ins_code"),
            "source": info.get("source"),
            "allergen_warning": sorted(allergen_hit) if allergen_hit else None,
            "condition_warning": sorted(condition_hit) if condition_hit else None,
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
