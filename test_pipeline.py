"""
Runs the cleaner + warnings on 8 realistic Indian label texts (OCR-style, with nutrition tables,
line wraps and trace statements) and prints warnings for a peanut-allergic user and a diabetic user.

The label texts are written by us to look like real packs; they are NOT copied from real products.
Run from the project folder:  python test_pipeline.py
"""
import os

import joblib

from pipeline import (check_warnings, clean_ingredient_text, compute_health_score,
                      load_knowledge_base)

BASE = os.path.dirname(os.path.abspath(__file__))

LABELS = {
    "biscuit": """INGREDIENTS: Refined wheat flour (maida) (58%), sugar, edible vegetable oil (palm),
invert syrup, milk solids, leavening agents (INS 500(ii), INS 503(ii)), iodised salt,
emulsifiers (soy lecithin, INS 322(i)), added flavours (artificial flavouring substances - vanilla).
Contains wheat, milk and soy. May contain traces of nuts.
NUTRITIONAL INFORMATION (approx. values per 100 g)
Energy 480 kcal
Protein 6.5 g
Total Fat 18 g
Sodium 320 mg
Manufactured by: ABC Foods Pvt Ltd, Pune. Best before 6 months from packaging.""",

    "chips": """Ingredients: Potato (87%), Edible vegetable oil (palmolein), Spices & condiments (chilli powder,
onion powder, garlic powder), Iodised salt, Maltodextrin, Flavour enhancer (INS 621), Acidity regulator (INS 330),
Sugar, Natural flavour. Contains: Milk. Allergen Information: May contain traces of peanuts and soy.
Nutrition Facts
Energy (kcal) 540
Total Carbohydrate 52 g
Storage: Store in a cool and dry place.""",

    "instant noodles": """INGREDIENTS
Noodles: Wheat flour, palm oil, salt, wheat gluten, thickener (INS 412), acidity regulators (INS 501(i)),
mineral (calcium carbonate), antioxidant (INS 319).
Tastemaker: Spices (onion powder, garlic powder, chilli powder, black pepper), sugar, salt, hydrolysed
soy protein, flavour enhancers (INS 627, INS 631), maltodextrin, colour (INS 150d), anticaking agent (INS 551).
Contains: Wheat, Soy. Produced in a facility that also handles peanuts, milk and eggs.
Nutrition information per serve
Protein 8 g, Sodium 1100 mg
Net weight 70 g  MRP Rs 15""",

    "fruit juice": """INGREDIENTS: Water, apple juice concentrate (22%), sugar, acidity regulator (330),
preservative (211), vitamin C (ascorbic acid), added flavour (nature identical flavouring substances),
sodium benzoate (211), permitted synthetic food colour (INS 110).
NUTRITION FACTS Amount per 100 ml Energy 52 kcal  Total Sugars 12 g
Once opened consume within 3 days. Customer care: 1800 000 0000""",

    "chocolate": """Ingredients: Sugar, Cocoa butter, Milk solids, Cocoa solids, Hazelnut paste (8%),
Emulsifier (soy lecithin), Whey protein concentrate, Skimmed milk powder, Natural and nature identical flavouring
substances (vanillin).
ALLERGEN INFORMATION: Contains milk, soy and hazelnut. May contain peanuts, almonds and wheat.
Nutritional Information per 100g Energy 535 kcal Fat 31 g Sugars 54 g""",

    "bread": """INGREDIENTS: Whole wheat flour (atta) 52%, refined wheat flour (maida), water, sugar, yeast, edible
vegetable oil, iodised salt, soy flour, wheat gluten, preservative (INS 282), emulsifiers (INS 471, INS 481(i)),
flour treatment agent (INS 300), acidity regulator (260).
Contains wheat and soy.
Nutrition facts per slice: Energy 70 kcal, Protein 2.4 g
Best before: 4 days from date of manufacture""",

    "namkeen": """INGREDIENTS: Besan (gram flour), peanuts (24%), edible vegetable oil (palmolein), rice flour,
moth dal, iodised salt, spices and condiments, citric acid, sugar, curry leaves, turmeric powder.
May contain traces of nuts and mustard.
Nutritional information: Energy 520 kcal Protein 18 g Carbohydrates 40 g
Packed by: XYZ Snacks, Indore""",

    "peanut chocolate bar": """INCREDIENTS
Milk chocolate coating (35%): Sugar, Milk solids, Cocoa butter, Cocoa solids, Edible
vegetable fats <5% (Sal fat, Palm oil), Dextrose, Emulsifier (INS 322); Centre filling
(65%): Liquid glucose, Peanuts (21%), Sugar, Hydrogenated vegetable oil (Palm oil), Milk
solids, Iodized salt, Cocoa butter, Cocoa solids, Edible vegetable fats <5% (Sal fat, Palm
oil), Dextrose, Emulsifier (INS 322). CONTAINS VEGETABLE FAT IN ADDITION TO COCOA BUTTER.
Contains Milk, Peanut and Soy. May contain tree nut.""",

    "ice cream": """Ingredients: Milk, sugar, cream, skimmed milk powder, liquid glucose, edible vegetable fat,
emulsifier (INS 471), stabilizers (INS 412, INS 407, INS 410), permitted natural colour (INS 160b),
added flavour (nature identical flavouring substances), mono and diglycerides of fatty acids, egg yolk powder.
Contains: milk, egg. May contain traces of nuts, wheat and soy.
Nutrition facts Amount per 100 g: Energy 207 kcal, Total fat 11 g
Store at -18 C""",
}

# Peanut chocolate bar label as the app's OCR read it (colon after "(35%)" lost, comma after the
# bracket lost, "Sal fat" joined to "Salfat"); Soy on the same line, as printed on the pack.
# NOTE: reconstructed from the label photo + the noise described; replace with the raw text from the
# app's "Raw OCR text (for debugging)" box if it differs.
PHOTO_OCR = """INGREDIENTS
Milk chocolate coating (35%) Sugar, Milk solids, Cocoa butter, Cocoa solids, Edible
vegetable fats <5% (Salfat, Palm oil) Dextrose, Emulsifier (INS 322); Centre filling (65%):
Liquid glucose, Peanuts (21%), Sugar, Hydrogenated vegetable oil (Palm oil), Milk solids,
Iodized salt, Cocoa butter, Cocoa solids, Edible vegetable fats <5% (Sal fat, Palm oil),
Dextrose, Emulsifier (INS 322).CONTAINS VEGETABLE FAT IN ADDITION TO COCOA BUTTER.
Contains Milk, Peanut and Soy. May contain tree nut."""
LABELS["photo ocr"] = PHOTO_OCR
LABELS["photo ocr, split before Soy"] = PHOTO_OCR.replace("Peanut and Soy.", "Peanut and\nSoy.")
LABELS["photo ocr, Soy read as Say"] = PHOTO_OCR.replace("Soy.", "Say.")
LABELS["photo ocr, split + Say + Peanul"] = PHOTO_OCR.replace("Peanut and Soy.", "Peanul and\nSay.")
LABELS["photo ocr, allergen word on next line"] = PHOTO_OCR.replace("Peanut and Soy.", "Peanut\nSoy.")


def names(items):
    return [str(i) + ("*" if getattr(i, "trace", False) else "") for i in items]


def show(title, results, profile):
    print(f"  [{title}] score = {compute_health_score(results)}")
    any_w = False
    for a in getattr(results, "label_declares", ()):
        any_w = True
        print(f"     - Label declares: {a}  (CONFIRMED)")
    for a in getattr(results, "label_possible", ()):
        any_w = True
        print(f"     - Label declares (ambiguous): {a}  (POSSIBLE)")
    for r in results:
        msgs = []
        if r["allergen_warning"]:
            msgs.append("CONFIRMED allergen: " + ", ".join(r["allergen_warning"]))
        if r["possible_allergen_warning"]:
            msgs.append("POSSIBLE allergen: " + ", ".join(r["possible_allergen_warning"]))
        if r["trace_allergen_warning"]:
            msgs.append("TRACE allergen: " + ", ".join(r["trace_allergen_warning"]))
        if r["condition_warning"]:
            msgs.append("condition: " + ", ".join(r["condition_warning"]))
        if msgs:
            any_w = True
            print(f"     - {r['ingredient']:<38s} {' | '.join(msgs)}")
    if not any_w:
        print("     (no warnings)")


def main():
    kb = load_knowledge_base()
    clf = joblib.load(os.path.join(BASE, "models", "ingredient_classifier.pkl"))
    vec = joblib.load(os.path.join(BASE, "models", "tfidf_vectorizer.pkl"))
    peanut_user = {"allergies": ["peanut"], "conditions": []}
    diabetic_user = {"allergies": [], "conditions": ["diabetes"]}

    parsed = {}
    for label, text in LABELS.items():
        items = clean_ingredient_text(text)
        parsed[label] = items
        print("=" * 100)
        print(f"{label.upper()}  ({len(items)} items; * = trace / may-contain)")
        print("  cleaned:", names(items))
        res_p = check_warnings(items, kb, clf, vec, peanut_user)
        res_d = check_warnings(items, kb, clf, vec, diabetic_user)
        print("  sources :", {r["ingredient"]: r["source"] for r in res_p
                              if r["source"] not in ("knowledge_base",)})
        show("peanut-allergic user", res_p, peanut_user)
        show("diabetic user", res_d, diabetic_user)

    # ---- assertions on the behaviours that were specifically asked for ----
    def has(label, name):
        return any(str(i) == name for i in parsed[label])

    assert has("fruit juice", "sodium benzoate"), "sodium benzoate must survive the nutrient filter"
    assert has("bread", "ins 282") or any("282" in i.ins_codes for i in parsed["bread"]), "INS 282"
    assert has("namkeen", "peanuts"), "peanuts"
    assert has("chocolate", "whey protein concentrate"), "whey protein concentrate"
    assert any(i.ins_codes == ["322(i)"] for i in parsed["biscuit"]), "INS 322(i) extracted"
    assert has("instant noodles", "hydrolysed soy protein") or has("instant noodles", "soy protein"), "soy protein"
    assert not any("kcal" in str(i) or "protein 8" in str(i) for p in parsed.values() for i in p), "table rows leaked"
    assert any(i.trace for i in parsed["biscuit"]) and any(str(i) == "nuts" and i.trace for i in parsed["biscuit"]), \
        "may contain traces of nuts -> trace"
    assert not any(str(i) == "mono" for i in parsed["ice cream"]), "'mono and diglycerides' wrongly split"
    # namkeen: confirmed peanut for a peanut-allergic user, big score drop
    res = check_warnings(parsed["namkeen"], kb, clf, vec, peanut_user)
    assert any(r["allergen_warning"] == ["peanut"] for r in res)
    # plain foods are not penalised for allergens the user did not select
    res = check_warnings(parsed["namkeen"], kb, clf, vec, {"allergies": [], "conditions": []})
    assert compute_health_score(res) >= 80
    # ---- peanut chocolate bar (soy + peanut + gluten user) ----
    bar = parsed["peanut chocolate bar"]
    bar_names = [str(i) for i in bar]
    assert "peanuts" in bar_names and "sugar" in bar_names and "peanuts sugar" not in bar_names
    assert not any(n in bar_names for n in ("milk", "peanut", "soy", "milk peanut")), "Contains-line leaked"
    assert not any("contains" in n or "coating" in n or "filling" in n or "<" in n for n in bar_names)
    assert {"sal fat", "palm oil"} <= set(bar_names), "bracket sub-ingredients"
    assert bar.declared_allergens == ["dairy", "peanut", "soy"], bar.declared_allergens
    assert any(str(i) == "tree nut" and i.trace for i in bar), "may contain -> trace"
    triple = {"allergies": ["soy", "peanut", "gluten"], "conditions": []}
    res = check_warnings(bar, kb, clf, vec, triple)
    assert res.label_declares == ["peanut", "soy"], res.label_declares
    seen = [a for r in res for k in ("allergen_warning", "possible_allergen_warning", "trace_allergen_warning")
            for a in (r[k] or [])] + list(res.label_declares)
    assert len(seen) == len(set(seen)), "an allergen was reported twice"
    print("  [soy+peanut+gluten user] Label declares:", ", ".join(res.label_declares),
          "| score =", compute_health_score(res))
    # ---- photo OCR case (colon/comma lost, "Salfat", Soy on the next line) ----
    for key in [k for k in LABELS if k.startswith("photo ocr")]:
        p_items = parsed[key]
        p_names = [str(i) for i in p_items]
        assert p_items.declared_allergens == ["dairy", "peanut", "soy"], (key, p_items.declared_allergens)
        assert not any("coating" in n or "filling" in n for n in p_names), (key, p_names)
        assert "peanuts" in p_names and "sugar" in p_names and "peanuts sugar" not in p_names, (key, p_names)
        assert not any("3593" in n for n in p_names), key
        assert "edible vegetable fats dextrose" not in p_names and "dextrose" in p_names, (key, p_names)
        assert not any(n in p_names for n in ("soy", "say", "milk", "peanut")), "Contains-line leaked"
        assert any(str(i) == "tree nut" and i.trace for i in p_items), "may contain -> trace"
        res = check_warnings(p_items, kb, clf, vec, {"allergies": ["soy", "peanut"], "conditions": []})
        assert res.label_declares == ["peanut", "soy"], (key, res.label_declares)
        assert not any("salfat" in r["ingredient"] and r["source"] == "ml_prediction" for r in res), \
            "salfat must resolve through the KB alias"
        print(f"  [{key}] soy+peanut user: Label declares: {', '.join(res.label_declares)}"
              f" | score = {compute_health_score(res)}")
    # each selected condition is counted once (15 each), not once per ingredient
    res = check_warnings(parsed["photo ocr"], kb, clf, vec, {"allergies": [], "conditions": ["diabetes"]})
    assert sum(1 for r in res if r["condition_warning"]) > 1, "several ingredients hit diabetes"
    none = check_warnings(parsed["photo ocr"], kb, clf, vec, {"allergies": [], "conditions": []})
    assert compute_health_score(none) - compute_health_score(res) == 15, "condition penalty must be applied once"
    full = {"allergies": ["dairy", "soy", "peanut", "sulphites"], "conditions": ["diabetes", "hypertension"]}
    res = check_warnings(parsed["photo ocr"], kb, clf, vec, full)
    assert res.label_declares == ["dairy", "peanut", "soy"], res.label_declares
    print("  [original profile: dairy+soy+peanut+sulphites, diabetes+hypertension] Label declares:",
          ", ".join(res.label_declares), "| score =", compute_health_score(res))
    # statement parsing: the soy fix must not touch ordinary ingredient text
    assert [str(i) for i in clean_ingredient_text("ingredients: say cheese, sugar. Contains milk")] \
        == ["say cheese", "sugar"], "no fuzzy matching on ordinary ingredient text"
    # INS range, heading typo, unrecognised items
    assert not any("3593" in str(i) or "3593" in "".join(i.ins_codes) for i in clean_ingredient_text(
        "Incredients: sugar, emulsifier (INS3593), colour (INS 3593), preservative (211)"))
    junk = check_warnings(clean_ingredient_text("ingredients: sugar, xkq, bcdfg"), kb, clf, vec, triple)
    assert [r["source"] for r in junk if r["ingredient"] in ("xkq", "bcdfg")] == ["unrecognised"] * 2, junk
    assert compute_health_score(junk) == 100, "unrecognised items must not be penalised"
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
