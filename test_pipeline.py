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

    "ice cream": """Ingredients: Milk, sugar, cream, skimmed milk powder, liquid glucose, edible vegetable fat,
emulsifier (INS 471), stabilizers (INS 412, INS 407, INS 410), permitted natural colour (INS 160b),
added flavour (nature identical flavouring substances), mono and diglycerides of fatty acids, egg yolk powder.
Contains: milk, egg. May contain traces of nuts, wheat and soy.
Nutrition facts Amount per 100 g: Energy 207 kcal, Total fat 11 g
Store at -18 C""",
}


def names(items):
    return [str(i) + ("*" if getattr(i, "trace", False) else "") for i in items]


def show(title, results, profile):
    print(f"  [{title}] score = {compute_health_score(results)}")
    any_w = False
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
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
