import re
import string
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Harmful ingredients database
harmful_ingredients = {
    "potassium bromate": "0.1-0.5%",
    "propylparaben": "0.05-0.1%",
    "butylated hydroxyanisole(bha)": "0.01-0.05%",
    "butylated hydroxytoluene(bht)": "0.01-0.05%",
    "titanium dioxide": "0.1-0.5%",
    "aspartame": "0.1-0.5%",
    "azodicarbonamide": "0.1-0.5%",
    "propyl gallate": "0.01-0.05%",
    "sodium benzoate": "0.1-0.25%",
    "methylene chloride": "0.1-0.5%",
    "trichloroethylene": "0.1-0.5%",
    "ethylene dichloride": "0.1-0.5%",
    "sodium nitrite": "0.01-0.015%",
    "lead chromate": "0.1-0.5%",
    "monosodium glutamate(msg)": "0.1-0.5%",
    "high fructose corn syrup": "0.1-0.5%",
    "trans fat": "0.1-0.5%",
    "erythrosine": "0.005-0.01%",
    "allura red ac": "0.01-0.025%",
    "tartrazine (fd&c yellow no.5)": "0.01-0.025%",
    "sunset yellow fcf": "0.01-0.025%",
    "brilliant blue fcf": "0.01-0.025%",
    "indigotine": "0.01-0.025%",
    "fast green fcf": "0.01-0.1%",
}

# Define Fuzzy Variables
toxicity = ctrl.Antecedent(np.arange(0, 11, 1), 'toxicity')
occurrence = ctrl.Antecedent(np.arange(0, 11, 1), 'occurrence')
risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')

toxicity['low'] = fuzz.trimf(toxicity.universe, [0, 0, 3])
toxicity['medium'] = fuzz.trimf(toxicity.universe, [2, 5, 7])
toxicity['high'] = fuzz.trimf(toxicity.universe, [5, 10, 10])

occurrence['low'] = fuzz.trimf(occurrence.universe, [0, 0, 3])
occurrence['medium'] = fuzz.trimf(occurrence.universe, [2, 5, 7])
occurrence['high'] = fuzz.trimf(occurrence.universe, [5, 10, 10])

risk['safe'] = fuzz.trimf(risk.universe, [0, 0, 3])
risk['low'] = fuzz.trimf(risk.universe, [2, 4, 5])
risk['moderate'] = fuzz.trimf(risk.universe, [4, 6, 8])
risk['high'] = fuzz.trimf(risk.universe, [7, 10, 10])

rule1 = ctrl.Rule(toxicity['high'] & occurrence['high'], risk['high'])
rule2 = ctrl.Rule(toxicity['medium'] & occurrence['medium'], risk['moderate'])
rule3 = ctrl.Rule(toxicity['low'], risk['safe'])
rule4 = ctrl.Rule(toxicity['medium'] & occurrence['low'], risk['low'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

# Test dataset (updated to remove INS codes)
test_data = [
    {
        "product_name": "Ketch up",
        "ingredients_text": "TOMATO CONCENTRATE, HIGH FRUCTOSE CORN SYRUP, VINEGAR, CORN SYRUP, SALT, ONION POWDER, SPICE, NATURAL FLAVORS",
        "ground_truth_toxic": ["high fructose corn syrup"],
        "ground_truth_is_safe": False
    },
    {
        "product_name": "Milk Shakes",
        "ingredients_text": "Toned Milk (73.4%), Water, Sugar, Milk Solids, Cocoa Powder (1.5%), Emulsifier & Stabilizer (INS 471, INS 407, INS 412), Acidity Regulator (INS339(ii)), Natural Flavouring Substances, Edible Common Salt, Vitamins (A, D)",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Paasta",
        "ingredients_text": "MODIFIED WHEAT STARCH, DURUM WHEAT, VITAL WHEAT GLUTEN. CONTAINS WHEAT INGREDIENTS",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Peanut Butter",
        "ingredients_text": "PEANUTS, POLYDEXTROSE, MILK PROTEINS (MILK PROTEIN ISOLATE, CALCIUM CASEINATE, WHEY PROTEIN ISOLATE), WATER, ERYTHRITOL, BUTTER (CREAM, NATURAL FLAVOR), VEGETABLE GLYCERIN, NATURAL FLAVORS, CONTAINS LESS THAN 2% OF THE FOLLOWING: SALT, LECITHIN (SUNFLOWER, CANOLA, AND/OR SOY LECITHIN), XANTHAN GUM, BAKING SODA, SUCRALOSE",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Noodles",
        "ingredients_text": "Refined Flour Noodles, Green Capsicum, Palm Oil, French Beans, Carrots, Onions, Garlic, Spring Onion, Celery, Ginger, Tomato Puree, Soya Sauce, Corn Flour, Vinegar, Refined Oil, Salt, Sugar, Dry Red Chilli, Star Anise, Spices and Seasoning, Water",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Kurkure",
        "ingredients_text": "Rice Meal (42.7%), Edible Vegetable Oil (Palmolein Oil), Corn Meal (19.7%), *Spices and Condiments (Onion Powder, Red Chilli Powder, Amchur Powder, Coriander Seed Powder, Garlic Flakes & Powder, Ginger Powder, Black Pepper Powder, Turmeric Powder, Spice Extract, Fenugreek), Gram Meal (3.3%), Salt, Sugar, Tomato powder (0.1%), Citric Acid (330), Dextrose, Milk Solids, Edible Starch",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Boost",
        "ingredients_text": "Cereal Extract (50%) (Barley, Wheat, Millet), Malted Barley (21%), Sugar, Wheat Flour (Atta), Milk Solids (6%), Minerals, Natural Colour (INS 150c), Wheat Gluten, Acidity Regulators (INS 501(ii), INS 500(ii)), Edible Iodised Salt, Cocoa Powder, Vitamins, Nature Identical Flavouring Substances, Soy Protein Isolate",
        "ground_truth_toxic": [],  
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Bourban Biscuit",
        "ingredients_text": "Refined Wheat Flour (Maida), Sugar, Refined Palm Oil, Cocoa Solids (4%), Invert Sugar Syrup, Raising Agents (503(ii), 500(ii)), Iodised Salt, Emulsifier of Vegetable Origin (Soy Lecithin), Contains Added Flavours (Artificial Flavouring Substances - Chocolate, Vanilla), Including 2% as Sprinkled Sugar",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Britania cakes",
        "ingredients_text": "Refined Wheat Flour, Sugar, Eggs, Edible Hydrogenated Vegetable Oil & Palmolein Oil, Cocoa Solids (2%), Milk Solids, Cake Gel Emulsifiers & Stabilizers (471 & 477) and Humectant (420(i), 477) and Humectant (420, 422), Maltose Syrup, Liquid Glucose, Cocoa Butter, Cocoa Solids, Raising Agents (341 & 500), Invert Syrup, Iodised Salt, Preservatives (200 & 282), Baking Powder, Plain Chocolate (0.1493%), Sugar, Cocoa Butter, Cocoa Solids, Emulsifiers (322 & 476), Contains Added Flavor Artificial Vanilla Flavouring Substances, Contains Permitted Natural (150c, 150d) and Synthetic (102 & 122 & 132 & 133) Food Colours and Added Flavor (Nature Identical and Artificial Vanilla & Chocolate) Flavouring Substances",
        "ground_truth_toxic": [],  
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Cheese",
        "ingredients_text": "Cultured Pasteurized Reduced Fat Milk, Salt, Enzymes, Anticake (Potato Starch, Corn Starch, Powdered Cellulose), Natamycin (Mold Inhibitor)",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Dairy Milk",
        "ingredients_text": "Sugar, Milk Solids (25%), Cocoa Butter, Cocoa Solids, Emulsifiers (442, 476), Flavours (Natural, Nature Identical and Artificial (Vanilla) Flavouring Substances)",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Popcorn",
        "ingredients_text": "Brown Sugar (sugar, molasses) Popcorn (corn, coconut oil), Butter (cream, salt), Water, Corn Syrup, Salt, Soy Lecithin, Baking Soda. CONTAINS SOY, MILK",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Rusk",
        "ingredients_text": "Refined Wheat Flour (Maida)(71%), Sugar, Refined Palm Oil and Palmolein, Semolina (Suji) (1.8%), Yeast, Invert Sugar Syrup, Vital Gluten, Iodised Salt, Improvers (1100(i), 1102, 1104), Spices - 0.1% (Green Cardamom Seeds & Its Oil), Wheat Fibre, Emulsifier Blend (481(i), 471, 472e), and Antioxidant (300)",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Venilla Ice Cream",
        "ingredients_text": "Water, Sugar, Glucose Syrup, Brown Sugar, Skimmed Milk, Butterfat, Butter, Egg Yolk, Peanut Oil, Salt, Carrageenan, Vanilla Extract",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Sweet",
        "ingredients_text": "sugar . Dried whole Milk . Cocoa Butter . Cocoa Mass . Tapioca Starch . Emulsifier: Soya Lecithin . Fruit, Plant and Vegetable Extracts (Carrot, Safflower, Spirulina) . Colour: Beetroot Red, Carotenes . Vanilla Flavouring. Milk Chocolate contains Cocoa Solids 30% minimum, Milk Solids 20% minimum",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Pickles",
        "ingredients_text": "FRESH CUCUMBERS, WATER, SALT, DISTILLED VINEGAR, CONTAINS LESS THAN 2% OF DRIED GARLIC, CALCIUM CHLORIDE, SODIUM BENZOATE (TO PRESERVE FLAVOR), SPICE, MUSTARD SEED, NATURAL FLAVOR, DRIED RED PEPPERS, POLYSORBATE 80, OLEORESIN TURMERIC",
        "ground_truth_toxic": ["sodium benzoate"],
        "ground_truth_is_safe": False
    },
    {
        "product_name": "Chewing Gum (Mentos)",
        "ingredients_text": "Xylitol, Chewing Gum Base, Sorbitol, Mannitol, Glycerol, Maltitol Syrup, Natural and Artificial Flavors, Contains Less Than 2 Percent of: Gum Arabic, Color (Titanium Dioxide), Gelatine, Soya Lecithin, Aspartame, Acesulfame K, Green Tea Extract, Carnauba Wax, Sucralose, Sodium Carboxymethylcellulose, BHA to Maintain Freshness, Sucrose Esters of Fatty Acids, Blue 1, Phenylketonurics: Contains Phenylalanine",
        "ground_truth_toxic": ["titanium dioxide", "aspartame"],
        "ground_truth_is_safe": False
    },
    {
        "product_name": "Yogurt",
        "ingredients_text": "Milk & Milk Solids, Sugar, Active Lactic Culture (Lbacillus delbrueckii Subsp. bulgaricus, S.thermophilus), Stabilizing agent (INS 440), Fruit Content (6%) from Blueberry preparation (Blueberry Fruit, Sugar, Water, Stabilizer (INS 440), Acidifying agent (INS 330), Preservative (INS 202)), Colours (INS 122, INS 133), Flavor (Nature - Identical Flavoring Substances - Blueberry)",
        "ground_truth_toxic": [],  
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Puff Corn",
        "ingredients_text": "Cereal Products (67%) (Corn Meal (53%), Rice Meal), Edible Vegetable Oil (Palmolein), +Seasoning (Sugar, Iodised Salt, Maltodextrin, *Spices and Condiments, Milk Solids, Cheese, Salt Substitute (Potassium Chloride), Flavour (Natural and Nature Identical Flavouring Substances), Flavour Enhancers (627, 631), Acidity Regulators (296, 330), Emulsifying & Stabilising Agent (331(iii), 471, 327), Gram Meal, Firming Agent (170(i)), Iodised Salt",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Waffy",
        "ingredients_text": "SUGAR, WHEAT FLOUR, EDIBLE VEGETABLE OIL, STARCH, DEXTROSE, MILK SOLIDS, EMULSIFIER (322), EDIBLE COMMON SALT, RAISING AGENT [500(ii)], DOUGH CONDITIONER (223) AND IMPROVER [1101(i)]",
        "ground_truth_toxic": [],
        "ground_truth_is_safe": True
    },
    {
        "product_name": "Candy",
        "ingredients_text": "Dextrose, Citric Acid, Calcium Stearate, Natural and Artificial Flavors, Colors (Red 40 Lake, Yellow 5 Lake, Yellow 6 Lake, Blue 2 Lake)",
        "ground_truth_toxic": ["allura red ac", "tartrazine (fd&c yellow no.5)", "sunset yellow fcf", "indigotine"],  # Mapped: Red 40 Lake → allura red ac, Yellow 5 Lake → tartrazine, Yellow 6 Lake → sunset yellow fcf, Blue 2 Lake → indigotine
        "ground_truth_is_safe": False
    },
    {
        "product_name": "Bread",
        "ingredients_text": "Enriched Flour (Wheat Flour, Malted Barley Flour, Niacin, Reduced Iron, Thiamin Mononitrate, Riboflavin, Folic Acid), Water, High Fructose Corn Syrup, Yeast, Contains 2% or less of the following: Salt, Soybean Oil, Calcium Sulfate, Monoglycerides, Calcium Propionate (a preservative), Sodium Stearoyl Lactylate, Guar Gum, Datem, Enzymes, Ascorbic Acid (dough conditioner), Azodicarbonamide",
        "ground_truth_toxic": ["azodicarbonamide", "high fructose corn syrup"],
        "ground_truth_is_safe": False
    },
]

# Functions
def extract_numeric_value(toxicity_str):
    numbers = re.findall(r"\d+\.\d+|\d+", toxicity_str)
    if not numbers:
        return 0.0
    numbers = [float(num) for num in numbers]
    if len(numbers) == 2:
        return sum(numbers) / 2
    return numbers[0]

def check_safety(ingredients_text, extracted_ingredients):
    unsafe_ingredients = {}
    for ingredient in harmful_ingredients.keys():
        if ingredient in [ei.lower() for ei in extracted_ingredients]:
            toxicity_str = harmful_ingredients[ingredient]
            toxicity_value = extract_numeric_value(toxicity_str)
            occurrence_level = ingredients_text.lower().count(ingredient) * 2
            risk_simulation.input['toxicity'] = toxicity_value
            risk_simulation.input['occurrence'] = float(occurrence_level)
            risk_simulation.compute()
            risk_level = risk_simulation.output['risk']
            unsafe_ingredients[ingredient] = risk_level
    return unsafe_ingredients

def process_ingredients(ingredients_text):
    ingredients_text = re.sub(r'\n+', ' ', ingredients_text)
    ingredients_text = re.sub(r'\s+', ' ', ingredients_text).strip()

    all_ingredients = []
    exclude_phrases = [
        'ingredient', 'ingredients', 'contains less than 2 percent of',
        'bha to maintain freshness', 'phenylketonurics contains phenylalanine',
        'contains phenylalanine', 'phenylketonurics', 'contains', 'less than 2 percent of'
    ]

    initial_ingredients = ingredients_text.split(',')
    for ingredient in initial_ingredients:
        ingredient = ingredient.strip()
        if not ingredient:
            continue
        if any(exclude_phrase in ingredient.lower() for exclude_phrase in exclude_phrases):
            continue

        bracket_matches = re.findall(r'\(([^)]+)\)', ingredient)
        for match in bracket_matches:
            match = re.sub(r'\n+', ' ', match)
            match = re.sub(r'\s+', ' ', match).strip()
            if ',' in match:
                bracket_items = [item.strip() for item in match.split(',') if item.strip() and not any(exclude_phrase in item.lower() for exclude_phrase in exclude_phrases)]
                all_ingredients.extend(bracket_items)
            else:
                bracket_item = match.strip()
                if bracket_item and not any(exclude_phrase in bracket_item.lower() for exclude_phrase in exclude_phrases):
                    all_ingredients.append(bracket_item)

        cleaned_ingredient = re.sub(r'\([^)]*\)', '', ingredient).strip()
        if cleaned_ingredient and not any(exclude_phrase in cleaned_ingredient.lower() for exclude_phrase in exclude_phrases):
            all_ingredients.append(cleaned_ingredient)

    translator = str.maketrans('', '', string.punctuation)
    ingredients = [ingredient.translate(translator).strip() for ingredient in all_ingredients if ingredient.translate(translator).strip()]
    seen = set()
    ingredients = [ingredient for ingredient in ingredients if not (ingredient in seen or seen.add(ingredient))]
    return ingredients_text, ingredients

def evaluate_toxic_ingredients(pred_toxic, gt_toxic):
    pred_set = set(ingredient.lower() for ingredient, _ in pred_toxic.items())
    gt_set = set(ingredient.lower() for ingredient in gt_toxic)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def calculate_accuracy():
    toxic_precisions = []
    toxic_recalls = []
    toxic_f1s = []
    safety_predictions = []
    safety_ground_truth = []

    for idx, data in enumerate(test_data):
        logger.info(f"Processing test case {idx + 1}: {data['product_name']}")

        # Process ingredients
        ingredients_text, extracted_ingredients = process_ingredients(data["ingredients_text"])

        # Check safety
        unsafe_ingredients = check_safety(ingredients_text, extracted_ingredients)
        toxic_ingredients = {ingredient: risk for ingredient, risk in unsafe_ingredients.items()}
        is_safe = not bool(unsafe_ingredients)

        # Ground truth
        gt_toxic = data["ground_truth_toxic"]
        gt_is_safe = data["ground_truth_is_safe"]

        # Evaluate toxic ingredient identification
        if gt_toxic:
            precision, recall, f1 = evaluate_toxic_ingredients(toxic_ingredients, gt_toxic)
            toxic_precisions.append(precision)
            toxic_recalls.append(recall)
            toxic_f1s.append(f1)
            logger.info(
                f"Test case {idx + 1} - Toxic Ingredients: Precision={precision*100:.2f}%, Recall={recall*100:.2f}%, F1={f1*100:.2f}%"
            )
            logger.info(f"Predicted Toxic: {list(toxic_ingredients.keys())}")
            logger.info(f"Ground Truth Toxic: {gt_toxic}")
        else:
            logger.info(f"Test case {idx + 1} - No toxic ingredients in ground truth")

        # Evaluate safety classification
        safety_predictions.append(is_safe)
        safety_ground_truth.append(gt_is_safe)
        logger.info(
            f"Test case {idx + 1} - Safety: Predicted={'Safe' if is_safe else 'Unsafe'}, "
            f"Ground Truth={'Safe' if gt_is_safe else 'Unsafe'}"
        )

    # Calculate average metrics
    results = {}
    if toxic_precisions:
        results["toxic_precision"] = sum(toxic_precisions) / len(toxic_precisions)
        results["toxic_recall"] = sum(toxic_recalls) / len(toxic_recalls)
        results["toxic_f1"] = sum(toxic_f1s) / len(toxic_f1s)
        logger.info(
            f"Average Toxic Ingredient Metrics: Precision={results['toxic_precision']*100:.2f}%, "
            f"Recall={results['toxic_recall']*100:.2f}%, F1={results['toxic_f1']*100:.2f}%"
        )
    else:
        logger.info("No toxic ingredient metrics calculated")
        results["toxic_precision"] = results["toxic_recall"] = results["toxic_f1"] = None

    if safety_predictions:
        results["safety_accuracy"] = accuracy_score(safety_ground_truth, safety_predictions)
        logger.info(f"Safety Classification Accuracy: {results['safety_accuracy']*100:.2f}%")
    else:
        logger.info("No safety classification metrics calculated")
        results["safety_accuracy"] = None

    return results

if __name__ == "__main__":
    results = calculate_accuracy()
    print("\nFinal Results:")
    if results["toxic_precision"] is not None:
        print(f"Toxic Ingredient Identification:")
        print(f"  Precision: {results['toxic_precision']*100:.2f}%")
        print(f"  Recall: {results['toxic_recall']*100:.2f}%")
        print(f"  F1 Score: {results['toxic_f1']*100:.2f}%")
    else:
        print("Toxic Ingredient Identification: No metrics calculated")
    if results["safety_accuracy"] is not None:
        print(f"Safety Classification Accuracy: {results['safety_accuracy']*100:.2f}%")
    else:
        print("Safety Classification Accuracy: No metrics calculated")