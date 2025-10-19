from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from datetime import timedelta, datetime
from PIL import Image
import io
import json
import base64
import pytesseract
import pandas as pd
from openpyxl import load_workbook
from sqlalchemy import desc
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import numpy as np
import re
import string

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load harmful ingredients from Excel
try:
    harmful_ingredients_file = "harmful_ingredients.xlsx"
    wb = load_workbook(harmful_ingredients_file)
    sheet = wb.active
    harmful_ingredients = {}
    for row in sheet.iter_rows(min_row=2, values_only=True):
        ingredient, toxicity_level = row
        if ingredient is not None and isinstance(ingredient, str) and ingredient.strip():
            harmful_ingredients[ingredient.lower()] = toxicity_level
        else:
            logger.warning(f"Skipping invalid ingredient entry: {row}")
except Exception as e:
    harmful_ingredients = {}
    logger.error(f"Failed to load harmful ingredients file: {e}")

# Define toxic agents and their health risks (causes)
TOXIC_AGENTS_CAUSES = {
    "aspartame": "May cause headaches, dizziness, or allergic reactions in sensitive individuals.",
    "sodium benzoate": "Potential to trigger asthma attacks or hyperactivity in children when combined with certain food colorings.",
    "monosodium glutamate": "Can cause headaches, flushing, or sweating in some individuals (Chinese Restaurant Syndrome).",
    "potassium bromate": "Linked to kidney and thyroid tumors in animal studies; banned in some countries.",
    "butylated hydroxyanisole": "Possible carcinogen; may cause stomach tumors in high doses.",
    "titanium dioxide": "Potential risk of gut inflammation and increased cancer risk with prolonged exposure; banned in some regions."
}

# Define Fuzzy Variables
toxicity = ctrl.Antecedent(np.arange(0, 11, 1), 'toxicity')
occurrence = ctrl.Antecedent(np.arange(0, 11, 1), 'occurrence')
risk = ctrl.Consequent(np.arange(0, 11, 1), 'risk')

# Membership functions
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

# Define Fuzzy Rules
rule1 = ctrl.Rule(toxicity['high'] & occurrence['high'], risk['high'])
rule2 = ctrl.Rule(toxicity['medium'] & occurrence['medium'], risk['moderate'])
rule3 = ctrl.Rule(toxicity['low'], risk['safe'])
rule4 = ctrl.Rule(toxicity['medium'] & occurrence['low'], risk['low'])

# Create Fuzzy Inference System
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
risk_simulation = ctrl.ControlSystemSimulation(risk_ctrl)

# List of common food-related keywords
FOOD_KEYWORDS = [
    'sugar', 'salt', 'flour', 'oil', 'butter', 'milk', 'egg', 'cheese', 'wheat',
    'rice', 'corn', 'soy', 'yeast', 'honey', 'vinegar', 'spice', 'herb', 'extract',
    'flavor', 'preservative', 'color', 'acid', 'syrup', 'starch', 'protein', 'fat',
    'water', 'juice', 'fruit', 'vegetable', 'grain', 'nut', 'seed',
    'sodium', 'calcium', 'potassium', 'gum', 'lecithin', 'benzoate', 'citric'
]

# Define User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    name = db.Column(db.String(120), nullable=False)

    def __repr__(self):
        return f'<User {self.email}>'

# Define Product model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    is_safe = db.Column(db.Boolean, nullable=False)
    scan_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# Fuzzy logic implementation
def extract_numeric_value(toxicity_str):
    """
    Extracts a numeric value from a string.
    If it's a range like '0.1-0.5%', it returns the average value.
    """
    numbers = re.findall(r"\d+\.\d+|\d+", toxicity_str)  # Find all numbers (integer or decimal)
    if not numbers:
        return 0.0  # Default value if no number is found
    numbers = [float(num) for num in numbers]  # Convert to float
    if len(numbers) == 2:  # If it's a range (e.g., 0.1-0.5)
        return sum(numbers) / 2  # Return the average
    return numbers[0]  # If it's a single number, return it

def check_safety(ingredients_text, extracted_ingredients):
    """
    Checks the safety of ingredients using the Fuzzy Inference System (FIS).
    Returns unsafe ingredients with their risk levels and health risks (causes).
    """
    unsafe_ingredients = {}

    for ingredient in harmful_ingredients.keys():
        # Check if the harmful ingredient is in the extracted ingredients list
        if ingredient in [ei.lower() for ei in extracted_ingredients]:
            toxicity_str = harmful_ingredients[ingredient]  # Get toxicity level as string
            toxicity_value = extract_numeric_value(str(toxicity_str))  # Convert to float
            occurrence_level = ingredients_text.lower().count(ingredient) * 2  # Frequency weight

            risk_simulation.input['toxicity'] = toxicity_value
            risk_simulation.input['occurrence'] = float(occurrence_level)
            risk_simulation.compute()

            risk_level = risk_simulation.output['risk']
            # Use TOXIC_AGENTS_CAUSES for health risks, default to generic message if not found
            unsafe_ingredients[ingredient] = {
                "risk_level": risk_level,
                "causes": TOXIC_AGENTS_CAUSES.get(ingredient, "Health risks not specified for this ingredient.")
            }

    return unsafe_ingredients

def is_food_ingredient_text(text):
    """
    Checks if the extracted text likely contains food ingredients.
    Returns True if food-related, False otherwise.
    """
    if not text or len(text.strip()) == 0:
        return False
    
    text_lower = text.lower()
    # Check for presence of food-related keywords
    keyword_count = sum(1 for keyword in FOOD_KEYWORDS if keyword in text_lower)
    
    # Additional heuristic: ingredient lists often have commas or line breaks
    has_list_structure = ',' in text or '\n' in text
    
    # Consider text as food-related if it has multiple keywords or list-like structure
    return keyword_count >= 2 or (keyword_count >= 1 and has_list_structure)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid email or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        existing_user = User.query.filter_by(email=request.form['email']).first()
        if existing_user:
            flash('Email already in use. Please log in or use a different email.')
            return redirect(url_for('login'))
        
        if request.form['password'] != request.form['confirm_password']:
            flash('Passwords do not match. Please try again.')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(request.form['password'])
        new_user = User(email=request.form['email'], password=hashed_password, name=request.form['name'])
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful, please login')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/home')
@login_required
def home():
    return render_template('home.html', name=current_user.name)

@app.route('/logout')
@login_required
def logout():
    logger.info(f"Session data before logout: {session}")
    logout_user()
    logger.info(f"Session data after logout: {session}")
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/scanner')
@login_required
def scanner_home():
    return render_template('scanner.html')

@app.route('/scanner1')
@login_required
def scanner_home1():
    return render_template('scanner1.html')

@app.route('/upload_and_process', methods=['POST'])
def upload_and_process():
    image_data = request.json.get('image')
    if image_data:
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]  # Remove data URL part
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(image)
            
            # Validate if the text is food-related
            if not is_food_ingredient_text(text):
                logger.warning(f"Non-food-related text detected: {text[:100]}...")
                return jsonify({
                    "error": "The uploaded image does not contain food ingredients. Please upload or scan an image of a food product ingredient list."
                }), 400
            
            return jsonify({"text": text})
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return jsonify({"error": "OCR processing failed"}), 500
    return jsonify({"error": "No image data provided"}), 400

@app.route('/process_scan', methods=['POST'])
def process_scan():
    data = request.json
    ingredients_text = data.get('ingredients', '').lower()
    
    # Normalize the input: replace newlines with spaces and clean up extra spaces
    ingredients_text = re.sub(r'\n+', ' ', ingredients_text)
    ingredients_text = re.sub(r'\s+', ' ', ingredients_text).strip()
    
    # Initialize list to store all ingredients
    all_ingredients = []
    
    # List of phrases to exclude (non-ingredients)
    exclude_phrases = [
        'ingredient', 'ingredients', 'contains less than 2 percent of',
        'bha to maintain freshness', 'phenylketonurics contains phenylalanine',
        'contains phenylalanine', 'phenylketonurics', 'contains', 'less than 2 percent of'
    ]
    
    # Split by commas to get initial ingredients
    initial_ingredients = ingredients_text.split(',')
    
    # Process each ingredient
    for ingredient in initial_ingredients:
        ingredient = ingredient.strip()
        if not ingredient:
            continue
            
        # Skip if the ingredient matches any excluded phrase
        if any(exclude_phrase in ingredient for exclude_phrase in exclude_phrases):
            continue
            
        # Extract content inside brackets
        bracket_matches = re.findall(r'\(([^)]+)\)', ingredient)
        for match in bracket_matches:
            # Clean the bracket content: replace newlines and extra spaces
            match = re.sub(r'\n+', ' ', match)
            match = re.sub(r'\s+', ' ', match).strip()
            
            # Check if the bracket content contains a comma
            if ',' in match:
                # Split bracket content by commas if it contains multiple items
                bracket_items = [item.strip() for item in match.split(',') if item.strip() and not any(exclude_phrase in item for exclude_phrase in exclude_phrases)]
                all_ingredients.extend(bracket_items)
            else:
                # Treat the bracket content as a single ingredient if no comma is present
                bracket_item = match.strip()
                if bracket_item and not any(exclude_phrase in bracket_item for exclude_phrase in exclude_phrases):
                    all_ingredients.append(bracket_item)
        
        # Clean the ingredient outside brackets
        # Remove brackets and their content
        cleaned_ingredient = re.sub(r'\([^)]*\)', '', ingredient).strip()
        if cleaned_ingredient and not any(exclude_phrase in cleaned_ingredient for exclude_phrase in exclude_phrases):
            all_ingredients.append(cleaned_ingredient)
    
    # Remove punctuation from each ingredient and filter out empty results
    translator = str.maketrans('', '', string.punctuation)
    ingredients = [ingredient.translate(translator).strip() for ingredient in all_ingredients if ingredient.translate(translator).strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    ingredients = [ingredient for ingredient in ingredients if not (ingredient in seen or seen.add(ingredient))]
    
    unsafe_ingredients = check_safety(ingredients_text, ingredients)
    
    causes = []
    for ingredient, info in unsafe_ingredients.items():
        ingredient_info = {
            "name": ingredient,
            "risk_level": round(info["risk_level"], 2),
            "causes": info["causes"]
        }
        causes.append(ingredient_info)

    safe = not bool(unsafe_ingredients)
    product_name = request.json.get('product_name', 'Unknown Product')

    new_product = Product(name=product_name, is_safe=safe)

    try:
        db.session.add(new_product)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving product to database: {e}")

    response_data = {
        'safe': safe,
        'causes': causes,
        'full_text': ingredients_text
    }

    logger.info(f"Processed scan response: {response_data}")  # Logging for debugging

    return jsonify(response_data)

@app.route('/result')
def result():
    logger.info(f"Received data in result route: {request.args}")
    safe = request.args.get('safe', 'true') == 'true'
    causes = json.loads(request.args.get('causes', '[]'))
    full_text = request.args.get('full_text', '')
    product_name = request.args.get('product_name', 'Unknown Product')

    return render_template('result.html', safe=safe, causes=causes, full_text=full_text, product_name=product_name)

@app.route('/recent_scans')
@login_required
def recent_scans():
    # Fetch last 5 scans ordered by date descending
    recent_products = Product.query.order_by(desc(Product.scan_date)).limit(5).all()
    return jsonify([{'name': product.name, 'is_safe': product.is_safe, 'scan_date': product.scan_date.strftime('%Y-%m-%d %H:%M')} for product in recent_products])

@app.route('/bmi_calculator', methods=['GET'])
@login_required
def bmi_calculator():
    return render_template('bmi_form_personal.html')

@app.route('/bmi_meals', methods=['POST'])
@login_required
def bmi_meals():
    session['weight'] = float(request.form['weight'])
    session['height'] = float(request.form['height']) / 100  # Convert cm to meters
    session['age'] = int(request.form['age'])
    session['gender'] = request.form['gender']
    return render_template('bmi_form_meals.html')

@app.route('/bmi_result', methods=['POST'])
@login_required
def bmi_result():
    # Retrieve personal details from session
    weight = session.get('weight')
    height = session.get('height')
    age = session.get('age')
    gender = session.get('gender')

    # Calculate BMI
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        category = "Underweight"
        base_calorie_range = "2000-2500 kcal"
    elif 18.5 <= bmi < 25:
        category = "Normal weight"
        base_calorie_range = "1800-2400 kcal"
    elif 25 <= bmi < 30:
        category = "Overweight"
        base_calorie_range = "1500-2000 kcal"
    else:
        category = "Obese"
        base_calorie_range = "1200-1800 kcal"

    # Calculate BMR
    if gender == "male":
        bmr = 10 * weight + 6.25 * (height * 100) - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * (height * 100) - 5 * age - 161
    recommended_calories = bmr * 1.2  # Sedentary activity level

    # Calculate total calorie intake from meals
    total_calories = 0

    # Morning meal (Tiffin)
    if 'morning' in request.form:
        morning_items = request.form.getlist('morning')
        if 'idli' in morning_items:
            qty = float(request.form.get('morning_idli_qty', 0))
            total_calories += (70 / 2) * qty  # 70 kcal per 2 idlis
        if 'dosa' in morning_items:
            qty = float(request.form.get('morning_dosa_qty', 0))
            total_calories += 120 * qty  # 120 kcal per dosa
        if 'poha' in morning_items:
            qty = float(request.form.get('morning_poha_qty', 0))
            total_calories += (130 / 100) * qty  # 130 kcal per 100g
        if 'pongal' in morning_items:
            qty = float(request.form.get('morning_pongal_qty', 0))
            total_calories += (150 / 100) * qty  # 150 kcal per 100g
        if 'vada' in morning_items:
            qty = float(request.form.get('morning_vada_qty', 0))
            total_calories += 100 * qty  # 100 kcal per vada
        if 'poori' in morning_items:
            qty = float(request.form.get('morning_poori_qty', 0))
            total_calories += 110 * qty  # 110 kcal per poori
        if 'chapaathi' in morning_items:
            qty = float(request.form.get('morning_chapaathi_qty', 0))
            total_calories += 80 * qty  # 80 kcal per chapaathi
        if 'tea' in morning_items:
            qty = float(request.form.get('morning_tea_qty', 0))
            total_calories += 30 * qty  # 30 kcal per cup
        if 'coffee' in morning_items:
            qty = float(request.form.get('morning_coffee_qty', 0))
            total_calories += 40 * qty  # 40 kcal per cup
        if 'milk' in morning_items:
            qty = float(request.form.get('morning_milk_qty', 0))
            total_calories += 120 * qty  # 120 kcal per cup
        if 'bread' in morning_items:
            qty = float(request.form.get('morning_bread_qty', 0))
            total_calories += 70 * qty  # 70 kcal per slice
        if 'peanut_butter' in morning_items:
            qty = float(request.form.get('morning_peanut_butter_qty', 0))
            total_calories += 90 * qty  # 90 kcal per tbsp
        if 'jam' in morning_items:
            qty = float(request.form.get('morning_jam_qty', 0))
            total_calories += 50 * qty  # 50 kcal per tbsp

    # Afternoon meal (Lunch)
    if 'afternoon' in request.form:
        afternoon_items = request.form.getlist('afternoon')
        if 'rice_dal' in afternoon_items:
            qty = float(request.form.get('afternoon_rice_dal_qty', 0))
            total_calories += (250 / 200) * qty  # 250 kcal per 200g
        if 'roti_sabzi' in afternoon_items:
            qty = float(request.form.get('afternoon_roti_sabzi_qty', 0))
            total_calories += 240 * qty  # 240 kcal per set
        if 'chicken_curry' in afternoon_items:
            qty = float(request.form.get('afternoon_chicken_curry_qty', 0))
            total_calories += (300 / 150) * qty  # 300 kcal per 150g
        if 'fried_rice' in afternoon_items:
            qty = float(request.form.get('afternoon_fried_rice_qty', 0))
            total_calories += (260 / 200) * qty  # 260 kcal per 200g
        if 'kushka' in afternoon_items:
            qty = float(request.form.get('afternoon_kushka_qty', 0))
            total_calories += (220 / 200) * qty  # 220 kcal per 200g
        if 'lemon_rice' in afternoon_items:
            qty = float(request.form.get('afternoon_lemon_rice_qty', 0))
            total_calories += (210 / 200) * qty  # 210 kcal per 200g
        if 'curd_rice' in afternoon_items:
            qty = float(request.form.get('afternoon_curd_rice_qty', 0))
            total_calories += (200 / 200) * qty  # 200 kcal per 200g
        if 'egg_rice' in afternoon_items:
            qty = float(request.form.get('afternoon_egg_rice_qty', 0))
            total_calories += (250 / 200) * qty  # 250 kcal per 200g
        if 'fish' in afternoon_items:
            qty = float(request.form.get('afternoon_fish_qty', 0))
            total_calories += (280 / 150) * qty  # 280 kcal per 150g
        if 'boiled_egg' in afternoon_items:
            qty = float(request.form.get('afternoon_boiled_egg_qty', 0))
            total_calories += 75 * qty  # 75 kcal per egg
        if 'omelette' in afternoon_items:
            qty = float(request.form.get('afternoon_omelette_qty', 0))
            total_calories += 120 * qty  # 120 kcal per omelette
        if 'sambar' in afternoon_items:
            qty = float(request.form.get('afternoon_sambar_qty', 0))
            total_calories += (150 / 200) * qty  # 150 kcal per 200g

    # Dinner
    if 'dinner' in request.form:
        dinner_items = request.form.getlist('dinner')
        if 'khichdi' in dinner_items:
            qty = float(request.form.get('dinner_khichdi_qty', 0))
            total_calories += (200 / 200) * qty  # 200 kcal per 200g
        if 'grilled_fish' in dinner_items:
            qty = float(request.form.get('dinner_grilled_fish_qty', 0))
            total_calories += (120 / 100) * qty  # 120 kcal per 100g
        if 'vegetable_soup' in dinner_items:
            qty = float(request.form.get('dinner_vegetable_soup_qty', 0))
            total_calories += 50 * qty  # 50 kcal per bowl
        if 'fried_rice' in dinner_items:
            qty = float(request.form.get('dinner_fried_rice_qty', 0))
            total_calories += (260 / 200) * qty  # 260 kcal per 200g
        if 'parotta' in dinner_items:
            qty = float(request.form.get('dinner_parotta_qty', 0))
            total_calories += 150 * qty  # 150 kcal per parotta
        if 'idli' in dinner_items:
            qty = float(request.form.get('dinner_idli_qty', 0))
            total_calories += (70 / 2) * qty  # 70 kcal per 2 idlis
        if 'chapaathi' in dinner_items:
            qty = float(request.form.get('dinner_chapaathi_qty', 0))
            total_calories += 80 * qty  # 80 kcal per chapaathi
        if 'dosa' in dinner_items:
            qty = float(request.form.get('dinner_dosa_qty', 0))
            total_calories += 120 * qty  # 120 kcal per dosa
        if 'chicken' in dinner_items:
            qty = float(request.form.get('dinner_chicken_qty', 0))
            total_calories += (300 / 150) * qty  # 300 kcal per 150g

    # Custom items for Morning
    if 'morning_custom[0][name]' in request.form:
        for index in range(len(request.form.getlist('morning_custom[0][name]'))):
            name = request.form.get(f'morning_custom[{index}][name]')
            qty = float(request.form.get(f'morning_custom[{index}][qty]', 0))
            calories = float(request.form.get(f'morning_custom[{index}][calories]', 100))  # Use submitted calorie value
            if name and qty > 0:
                total_calories += calories * qty

    # Custom items for Afternoon
    if 'afternoon_custom[0][name]' in request.form:
        for index in range(len(request.form.getlist('afternoon_custom[0][name]'))):
            name = request.form.get(f'afternoon_custom[{index}][name]')
            qty = float(request.form.get(f'afternoon_custom[{index}][qty]', 0))
            calories = float(request.form.get(f'afternoon_custom[{index}][calories]', 100))  # Use submitted calorie value
            if name and qty > 0:
                total_calories += calories * qty

    # Custom items for Dinner
    if 'dinner_custom[0][name]' in request.form:
        for index in range(len(request.form.getlist('dinner_custom[0][name]'))):
            name = request.form.get(f'dinner_custom[{index}][name]')
            qty = float(request.form.get(f'dinner_custom[{index}][qty]', 0))
            calories = float(request.form.get(f'dinner_custom[{index}][calories]', 100))  # Use submitted calorie value
            if name and qty > 0:
                total_calories += calories * qty

    # Calculate calorie difference and status
    calorie_difference = total_calories - recommended_calories
    if calorie_difference > 0:
        calorie_status = "Over Intake"
    elif calorie_difference < 0:
        calorie_status = "Less Intake"
    else:
        calorie_status = "Balanced Intake"

    return render_template('bmi_result.html',
                          bmi=round(bmi, 2),
                          category=category,
                          calorie_range=base_calorie_range,
                          total_calories=round(total_calories, 2),
                          recommended_calories=round(recommended_calories, 2),
                          calorie_status=calorie_status,
                          calorie_difference=round(abs(calorie_difference), 2))

if __name__ == '__main__':
    with app.app_context():
        db_path = os.path.join(os.getcwd(), "instance", "app.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Ensure tables are created
        db.create_all()
        print("✅ Database initialized at:", db_path)

    # Use Render’s expected port if available
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
