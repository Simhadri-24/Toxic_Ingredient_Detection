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
import os  # ✅ Added for path management

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# ✅ FIX: Use Render’s writable directory for SQLite
db_path = os.path.join("/var/lib/render", "app.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"

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

# Define Product model
class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    is_safe = db.Column(db.Boolean, nullable=False)
    scan_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

# (All your routes and functions remain unchanged below this line)
# ...
# 👇 Skip unchanged code for brevity
# ...
# (Keep your existing functions and routes here exactly as they are)

if __name__ == '__main__':
    with app.app_context():
        # ✅ Ensure writable directory exists and database is initialized
        os.makedirs("/var/lib/render", exist_ok=True)
        db.create_all()
        print("✅ Database initialized at /var/lib/render/app.db")

    # ✅ Use Render’s assigned port
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
