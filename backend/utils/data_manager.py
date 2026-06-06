import json
import os

# This finds the absolute path of your project folder automatically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, 'expert_data.json')

# ==================== NEW FACULTY DEFAULT TEMPLATE ====================
# This will be used to create the JSON file if it doesn't exist.
DEFAULT_CONFIG = {
    "weights_manual": {
        "spm_results": 0.25,
        "previous_semester": 0.25,
        "technical_skills": 0.25,
        "aptitude_test": 0.25
    },
    "riasec_weights": {
        "CYBERSECURITY": {"R": 5, "I": 8, "A": 1, "S": 2, "E": 4, "C": 10},
        "CLOUD COMPUTING": {"R": 7, "I": 6, "A": 1, "S": 1, "E": 5, "C": 8},
        "IDEX": {"R": 1, "I": 4, "A": 10, "S": 8, "E": 7, "C": 2},
        "DATA ANALYTICS": {"R": 2, "I": 10, "A": 4, "S": 3, "E": 6, "C": 9},
        "DIGITAL TRANSFORMATION": {"R": 1, "I": 6, "A": 5, "S": 7, "E": 10, "C": 4}
    },
    "criteria": {
        "spm_results": {
            "Mathematics": {"CYBERSECURITY": 10, "CLOUD COMPUTING": 10, "IDEX": 5, "DATA ANALYTICS": 10, "DIGITAL TRANSFORMATION": 5},
            "Additional_Mathematics": {"CYBERSECURITY": 10, "CLOUD COMPUTING": 10, "IDEX": 3, "DATA ANALYTICS": 10, "DIGITAL TRANSFORMATION": 5},
            "English": {"CYBERSECURITY": 7, "CLOUD COMPUTING": 7, "IDEX": 10, "DATA ANALYTICS": 7, "DIGITAL TRANSFORMATION": 10},
            "Physics": {"CYBERSECURITY": 8, "CLOUD COMPUTING": 10, "IDEX": 5, "DATA ANALYTICS": 8, "DIGITAL TRANSFORMATION": 5}
        },
        "previous_semester": {
            "Programming_Fundamentals": {"CYBERSECURITY": 10, "CLOUD COMPUTING": 10, "IDEX": 10, "DATA ANALYTICS": 10, "DIGITAL TRANSFORMATION": 10},
            "Computer_Networking": {"CYBERSECURITY": 10, "CLOUD COMPUTING": 10, "IDEX": 3, "DATA ANALYTICS": 5, "DIGITAL TRANSFORMATION": 8},
            "Database_Systems": {"CYBERSECURITY": 8, "CLOUD COMPUTING": 8, "IDEX": 5, "DATA ANALYTICS": 10, "DIGITAL TRANSFORMATION": 8},
            "Human_Computer_Interaction": {"CYBERSECURITY": 3, "CLOUD COMPUTING": 3, "IDEX": 10, "DATA ANALYTICS": 5, "DIGITAL TRANSFORMATION": 8}
        },
        "technical_skills": {
            "Python_Programming": {"CYBERSECURITY": 8, "CLOUD COMPUTING": 8, "IDEX": 5, "DATA ANALYTICS": 10, "DIGITAL TRANSFORMATION": 7},
            "Cybersecurity_Basics": {"CYBERSECURITY": 10, "CLOUD COMPUTING": 5, "IDEX": 1, "DATA ANALYTICS": 3, "DIGITAL TRANSFORMATION": 5},
            "Cloud_Infrastructure": {"CYBERSECURITY": 7, "CLOUD COMPUTING": 10, "IDEX": 3, "DATA ANALYTICS": 5, "DIGITAL TRANSFORMATION": 8},
            "UI_UX_Design": {"CYBERSECURITY": 1, "CLOUD COMPUTING": 1, "IDEX": 10, "DATA ANALYTICS": 5, "DIGITAL TRANSFORMATION": 8}
        }
    },
    "mcdm": {
        "qrof": {
            "matrix": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
        },
        "bwm": {
            "best_criteria": "technical_skills",
            "worst_criteria": "aptitude_test",
            "best_vectors": [1.0, 1.0, 1.0, 1.0],
            "worst_vectors": [1.0, 1.0, 1.0, 1.0]
        },
        "swara": {
            "rank_order": ["technical_skills", "previous_semester", "spm_results", "aptitude_test"],
            "comparative_scores": [0.1, 0.1, 0.1]
        }
    }
}

def load_data():
    """Loads expert data from JSON. If file is missing, creates it using DEFAULT_CONFIG."""
    if not os.path.exists(CONFIG_FILE):
        print(f"File {CONFIG_FILE} not found. Creating a new one with new faculty specializations...")
        save_data(DEFAULT_CONFIG)
        return DEFAULT_CONFIG
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            return data
    except (json.JSONDecodeError, IOError):
        print("Error reading JSON file. Falling back to defaults.")
        return DEFAULT_CONFIG

def save_data(data):
    """Writes the updated configuration to the JSON file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Error saving data to {CONFIG_FILE}: {e}")