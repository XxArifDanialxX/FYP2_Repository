import os
import json
import traceback
import numpy as np
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai  # 2026 SDK Standard
from tavily import TavilyClient

# Internal logic imports
from utils.data_manager import load_data, save_data
from utils.mcdm_logic import QROFAHP, BWM_VIKOR, SWARA_MOORA, CRITIC_EDAS

app = Flask(__name__)
CORS(app)

# ==================== CONFIGURATION ====================
CRITERIA_KEYS = ["spm_results", "previous_semester", "technical_skills", "aptitude_test"]
SPEC_NAMES = [
    'CYBERSECURITY', 
    'CLOUD COMPUTING', 
    'IDEX', 
    'DATA ANALYTICS', 
    'DIGITAL TRANSFORMATION'
]

# API Keys (Set as environment variables on Render)
GENAI_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")

client = genai.Client(api_key=GENAI_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)
MODEL_ID = "gemini-2.5-flash" 

# ==================== HELPERS ====================

def calc_student_performance_score(scores, weights_map, is_skill=False):
    """Converts raw grades or 1-10 skills into a weighted 0-100 score."""
    total_possible = 0
    weighted_score = 0
    for subject, importance in weights_map.items():
        raw_score = scores.get(subject, 0)
        # Normalize skill ratings (1-10) to percentage (0-100)
        # Formula: (x-1)/9 * 100
        normalized = (raw_score - 1) * (11.11) if is_skill and raw_score > 0 else raw_score
        weighted_score += normalized * importance
        total_possible += 100 * importance
    return (weighted_score / total_possible * 100) if total_possible > 0 else 0

def clean_json_string(text):
    """Robust extractor to find JSON blocks within AI conversational text."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return None
    except:
        return None

# ==================== AGENT A: THE MULTIDIMENSIONAL ORCHESTRATOR ====================

def get_agent_orchestration_decision(student_scores, expert_db):
    """
    AGENT A: Analyzes Data Health across 3 dimensions:
    1. Statistical Variance (Numerical consistency)
    2. Passion-Skill Alignment (RIASEC vs Technical Proficiency)
    3. Expert Maturity (Configuration depth)
    """
    # 1. Calculate Grade Variance
    grades = list(student_scores['spm'].values()) + list(student_scores['uni'].values())
    grade_variance = float(np.var(grades)) if grades else 0

    # 2. Calculate Alignment (Are they good at what they enjoy?)
    avg_skill = sum(student_scores['skills'].values()) / len(student_scores['skills'])
    # Find top RIASEC trait
    top_trait = max(student_scores['riasec'], key=student_scores['riasec'].get)
    alignment = "High" if (avg_skill > 7 and student_scores['riasec'][top_trait] > 10) else "Moderate"

    # 3. Check Expert Maturity
    matrix = expert_db['mcdm']['qrof']['matrix']
    expert_maturity = "Low" if all(all(cell == 1 for cell in row) for row in matrix) else "High"

    prompt = f"""
    You are the MCDM Orchestrator AI. You must decide Trust Weights for 4 algorithms.
    
    STUDENT PROFILE:
    - Grade Variance: {grade_variance:.2f}
    - Interest-Skill Alignment: {alignment}
    - Expert Configuration Maturity: {expert_maturity}

    ALGORITHMS TO WEIGHT:
    - qrof: (Subjective: Expert Hesitancy)
    - bwm: (Subjective: Logical Consistency)
    - swara: (Subjective: Human Intuition)
    - critic: (Objective: Data Variance)

    RULES:
    1. Total weights must sum to 1.0.
    2. If Expert Maturity is LOW, trust 'critic' 0.4.
    3. If Alignment is HIGH and Variance is HIGH, trust 'qrof' and 'swara' more (Specialist student).
    4. If Variance is LOW and Alignment is LOW, trust 'critic' and 'bwm' more (Needs objective guidance).

    RETURN ONLY RAW JSON:
    {{
        "reasoning": "A technical justification sentence.",
        "trust_weights": {{"qrof": 0.25, "bwm": 0.25, "swara": 0.25, "critic": 0.25}}
    }}
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        data = clean_json_string(response.text)
        if data and "trust_weights" in data:
            return data
        raise ValueError("Invalid AI Logic")
    except:
        # Static Fallback based on Variance if AI fails
        fall_weights = {"qrof": 0.2, "bwm": 0.2, "swara": 0.2, "critic": 0.4} if grade_variance > 25 else {"qrof": 0.25, "bwm": 0.25, "swara": 0.25, "critic": 0.25}
        return {
            "reasoning": "Utilizing statistical variance-adjusted consensus for mathematical stability.",
            "trust_weights": fall_weights
        }

# ==================== AGENT B: EXCELLENCE ROADMAP ====================

def get_career_insight(top_spec, weak_skills):
    """
    AGENT B: Performs live research on 2026 Malaysian job trends 
    to create a personalized roadmap.
    """
    try:
        search_query = f"trending technical skills and junior job market for {top_spec} in Malaysia 2026"
        search_result = tavily.search(query=search_query, max_results=3)
        context = search_result['results']
    except:
        context = "Market data currently unavailable."

    prompt = f"""
    Role: Industry Strategist. Spec: {top_spec}. Student Weaknesses: {weak_skills}. Context: {context}
    
    TASK: Generate a 6-month Excellence Roadmap in JSON (make it short sentences only):
    {{
        "core_knowledge": ["4 Theoretical concepts"],
        "technical_skills": ["4 Software tools for 2026"],
        "gap_bridging": "2 paragraphs on transforming {weak_skills} into strengths for {top_spec}.",
        "certifications": ["3 Specific industry certificates"],
        "project_idea": "Title and 2-sentence technical description."
    }}
    RETURN ONLY JSON.
    """
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        data = clean_json_string(response.text)
        if data: return data
        raise ValueError("AI data empty")
    except:
        return {
            "core_knowledge": [f"Advanced {top_spec} Theory", "Systems Design", "Security & Privacy", "Logic Optimization"],
            "technical_skills": ["Python / C++", "Cloud Infrastructure", "SQL Mastery", "Git & CI/CD"],
            "gap_bridging": f"To overcome your weaknesses in {weak_skills}, begin building small projects that solve real-world problems in the {top_spec} domain.",
            "certifications": [f"Professional {top_spec} Associate", "Cloud Solutions Architect", "Project Management"],
            "project_idea": f"Advanced {top_spec} Prototype: Build a scalable application that demonstrates mastery of current industry stacks."
        }

# ==================== CORE ROUTES ====================

@app.route('/api/get_data', methods=['GET'])
def get_data():
    return jsonify(load_data())

@app.route('/api/save_data', methods=['POST'])
def save():
    save_data(request.json)
    return jsonify({"status": "success"})

@app.route('/api/calculate_weights', methods=['GET'])
def calc_weights():
    data = load_data()
    # Default fallbacks
    results = {
        'qrof': [0.25, 0.25, 0.25, 0.25], 
        'bwm': [0.25, 0.25, 0.25, 0.25], 
        'swara': {k: 0.25 for k in CRITERIA_KEYS}
    }

    try:
        # --- 1. QROF Calculation ---
        if 'qrof' in data['mcdm']:
            matrix = [[float(cell) for cell in row] for row in data['mcdm']['qrof']['matrix']]
            results['qrof'] = [round(float(x), 4) for x in QROFAHP().calculate_weights(matrix)]

        # --- 2. BWM Calculation ---
        if 'bwm' in data['mcdm']:
            bd = data['mcdm']['bwm']
            # Map the name (e.g. 'spm_results') to its index (0-3)
            k_map = {k: i for i, k in enumerate(CRITERIA_KEYS)}
            
            best_idx = k_map.get(bd['best_criteria'], 0)
            worst_idx = k_map.get(bd['worst_criteria'], 3)
            best_v = [float(x) for x in bd['best_vectors']]
            worst_v = [float(x) for x in bd['worst_vectors']]

            # Only calculate if vectors are configured (not all 1.0)
            if any(v > 1 for v in best_v):
                w_bwm = BWM_VIKOR().solve_bwm_weights(best_idx, worst_idx, best_v, worst_v)
                results['bwm'] = [round(float(x), 4) for x in w_bwm]

        # --- 3. SWARA Calculation ---
        if 'swara' in data['mcdm']:
            sd = data['mcdm']['swara']
            if sd['rank_order'] and len(sd.get('comparative_scores', [])) > 0:
                comps = [float(x) for x in sd['comparative_scores']]
                w_dict = SWARA_MOORA().calculate_swara_weights(sd['rank_order'], comps)
                # Ensure we return all 4 keys even if some are missing
                results['swara'] = {k: round(float(w_dict.get(k, 0.25)), 4) for k in CRITERIA_KEYS}

    except Exception:
        # Log the error so you can see it in VS Code Terminal
        traceback.print_exc()
        
    return jsonify(results)

@app.route('/api/process_recommendation', methods=['POST'])
def process_recommendation():
    try:
        req = request.json
        scores = req['student_scores']
        db = load_data()
        crit_db = db.get('criteria', {})

        # 1. AI Decision (Agent A)
        orch = get_agent_orchestration_decision(scores, db)
        tw = orch['trust_weights']

        # 2. Build Matrix
        mat = np.zeros((5, 4))
        for i, s in enumerate(SPEC_NAMES):
            mat[i,0] = calc_student_performance_score(scores['spm'], {k: int(v.get(s,5)) for k,v in crit_db.get('spm_results',{}).items()})
            mat[i,1] = calc_student_performance_score(scores['uni'], {k: int(v.get(s,5)) for k,v in crit_db.get('previous_semester',{}).items()})
            mat[i,2] = calc_student_performance_score(scores['skills'], {k: int(v.get(s,5)) for k,v in crit_db.get('technical_skills',{}).items()}, is_skill=True)
            r_w = db.get('riasec_weights', {}).get(s, {})
            apt_val = sum(scores['riasec'].get(t,0)*r_w.get(t,0) for t in ['R','I','A','S','E','C'])
            max_apt = sum(abs(v)*2 for v in r_w.values()) or 1
            mat[i,3] = max(0, min(100, (apt_val+max_apt)/(2*max_apt)*100))

        # 3. RUN ALL MATH ENGINES (NO SHORTCUTS)
        # QROF
        q_w = QROFAHP().calculate_weights(db['mcdm']['qrof']['matrix'])
        q_sc = QROFAHP().calculate_scores(mat, q_w)
        
        # BWM
        bd = db['mcdm']['bwm']
        km = {k: idx for idx, k in enumerate(CRITERIA_KEYS)}
        b_w = BWM_VIKOR().solve_bwm_weights(km[bd['best_criteria']], km[bd['worst_criteria']], [float(x) for x in bd['best_vectors']], [float(x) for x in bd['worst_vectors']])
        b_sc = BWM_VIKOR().calculate_vikor(mat, b_w)

        # SWARA
        sd = db['mcdm']['swara']
        s_w_dict = SWARA_MOORA().calculate_swara_weights(sd['rank_order'], [float(x) for x in sd.get('comparative_scores', [])])
        s_w_arr = np.array([s_w_dict.get(k, 0.25) for k in CRITERIA_KEYS])
        s_sc = SWARA_MOORA().calculate_moora(mat, s_w_arr)

        # CRITIC
        c_sc, _ = CRITIC_EDAS().execute(mat)

        # 4. Final AI-Weighted Consensus
        con_dict = {}
        for i, name in enumerate(SPEC_NAMES):
            weighted = (q_sc[i]*tw['qrof']) + (b_sc[i]*tw['bwm']) + (s_sc[i]*tw['swara']) + (c_sc[i]*tw['critic'])
            con_dict[name] = round(float(weighted), 1)
        
        consensus_final = sorted([{"spec": k, "score": v} for k, v in con_dict.items()], key=lambda x: x['score'], reverse=True)
        for i, x in enumerate(consensus_final, 1): x['rank'] = i

        # 5. Agent B and Packaging
        weak = [k.replace('_',' ') for k, v in scores['skills'].items() if v < 6]
        roadmap = get_career_insight(consensus_final[0]['spec'], weak)

        def pack(sc):
            d = {SPEC_NAMES[j]: sc[j] for j in range(5)}
            return [{"spec": k, "score": round(v,1), "rank": j+1} for j, (k, v) in enumerate(sorted(d.items(), key=lambda x: x[1], reverse=True))]

        return jsonify({
            "consensus": consensus_final,
            "method_results": {
                "q-ROF-AHP": pack(q_sc), "BWM+VIKOR": pack(b_sc), 
                "SWARA+MOORA": pack(s_sc), "CRITIC+EDAS": pack(c_sc)
            },
            "agent_reasoning": orch['reasoning'],
            "career_roadmap": roadmap
        })
    except:
        traceback.print_exc()
        return jsonify({"error": "Math Error"}), 500

if __name__ == '__main__':
    app.run(port=5001, debug=True)