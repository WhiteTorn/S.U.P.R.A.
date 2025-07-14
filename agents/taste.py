import google.generativeai as genai
import json
from PIL import Image
import os
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Internal Core AI Function ---
def _generate_menu_json(content, temperature: float = 0.1):
    """
    Internal function that communicates with the Gemini model.
    It takes prepared content (Image or text) and returns structured JSON.
    """
    generation_config = genai.types.GenerationConfig(temperature=temperature)
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=generation_config
    )
    
    prompt = """
    You are T.A.S.T.E. (Text and Semantic Tactical Extractor).
    Your task is to analyze the provided menu content and extract up to the first 10 dishes.
    Return the output as a clean JSON list of objects.
    Each object must follow this exact structure:
    {
        "id": "dish_xxx",
        "name": "Name of the dish",
        "description": "A brief, appealing description.",
        "price": 0.00,
        "ingredients": ["ingredient1", "ingredient2"],
        "tags": ["tag1", "tag2"],
        "allergens": [],
        "image_url": "https://i.imgur.com/placeholder.jpg"
    }
    - The 'id' should be a unique identifier like 'dish_001'.
    - If a field is not mentioned, generate a plausible one.
    - Extract the price as a number only.
    - Return ONLY the JSON list, not wrapped in markdown.
    """
    
    try:
        response = model.generate_content([prompt, content])
        cleaned_response = response.text.strip().replace("``````", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"An error occurred while communicating with the AI model: {e}")
        return []

# --- Public-Facing Functions ---
def extract_from_file(file_path: str, temperature: float = 0.1):
    """
    Extracts dishes from a file (image or .txt).
    """
    try:
        if file_path.lower().endswith(('.png', '.jpeg', '.jpg')):
            content = Image.open(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print("Error: Unsupported file format.")
            return []
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
        
    return _generate_menu_json(content, temperature)

def extract_from_text(menu_text: str, temperature: float = 0.1):
    """
    Extracts dishes from a raw text string.
    """
    if not menu_text or not isinstance(menu_text, str):
        print("Error: Input must be a non-empty string.")
        return []
        
    return _generate_menu_json(menu_text, temperature)

