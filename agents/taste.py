import google.generativeai as genai
import json
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure the API key from the environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

def extract_menu_dishes(file_path: str, temperature: float = 0.1):
    """
    Universal T.A.S.T.E. function to extract dishes from a menu file.

    Args:
        file_path (str): The path to the image (.png, .jpg) or text (.txt) file.
        temperature (float): The creativity level for the model. Defaults to 0.1.

    Returns:
        list: A list of dish dictionaries, or an empty list if an error occurs.
    """
    # 1. Initialize the model with specific generation config
    generation_config = genai.types.GenerationConfig(temperature=temperature)
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=generation_config
    )
    
    # 2. Prepare the input content (image or text)
    try:
        if file_path.lower().endswith(('.png', '.jpeg', '.jpg')):
            content = Image.open(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print("Error: Unsupported file format. Please use an image or .txt file.")
            return []
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

    # 3. Define the master prompt
    prompt = """
    You are T.A.S.T.E. (Text and Semantic Tactical Extractor).
    Your task is to analyze the provided menu and extract up to the first 10 dishes.
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
    - If a field is not mentioned, generate a plausible one based on the dish name.
    - Extract the price as a number only.
    - 'image_url' can be a placeholder.
    - Return ONLY the JSON list and nothing else. Do not wrap it in markdown.
    """

    # 4. Generate content and parse the response
    try:
        response = model.generate_content([prompt, content])
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("``````", "")
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"An error occurred while communicating with the AI model: {e}")
        return []
