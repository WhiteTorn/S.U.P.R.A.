import google.generativeai as genai
import json
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

# --- Configuration ---
# Configure the Gemini API key. Replace "YOUR_API_KEY" with your actual key.
# It's recommended to use environment variables for security.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def get_taste_agent_response(file_path):
    """
    Initializes the T.A.S.T.E. agent to extract menu dishes from an image or text file.
    
    Args:
        file_path (str): The path to the image (.png, .jpg) or text (.txt) file.
        
    Returns:
        list: A list of dictionaries, where each dictionary represents a dish.
              Returns an empty list if an error occurs.
    """
    
    # 1. Define the model. 'gemini-1.5-flash' is fast and multimodal.
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 2. Prepare the input content (image or text)
    try:
        if file_path.lower().endswith(('.png', '.jpeg', '.jpg')):
            content = Image.open(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r') as f:
                content = f.read()
        else:
            print("Unsupported file format. Please use an image or .txt file.")
            return []
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

    # 3. Create the master prompt for the T.A.S.T.E. agent
    prompt = """
    You are T.A.S.T.E. (Text and Semantic Tactical Extractor).
    Your task is to analyze the provided menu from an image or text and extract up to the first 10 dishes.
    Return the output as a JSON list of objects.
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
    - The 'id' should be a unique identifier like 'dish_001', 'dish_002'.
    - If a field like 'description' or 'ingredients' is not explicitly mentioned, generate a plausible one based on the dish name.
    - For 'price', extract the number only.
    - 'image_url' can be a placeholder.
    - Return ONLY the JSON list and nothing else.
    """

    # 4. Get the response from the model
    try:
        response = model.generate_content([prompt, content])
        
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace("``````", "")
        
        # 5. Return the structured data
        return json.loads(cleaned_response)
        
    except Exception as e:
        print(f"An error occurred while communicating with the AI model: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    # Replace 'path/to/your/menu.jpg' with the actual file path
    # This can be an image file ('menu.jpg', 'menu.png') or a text file ('menu.txt')
    menu_file = 'path/to/your/menu.jpg' 
    
    print(f"Initializing T.A.S.T.E. to extract data from: {menu_file}")
    
    extracted_dishes = get_taste_agent_response(menu_file)
    
    if extracted_dishes:
        print("\n--- T.A.S.T.E. Extraction Complete ---")
        # Pretty print the JSON output
        print(json.dumps(extracted_dishes, indent=2, ensure_ascii=False))
        print(f"\nSuccessfully extracted {len(extracted_dishes)} dishes.")
