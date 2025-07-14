import os
import json
import asyncio
from typing import Dict, List, Any
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

class SupraSearchEngine:
    """Universal S.U.P.R.A. agent for multimodal restaurant search."""
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = self.model = "gemini-2.0-flash"
        self.restaurant_data = []

    def load_data(self, data_path: str = "data/rests.json"):
        """Loads restaurant data from a JSON file."""
        try:
            with open(data_path, 'r', encoding='utf-8') as file:
                self.restaurant_data = json.load(file)
            print(f"✅ Successfully loaded {len(self.restaurant_data)} restaurants.")
            return True
        except Exception as e:
            print(f"❌ Failed to load data from {data_path}: {e}")
            return False

    # def get_featured_restaurants(self, limit: int = 6) -> List[Dict]:
    #     """Gets featured restaurants, sorted by price range."""
    #     return sorted(
    #         self.restaurant_data, 
    #         key=lambda x: x.get('price_range', 0), 
    #         reverse=True
    #     )[:limit]
        
    def _process_image(self, image_path: str) -> types.Part:
        """Helper to read and prepare an image file for the API."""
        return types.Part.from_uri(
            uri=image_path,
            mime_type='image/jpeg' # Let the API handle detection
        )
        
    async def search(self, query: str = "", image_path: str = "", preferences: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Performs a multimodal search using either text, an image, or both.
        """
        prompt = ""
        contents = []

        try:
            restaurant_data_json = json.dumps(self.restaurant_data, ensure_ascii=False)
            
            if image_path:
                image_part = self._process_image(image_path)
                contents.append(image_part)
                prompt = f"""
                Analyze this food image and find similar dishes in the restaurant database.
                Additional user query: "{query}" if query else "None"
                Return up to {limit} matches.
                """
            else:
                prompt = f"""
                You are a Georgian cuisine expert. Find dishes matching the query: "{query}"
                Return up to {limit} matches.
                """

            if preferences:
                preferences_prompt = f"""
                User Preferences and allergies: "{preferences}"
                """

            full_prompt = f"""
            {prompt}
            
            RESTAURANT DATA:
            {restaurant_data_json}

            INSTRUCTIONS:
            1. Understand the user's intent (taste, price, dietary needs, cuisine type, etc.)
            2. Find the most relevant dishes with detailed restaurant information
            3. Return maximum {limit} results ranked by relevance
            4. Focus on Georgian cuisine authenticity when relevant
            5. Always focus on user preferences and allergies, they are top priority.
            User Preferences are: {preferences_prompt}

            also you should act like the waiters in the restaurant,
            professionally and politely pick the best dishes that user might also like
            and return them with the addition to the main query.
            focus on preferences and allergies user specified in the query.

            you are not allowed to return the same dish more than once.
            and you are not allowed to make mistakes in the data when returning them. you have IDEAL memory and ideal capabilities to return information as it was.

            OUTPUT FORMAT (JSON ONLY):
            {{
                "results": [
                    {{
                        "restaurant_id": "...",
                        "restaurant_name": "...",
                        "dish_name": "...",
                        "dish_price": 0.00,
                    }}
                ]
            }}
            """
            contents.append(full_prompt)

            response = self.client.models.generate_content(
                model = self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                )
            )
            
            return {"status": "success", "data": json.loads(response.text)}

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {"status": "error", "message": str(e)}

