import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()

class ConversationState:
    """Simple conversation state management."""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.selected_dishes: List[Dict[str, Any]] = []  # Current selection
        self.user_preferences: str = ""
        self.initial_query: str = ""
        self.turn_count: int = 0
        self.is_satisfied: bool = False
        
    def add_user_message(self, message: str):
        """Add a user message to conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "turn": self.turn_count
        })
        
    def update_selected_dishes(self, new_dishes: List[Dict[str, Any]]):
        """Update selected dishes with new results from AI."""
        self.selected_dishes = new_dishes
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant", 
            "content": f"Updated selection to {len(new_dishes)} dishes",
            "turn": self.turn_count
        })
            
    def get_conversation_context(self) -> str:
        """Get conversation context for AI."""
        context = f"CONVERSATION TURN: {self.turn_count}\n"
        context += f"USER PREFERENCES: {self.user_preferences}\n"
        
        if self.selected_dishes:
            dishes_info = []
            for i, dish in enumerate(self.selected_dishes, 1):
                dishes_info.append(f"{i}. {dish['dish_name']} from {dish['restaurant_name']} (${dish['dish_price']})")
            context += f"CURRENT SELECTION ({len(self.selected_dishes)} dishes):\n" + "\n".join(dishes_info) + "\n"
        
        if self.conversation_history:
            context += "RECENT CONVERSATION:\n"
            for msg in self.conversation_history[-3:]:
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                    
        return context

class SupraMultiSearchEngine:
    """Simplified conversational S.U.P.R.A. agent - let AI handle everything."""
    
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.restaurant_data = []
        self.conversation = ConversationState()

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

    def _process_image(self, image_path: str) -> types.Part:
        """Helper to read and prepare an image file for the API."""
        # Read local image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        return types.Part.from_bytes(
            data=image_data,
            mime_type='image/jpeg'
        )
        
    def _detect_satisfaction(self, user_input: str) -> bool:
        """Detect if user is satisfied."""
        satisfaction_indicators = [
            "perfect, thank you", "that's perfect", "perfect thanks", 
            "i'm satisfied", "that's all", "i'm done", "that's enough",
            "order these", "i'll order these", "book these"
        ]
        
        user_input_lower = user_input.lower().strip()
        return any(indicator in user_input_lower for indicator in satisfaction_indicators)

    async def chat(self, user_input: str, image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Main chat interface - AI handles all logic including filtering.
        """
        
        if not user_input.strip():
            return {
                "status": "no_response", 
                "message": "No input provided",
                "conversation_complete": self.conversation.is_satisfied
            }
        
        # Check satisfaction
        if self._detect_satisfaction(user_input):
            self.conversation.is_satisfied = True
            return {
                "status": "satisfied",
                "message": "Perfect! Enjoy your meal!",
                "conversation_complete": True,
                "final_selection": {
                    "dishes": self.conversation.selected_dishes,
                    "total_dishes": len(self.conversation.selected_dishes),
                    "total_cost": sum(d["dish_price"] for d in self.conversation.selected_dishes)
                }
            }
        
        try:
            # Handle initial query
            if self.conversation.turn_count == 0:
                self.conversation.initial_query = user_input
            
            # Process request with AI
            self.conversation.add_user_message(user_input)
            result = await self._process_with_ai(user_input, image_path, limit)
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "data": result["data"],
                    "conversation_complete": False,
                    "conversation_state": {
                        "turn_count": self.conversation.turn_count,
                        "total_dishes": len(self.conversation.selected_dishes)
                    }
                }
            else:
                return result
                
        except Exception as e:
            return {"status": "error", "message": str(e), "conversation_complete": False}

    async def _process_with_ai(self, query: str, image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Let AI handle ALL logic - additions, removals, filtering, everything.
        """
        contents = []
        
        try:
            restaurant_data_json = json.dumps(self.restaurant_data, ensure_ascii=False)
            conversation_context = self.conversation.get_conversation_context()
            
            # Handle image if provided
            if image_path:
                image_part = self._process_image(image_path)
                contents.append(image_part)
            
            # Current selection as JSON for AI to work with
            current_selection_json = json.dumps(self.conversation.selected_dishes, ensure_ascii=False)

            full_prompt = f"""
            You are a professional Georgian cuisine expert and waiter with PERFECT MEMORY.
            
            {conversation_context}
            
            USER REQUEST: "{query}"
            
            CURRENT USER SELECTION (what they have now):
            {current_selection_json}
            
            RESTAURANT DATA (available dishes):
            {restaurant_data_json}

            INSTRUCTIONS - Handle ALL operations naturally:
            
            {"🖼️ IMAGE ANALYSIS MODE:" if image_path else ""}
            {"- First, analyze the food image to identify what dish/cuisine it shows" if image_path else ""}
            {"- Then, search the restaurant database for ACTUAL similar dishes" if image_path else ""}
            {"- Return matching dishes from the database, not just description" if image_path else ""}
            {"- Focus on finding real dishes that match what you see in the image" if image_path else ""}
            
            1. UNDERSTAND the user's intent:
               - Adding dishes? ("add", "also", "more", "suggest")
               - Removing/filtering? ("only", "just", "don't want", "remove", "except")
               - Replacing? ("instead", "different")
               - Asking for information? ("show", "what do I have")
               - Image search? (when image provided, find similar dishes in database)
            
            2. HANDLE USER SELECTION DECISIONS:
                - If user says "ავიღებ X" or "X ავიღებ" (I'll take X) = choose ONLY X, remove other similar options
               - If user wants to ADD: keep current dishes + add new ones = - ALWAYS keep dishes user has already chosen (unless they specifically ask to remove)
               - If user asks for new category: show ALL options in that category + keep existing selection
               - If user says "I don't want X": remove X from current selection - Only remove items when user explicitly says "remove X" or "I don't want X"
               - If user has allergies: remove/avoid allergens
               - If IMAGE PROVIDED: search database for dishes similar to what's shown
               - If user chooses 1 item from multiple options (like "საქონლის ხინკალს" from 4 khinkali)
                - Keep the chosen item + remove other similar items from same category
            
            3. SHOW ALL AVAILABLE OPTIONS for what user requests - and after filtering RETURN FINAL COMPLETE SELECTION
               - If user asks for khinkali, show ALL khinkali options available
               - If user asks for drinks, show ALL drink options available  
               - If user asks for meat dishes, show ALL meat dish options
               - Don't make filtering decisions for the user - show options
               - Only filter when user explicitly says "remove X" or "only keep Y"
               - Maximum {limit} dishes total
               - NEVER add duplicates - always check! if exact same dish already exists in selection
                - If user selects existing dish, just keep that one, don't add again
               - Respect allergies and preferences
               - For images: MUST include actual matching dishes from database
            
            4. BE SMART about context:
               - "only khinkali" = keep only khinkali dishes from current selection
               - "I have pork allergy" = remove all pork dishes
               - "add drinks" = add drinks to existing selection
               - "remove everything except beef khinkali" = keep only beef khinkali
               - IMAGE + "What food is this?" = identify AND find similar dishes in database
               - if requested dish is not in the database and you can't find similar dishes, leave blank space for that dish

            CRITICAL SELECTION RULES:
                - "ხინკალი მინდა" = show ALL khinkali options (exploration phase)
                - "საქონლის ხინკალს ავიღებ" = choose ONLY beef khinkali, REMOVE all other khinkali (selection phase)  
                - "დავამატებ X" = choose X from shown options, remove other similar items
                - "ავიღებ X" = same as above - selection, not addition

                NEVER keep multiple items of same type after user makes a choice.
            
            5. CRITICAL FOR IMAGES:
               - Don't just describe the food - FIND MATCHING DISHES in the database
               - Look for dishes with similar ingredients, cooking methods, or cuisine type
               - Return actual available dishes, not descriptions
            
            <example>
            EXAMPLE CONVERSATION TO FOLLOW:
                User: "აჭარული ხაჭაპური მინდა" (I want Adjarian khachapuri)
                → Show ALL available აჭარული ხაჭაპური options from all restaurants

                User: "სახლი 11-ს აჭარული ავიღებ" (I'll take Adjarian from Sakhli #11)
                → Keep that specific khachapuri remove other khachapuri options

                User: "სასმელიც" (also drinks)  
                → Keep the khachapuri selection + show ALL available drink options from the same restaurant user chose

                User: "რამე ხორციანსაც" (something with meat too)
                → Keep khachapuri from same restaurant + keep drinks + show ALL meat dish options

                User: "აღარ მინდა სასმელი" (I don't want drinks)
                → Keep khachapuri + remove drinks + show ALL meat dish options

                User: "ვსო შევუკვეთავ" (I'll order these)
                → Keep khachapuri + keep meat dishes + show final selection

                This way user sees all options and makes their own filtering decisions.   
            </example>

            <example 2>
                User: "ხინკალი მინდა" (I want khinkali)
                → Show ALL 5 khinkali options

                User: "საქონლის ხინკალს ავიღებ" (I'll take beef khinkali)  
                → Keep ONLY "ხინკალი (საქონლის, 1 ცალი)" - remove other 4 khinkali options
                → Final selection: 1 dish (the chosen beef khinkali)

                User: "სასმელიც დავამატებ" (I'll also add drinks)
                → Keep beef khinkali + show ALL drink options
            </example 2>

            OUTPUT FORMAT (JSON ONLY):
            {{
                "results": [
                    {{
                        "restaurant_id": "...",
                        "restaurant_name": "...",
                        "dish_name": "...",
                        "dish_price": 0.00,
                    }}
                ],
                "operation_performed": "added" | "filtered" | "replaced" | "removed" | "no_change"
            }}
            """
            
            contents.append(full_prompt)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=4000
                )
            )
            
            result = json.loads(response.text)
            final_dishes = result.get("results", [])
            
            # Update our state with AI's final selection
            self.conversation.turn_count += 1
            self.conversation.update_selected_dishes(final_dishes)
            
            return {"status": "success", "data": result}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def start_new_conversation(self, preferences: str = ""):
        """Start a fresh conversation."""
        self.conversation = ConversationState()
        self.conversation.user_preferences = preferences

    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        return {
            "turn_count": self.conversation.turn_count,
            "selected_dishes": self.conversation.selected_dishes,
            "total_dishes": len(self.conversation.selected_dishes),
            "total_cost": sum(d["dish_price"] for d in self.conversation.selected_dishes),
            "is_satisfied": self.conversation.is_satisfied
        }

    def is_conversation_active(self) -> bool:
        """Check if conversation is still active."""
        return not self.conversation.is_satisfied