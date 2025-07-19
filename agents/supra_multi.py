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
        return types.Part.from_uri(
            uri=image_path,
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
            1. UNDERSTAND the user's intent:
               - Adding dishes? ("add", "also", "more", "suggest")
               - Removing/filtering? ("only", "just", "don't want", "remove", "except")
               - Replacing? ("instead", "different")
               - Asking for information? ("show", "what do I have")
            
            2. WORK WITH CURRENT SELECTION:
               - If user wants to ADD: keep current dishes + add new ones
               - If user wants ONLY specific items: filter current selection to keep ONLY those items
               - If user says "I don't want X": remove X from current selection
               - If user has allergies: remove/avoid allergens
            
            3. RETURN FINAL COMPLETE SELECTION (not just new dishes):
               - Always return the FULL selection user should have
               - Maximum {limit} dishes total
               - No duplicates
               - Respect allergies and preferences
            
            4. BE SMART about context:
               - "only khinkali" = keep only khinkali dishes from current selection
               - "I have pork allergy" = remove all pork dishes
               - "add drinks" = add drinks to existing selection
               - "remove everything except beef khinkali" = keep only beef khinkali
            
            OUTPUT FORMAT (JSON ONLY):
            {{
                "conversation_response": "Natural response explaining what you did",
                "results": [
                    {{
                        "restaurant_id": "...",
                        "restaurant_name": "...",
                        "dish_name": "...",
                        "dish_price": 0.00,
                        "reason": "Brief explanation"
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