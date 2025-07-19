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
    """Manages cumulative conversation state like a shopping cart."""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.selected_dishes: List[Dict[str, Any]] = []  # Cumulative dishes (our "cart")
        self.excluded_dishes: List[str] = []  # Dishes to never suggest again
        self.all_suggested_dishes: List[str] = []  # For duplicate prevention
        self.user_preferences: str = ""
        self.initial_query: str = ""
        self.turn_count: int = 0
        self.is_satisfied: bool = False
        
    def add_user_message(self, message: str, message_type: str = "query"):
        """Add a user message to conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": message,
            "type": message_type,
            "turn": self.turn_count
        })
        
    def update_selected_dishes(self, new_results: List[Dict[str, Any]]):
        """Update the cumulative selected dishes based on new results."""
        # For the first turn, just use the new results
        if self.turn_count == 0:
            self.selected_dishes = new_results.copy()
        else:
            # For subsequent turns, intelligently merge
            # Remove any explicitly excluded dishes first
            self.selected_dishes = [
                dish for dish in self.selected_dishes 
                if f"{dish['restaurant_name']}_{dish['dish_name']}" not in self.excluded_dishes
            ]
            
            # Add new dishes that aren't already in our selection
            existing_keys = [f"{d['restaurant_name']}_{d['dish_name']}" for d in self.selected_dishes]
            
            for dish in new_results:
                dish_key = f"{dish['restaurant_name']}_{dish['dish_name']}"
                if dish_key not in existing_keys and dish_key not in self.excluded_dishes:
                    self.selected_dishes.append(dish)
        
        # Update conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": self.selected_dishes.copy(),
            "turn": self.turn_count
        })
        
        # Track for duplicate prevention
        for dish in self.selected_dishes:
            dish_key = f"{dish['restaurant_name']}_{dish['dish_name']}"
            if dish_key not in self.all_suggested_dishes:
                self.all_suggested_dishes.append(dish_key)
        
    def remove_dish(self, dish_name: str, restaurant_name: str):
        """Remove a dish from selection and mark as excluded."""
        dish_key = f"{restaurant_name}_{dish_name}"
        
        # Remove from current selection
        self.selected_dishes = [
            dish for dish in self.selected_dishes
            if not (dish['dish_name'] == dish_name and dish['restaurant_name'] == restaurant_name)
        ]
        
        # Add to exclusion list
        if dish_key not in self.excluded_dishes:
            self.excluded_dishes.append(dish_key)
            
    def is_adding_request(self, user_input: str) -> bool:
        """Detect if user wants to add more dishes to existing selection."""
        adding_indicators = [
            "add", "also", "plus", "more", "additional", "include", 
            "suggest more", "what else", "anything else", "recommend more"
        ]
        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in adding_indicators)
    
    def is_replacement_request(self, user_input: str) -> bool:
        """Detect if user wants to replace current selection completely."""
        replacement_indicators = [
            "instead", "replace all", "start over", "forget", "new selection", 
            "different dishes", "something else entirely"
        ]
        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in replacement_indicators)
            
    def get_conversation_context(self) -> str:
        """Get formatted conversation context."""
        context = f"CONVERSATION TURN: {self.turn_count}\n"
        context += f"INITIAL QUERY: {self.initial_query}\n"
        context += f"USER PREFERENCES: {self.user_preferences}\n"
        
        if self.selected_dishes:
            selected_info = [f"{d['dish_name']} from {d['restaurant_name']} (${d['dish_price']})" for d in self.selected_dishes]
            context += f"CURRENT SELECTION ({len(self.selected_dishes)} dishes): {', '.join(selected_info)}\n"
        
        if self.excluded_dishes:
            context += f"EXCLUDED DISHES (never suggest): {', '.join(self.excluded_dishes[:5])}\n"
            
        if self.conversation_history:
            context += "RECENT CONVERSATION:\n"
            for msg in self.conversation_history[-2:]:
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                    
        return context

class SupraMultiSearchEngine:
    """Conversational S.U.P.R.A. agent with cumulative context awareness."""
    
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
        Main chat interface with cumulative context awareness.
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
            # Determine request type and search accordingly
            if self.conversation.turn_count == 0:
                # Initial request
                self.conversation.initial_query = user_input
                search_type = "initial"
                
            elif self.conversation.is_replacement_request(user_input):
                # Replace entire selection
                self.conversation.selected_dishes = []
                search_type = "replacement"
                
            elif self.conversation.is_adding_request(user_input):
                # Add to existing selection
                search_type = "addition"
                
            else:
                # Default to addition for ambiguous requests
                search_type = "addition"
            
            # Process any explicit removals first
            self._process_removals(user_input)
            
            # Perform search
            self.conversation.add_user_message(user_input)
            result = await self._search_with_context(user_input, search_type, image_path, limit)
            
            if result["status"] == "success":
                # Return final cumulative selection
                return {
                    "status": "success",
                    "data": {
                        "conversation_response": result["data"]["conversation_response"],
                        "results": self.conversation.selected_dishes,
                        "search_type": search_type,
                        "new_dishes_added": result["data"].get("new_dishes_count", 0)
                    },
                    "conversation_complete": False,
                    "conversation_state": {
                        "turn_count": self.conversation.turn_count,
                        "total_dishes": len(self.conversation.selected_dishes),
                        "excluded_count": len(self.conversation.excluded_dishes)
                    }
                }
            else:
                return result
                
        except Exception as e:
            return {"status": "error", "message": str(e), "conversation_complete": False}

    async def _search_with_context(self, query: str, search_type: str, image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Internal search with proper cumulative context.
        """
        contents = []
        
        try:
            restaurant_data_json = json.dumps(self.restaurant_data, ensure_ascii=False)
            conversation_context = self.conversation.get_conversation_context()
            
            # Handle image if provided
            if image_path:
                image_part = self._process_image(image_path)
                contents.append(image_part)
            
            # Determine search parameters
            current_selection_count = len(self.conversation.selected_dishes)
            
            if search_type == "initial":
                search_instruction = f"Find {limit} dishes matching: '{query}'"
                context_instruction = "This is the initial search."
                
            elif search_type == "replacement":
                search_instruction = f"Find {limit} completely different dishes for: '{query}'"
                context_instruction = "User wants to replace their entire selection."
                
            elif search_type == "addition":
                new_dishes_needed = max(1, limit - current_selection_count)
                search_instruction = f"Find {new_dishes_needed} NEW dishes to ADD to the existing selection for: '{query}'"
                context_instruction = f"User wants to ADD to their current {current_selection_count} dishes."
            else:
                search_instruction = f"Find dishes for: '{query}'"
                context_instruction = "General search."

            full_prompt = f"""
            You are a professional Georgian cuisine expert. SEARCH TYPE: {search_type.upper()}
            
            {conversation_context}
            
            REQUEST: "{query}"
            {context_instruction}
            
            RESTAURANT DATA:
            {restaurant_data_json}

            INSTRUCTIONS:
            1. {search_instruction}
            2. NEVER suggest dishes from EXCLUDED list
            3. NEVER duplicate dishes already in CURRENT SELECTION
            4. Focus on user preferences and dietary needs
            5. Return only NEW dishes to be added (not the existing selection)
            6. Each dish must be unique

            OUTPUT FORMAT (JSON ONLY):
            {{
                "conversation_response": "Brief response acknowledging the request",
                "results": [
                    {{
                        "restaurant_id": "...",
                        "restaurant_name": "...",
                        "dish_name": "...",
                        "dish_price": 0.00,
                        "reason": "Why this dish fits the request"
                    }}
                ],
                "new_dishes_count": number_of_new_dishes_returned
            }}
            """
            
            contents.append(full_prompt)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                    max_output_tokens=3000
                )
            )
            
            result = json.loads(response.text)
            new_dishes = result.get("results", [])
            
            # Update the cumulative selection
            self.conversation.turn_count += 1
            self.conversation.update_selected_dishes(new_dishes)
            
            return {"status": "success", "data": result}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _process_removals(self, user_input: str):
        """Process any explicit dish removal requests."""
        removal_indicators = ["remove", "delete", "don't want", "take out", "exclude"]
        user_input_lower = user_input.lower()
        
        for indicator in removal_indicators:
            if indicator in user_input_lower:
                # Simple removal by position (e.g., "remove #1", "delete item 2")
                import re
                position_match = re.search(r'#(\d+)|item (\d+)|number (\d+)', user_input_lower)
                if position_match:
                    position = int(position_match.group(1) or position_match.group(2) or position_match.group(3))
                    if 1 <= position <= len(self.conversation.selected_dishes):
                        dish_to_remove = self.conversation.selected_dishes[position - 1]
                        self.conversation.remove_dish(dish_to_remove["dish_name"], dish_to_remove["restaurant_name"])
                        print(f"🗑️ Removed: {dish_to_remove['dish_name']} from {dish_to_remove['restaurant_name']}")

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
            "excluded_dishes": self.conversation.excluded_dishes,
            "is_satisfied": self.conversation.is_satisfied
        }

    def is_conversation_active(self) -> bool:
        """Check if conversation is still active."""
        return not self.conversation.is_satisfied