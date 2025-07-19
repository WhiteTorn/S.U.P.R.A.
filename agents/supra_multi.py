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
    """Manages the state of the conversation including history and current results."""
    
    def __init__(self):
        self.conversation_history: List[Dict[str, Any]] = []
        self.current_results: List[Dict[str, Any]] = []
        self.excluded_dishes: List[str] = []  # Dishes user didn't like
        self.liked_dishes: List[Dict[str, Any]] = []  # Dishes user explicitly likes
        self.all_suggested_dishes: List[str] = []  # Track all dishes suggested to prevent duplicates
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
        
    def add_assistant_message(self, results: List[Dict[str, Any]]):
        """Add assistant response to conversation history."""
        self.conversation_history.append({
            "role": "assistant",
            "content": results,
            "turn": self.turn_count
        })
        self.current_results = results
        
        # Track suggested dishes to prevent duplicates
        for result in results:
            dish_key = f"{result['restaurant_name']}_{result['dish_name']}"
            if dish_key not in self.all_suggested_dishes:
                self.all_suggested_dishes.append(dish_key)
        
    def exclude_dish(self, dish_name: str, restaurant_name: str):
        """Add a dish to the exclusion list."""
        dish_key = f"{restaurant_name}_{dish_name}"
        if dish_key not in self.excluded_dishes:
            self.excluded_dishes.append(dish_key)
            
    def like_dish(self, dish_dict: Dict[str, Any]):
        """Add a dish to the liked dishes list."""
        dish_key = f"{dish_dict['restaurant_name']}_{dish_dict['dish_name']}"
        # Check if not already in liked dishes
        existing_keys = [f"{d['restaurant_name']}_{d['dish_name']}" for d in self.liked_dishes]
        if dish_key not in existing_keys:
            self.liked_dishes.append(dish_dict)
    
    def preserve_dishes_by_keyword(self, feedback: str):
        """Preserve dishes based on keywords in feedback (e.g., 'preserve khachapuris')."""
        feedback_lower = feedback.lower()
        
        # Map common English terms to Georgian dish types
        dish_mappings = {
            'khachapuri': ['ხაჭაპური'],
            'khachapuris': ['ხაჭაპური'], 
            'khinkali': ['ხინკალი'],
            'lobio': ['ლობიო'],
            'ostri': ['ოსტრი'],
            'pkhali': ['ფხალი'],
            'badrijani': ['ბადრიჯანი'],
            'cheese': ['ყველი'],
            'soup': ['სუპი'],
            'salad': ['სალათი'],
            'meat': ['ხორცი', 'საქონლის', 'ღორის'],
            'beef': ['საქონლის'],
            'pork': ['ღორის'],
            'vegetarian': ['ვეგეტარიანული']
        }
        
        preserve_keywords = ['preserve', 'keep', 'save', 'maintain']
        
        # Check if user wants to preserve specific dish types
        for preserve_word in preserve_keywords:
            if preserve_word in feedback_lower:
                for english_term, georgian_terms in dish_mappings.items():
                    if english_term in feedback_lower:
                        # Find matching dishes in current results
                        for result in self.current_results:
                            dish_name = result['dish_name']
                            # Check if any Georgian term appears in the dish name
                            for georgian_term in georgian_terms:
                                if georgian_term in dish_name:
                                    self.like_dish(result)
                                    print(f"🔒 Preserved: {dish_name} from {result['restaurant_name']}")
                                    break
        
        # Also check for "all" or "everything"
        if any(word in feedback_lower for word in ['preserve all', 'keep all', 'save all', 'preserve everything']):
            for result in self.current_results:
                self.like_dish(result)
                print(f"🔒 Preserved: {result['dish_name']} from {result['restaurant_name']}")
            
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for the AI."""
        context = f"CONVERSATION TURN: {self.turn_count}\n"
        context += f"INITIAL QUERY: {self.initial_query}\n"
        context += f"USER PREFERENCES: {self.user_preferences}\n"
        
        if self.liked_dishes:
            liked_dishes_info = [f"{d['dish_name']} from {d['restaurant_name']}" for d in self.liked_dishes]
            context += f"LIKED/PRESERVED DISHES (must include first): {', '.join(liked_dishes_info)}\n"
        
        if self.excluded_dishes:
            context += f"EXCLUDED DISHES (never suggest): {', '.join(self.excluded_dishes)}\n"
            
        if self.all_suggested_dishes:
            context += f"ALREADY SUGGESTED (avoid duplicates): {', '.join(self.all_suggested_dishes[:10])}\n"  # Limit to avoid long context
            
        if self.conversation_history:
            context += "RECENT CONVERSATION:\n"
            for msg in self.conversation_history[-3:]:  # Last 3 messages for context
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                else:
                    dishes_count = len(msg['content'])
                    context += f"Assistant: Suggested {dishes_count} dishes\n"
                    
        return context

class SupraMultiSearchEngine:
    """Conversational S.U.P.R.A. agent for iterative restaurant search."""
    
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
        """Automatically detect if user is satisfied with results - more precise detection."""
        
        # More specific satisfaction phrases that clearly indicate completion
        strong_satisfaction = [
            "perfect, thank you", "that's perfect", "perfect thanks", "these are perfect",
            "i'm satisfied", "that's all", "i'm done", "that's enough",
            "order these", "i'll order these", "book these", "reserve these",
            "exactly what i wanted", "this is what i wanted"
        ]
        
        # Closing phrases
        closing_phrases = [
            "thank you, bye", "thanks, goodbye", "that's all, thanks",
            "perfect, goodbye", "done, thank you"
        ]
        
        user_input_lower = user_input.lower().strip()
        
        # Check for strong satisfaction indicators
        for indicator in strong_satisfaction + closing_phrases:
            if indicator in user_input_lower:
                return True
        
        # Only treat very specific short responses as satisfaction
        if user_input_lower in ["perfect", "done", "finished", "enough"]:
            return True
            
        return False

    async def chat(self, user_input: str, image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Main chat interface - handles both initial queries and follow-up messages.
        Returns response with conversation status.
        """
        
        # If no input provided, return empty response
        if not user_input.strip():
            return {
                "status": "no_response",
                "message": "No input provided",
                "conversation_complete": self.conversation.is_satisfied
            }
        
        # Check if user is satisfied
        if self._detect_satisfaction(user_input):
            self.conversation.is_satisfied = True
            return {
                "status": "satisfied",
                "message": "Perfect! I'm glad you found exactly what you were looking for. Enjoy your meal!",
                "conversation_complete": True,
                "final_selection": {
                    "liked_dishes": self.conversation.liked_dishes,
                    "total_turns": self.conversation.turn_count
                }
            }
        
        try:
            # If this is the first message, start conversation
            if self.conversation.turn_count == 0:
                self.conversation.initial_query = user_input
                self.conversation.add_user_message(user_input, "initial_query")
                result = await self._search_with_context(user_input, image_path, limit)
            else:
                # Process feedback and continue conversation
                self._process_feedback(user_input)
                result = await self._search_with_context(user_input, image_path, limit)
            
            if result["status"] == "success":
                return {
                    "status": "success",
                    "data": result["data"],
                    "conversation_complete": False,
                    "conversation_state": {
                        "turn_count": self.conversation.turn_count,
                        "liked_dishes_count": len(self.conversation.liked_dishes),
                        "excluded_dishes_count": len(self.conversation.excluded_dishes),
                        "total_suggested": len(self.conversation.all_suggested_dishes)
                    }
                }
            else:
                return result
                
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "conversation_complete": False
            }

    def _has_explicit_feedback(self, user_input: str) -> bool:
        """Check if user provided explicit like/dislike feedback."""
        feedback_indicators = [
            "like", "don't like", "dislike", "hate", "love", "remove", "replace", 
            "keep", "preserve", "save", "good", "bad", "terrible", "excellent", "not interested"
        ]
        
        user_input_lower = user_input.lower()
        return any(indicator in user_input_lower for indicator in feedback_indicators)

    async def _search_with_context(self, query: str = "", image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Internal search method with context awareness and proper preservation.
        """
        contents = []
        
        try:
            restaurant_data_json = json.dumps(self.restaurant_data, ensure_ascii=False)
            conversation_context = self.conversation.get_conversation_context()
            
            # Handle image if provided
            if image_path:
                image_part = self._process_image(image_path)
                contents.append(image_part)
                
            # Calculate dishes needed
            preserved_count = len(self.conversation.liked_dishes)
            new_dishes_needed = max(0, limit - preserved_count)
            
            # Create contextual prompt
            if self.conversation.turn_count == 0:
                prompt_type = "INITIAL SEARCH"
                search_instruction = f"Find {limit} unique dishes matching: '{query}'"
            else:
                prompt_type = "REFINEMENT SEARCH"
                if preserved_count > 0:
                    search_instruction = f"PRESERVE all {preserved_count} liked dishes at the TOP with status='preserved', then add {new_dishes_needed} NEW dishes with status='new'"
                else:
                    search_instruction = f"Find {new_dishes_needed} NEW dishes based on: '{query}'"

            # Format preserved dishes for clear instructions
            liked_dishes_list = []
            for dish in self.conversation.liked_dishes:
                liked_dishes_list.append({
                    "dish_name": dish["dish_name"],
                    "restaurant_name": dish["restaurant_name"],
                    "dish_price": dish["dish_price"],
                    "status": "preserved"
                })

            full_prompt = f"""
            You are a professional Georgian cuisine expert. {prompt_type}
            
            {conversation_context}
            
            CURRENT REQUEST: "{query}"
            
            RESTAURANT DATA:
            {restaurant_data_json}

            CRITICAL INSTRUCTIONS:
            1. {search_instruction}
            2. FIRST: Include ALL preserved dishes exactly as listed below with status="preserved"
            3. THEN: Add NEW unique dishes with status="new"
            4. NEVER duplicate any dish (check restaurant_name + dish_name combination)
            5. NEVER suggest dishes from EXCLUDED or ALREADY SUGGESTED lists
            6. Total results must be exactly {limit} dishes
            7. Keep responses brief to prevent truncation

            PRESERVED DISHES (MUST include first with exact same data):
            {json.dumps(liked_dishes_list, ensure_ascii=False)}

            OUTPUT FORMAT (JSON ONLY):
            {{
                "conversation_response": "Brief response (max 100 chars)",
                "results": [
                    {{
                        "restaurant_id": "exact_id_from_data",
                        "restaurant_name": "exact_name_from_data", 
                        "dish_name": "exact_name_from_data",
                        "dish_price": exact_price_from_data,
                        "reason": "Brief reason (max 80 chars)",
                        "status": "preserved" | "new"
                    }}
                ]
            }}
            """
            
            contents.append(full_prompt)

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.05,  # Very low temperature for consistency
                    max_output_tokens=3000
                )
            )
            
            result = json.loads(response.text)
            
            # Validate and enforce preservation order
            results = result.get("results", [])
            preserved_results = [r for r in results if r.get("status") == "preserved"]
            new_results = [r for r in results if r.get("status") != "preserved"]
            
            # Ensure preserved dishes come first
            final_results = preserved_results + new_results[:limit-len(preserved_results)]
            
            # Remove any duplicates as final safety check
            final_results = self._remove_duplicates(final_results)
            
            result["results"] = final_results
            self.conversation.add_assistant_message(final_results)
            
            return {"status": "success", "data": result}

        except Exception as e:
            print(f"Error in search: {e}")
            return {"status": "error", "message": str(e)}

    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates ensuring no same dish from same restaurant appears twice."""
        seen = set()
        cleaned = []
        
        for result in results:
            dish_key = f"{result['restaurant_name']}_{result['dish_name']}"
            if dish_key not in seen:
                seen.add(dish_key)
                cleaned.append(result)
        
        return cleaned

    def _process_feedback(self, feedback: str):
        """Enhanced feedback processing with better preservation detection."""
        self.conversation.turn_count += 1
        self.conversation.add_user_message(feedback, "feedback")
        
        feedback_lower = feedback.lower()
        
        # First check for preservation keywords
        self.conversation.preserve_dishes_by_keyword(feedback)
        
        # Then check for specific dish feedback
        for result in self.conversation.current_results:
            dish_name_lower = result["dish_name"].lower()
            restaurant_name = result["restaurant_name"]
            
            # Positive indicators
            positive_indicators = ["like this", "love this", "good", "great", "want this", "yes"]
            negative_indicators = ["don't like", "not interested", "remove", "replace", "hate", "dislike", "no"]
            
            # Check mentions of this specific dish (by position or partial name)
            dish_mentioned = False
            
            # Check if dish is mentioned by position (e.g., "I like #1" or "remove item 2")
            for i, res in enumerate(self.conversation.current_results, 1):
                if res == result:
                    if f"#{i}" in feedback_lower or f"item {i}" in feedback_lower or f"number {i}" in feedback_lower:
                        dish_mentioned = True
                        break
            
            if dish_mentioned:
                # Check if positive or negative
                for indicator in positive_indicators:
                    if indicator in feedback_lower:
                        self.conversation.like_dish(result)
                        print(f"👍 Liked: {result['dish_name']}")
                        break
                        
                for indicator in negative_indicators:
                    if indicator in feedback_lower:
                        self.conversation.exclude_dish(result["dish_name"], restaurant_name)
                        print(f"👎 Excluded: {result['dish_name']}")
                        break

    def start_new_conversation(self, preferences: str = ""):
        """Start a fresh conversation with optional user preferences."""
        self.conversation = ConversationState()
        self.conversation.user_preferences = preferences

    def get_conversation_state(self) -> Dict[str, Any]:
        """Get current conversation state."""
        return {
            "turn_count": self.conversation.turn_count,
            "initial_query": self.conversation.initial_query,
            "user_preferences": self.conversation.user_preferences,
            "liked_dishes": self.conversation.liked_dishes,
            "excluded_dishes": self.conversation.excluded_dishes,
            "current_results_count": len(self.conversation.current_results),
            "total_suggested": len(self.conversation.all_suggested_dishes),
            "is_satisfied": self.conversation.is_satisfied
        }

    def is_conversation_active(self) -> bool:
        """Check if conversation is still active (not satisfied)."""
        return not self.conversation.is_satisfied