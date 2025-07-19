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
        self.user_preferences: str = ""
        self.initial_query: str = ""
        self.turn_count: int = 0
        
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
        
    def exclude_dish(self, dish_name: str, restaurant_name: str):
        """Add a dish to the exclusion list."""
        dish_key = f"{restaurant_name}_{dish_name}"
        if dish_key not in self.excluded_dishes:
            self.excluded_dishes.append(dish_key)
            
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for the AI."""
        context = f"CONVERSATION TURN: {self.turn_count}\n"
        context += f"INITIAL QUERY: {self.initial_query}\n"
        context += f"USER PREFERENCES: {self.user_preferences}\n"
        
        if self.excluded_dishes:
            context += f"EXCLUDED DISHES (user didn't like): {', '.join(self.excluded_dishes)}\n"
            
        if self.conversation_history:
            context += "PREVIOUS CONVERSATION:\n"
            for msg in self.conversation_history[-4:]:  # Last 4 messages for context
                if msg["role"] == "user":
                    context += f"User: {msg['content']}\n"
                else:
                    dishes = [f"{r['dish_name']} from {r['restaurant_name']}" for r in msg['content']]
                    context += f"Assistant: Suggested {len(dishes)} dishes\n"
                    
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

    def load_data(self, data_path: str = "../data/rests.json"):
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
        
    def start_conversation(self, initial_query: str, preferences: str = "", image_path: str = ""):
        """Start a new conversation with initial query."""
        self.conversation = ConversationState()
        self.conversation.initial_query = initial_query
        self.conversation.user_preferences = preferences
        self.conversation.add_user_message(initial_query, "initial_query")
        
        print(f"🍽️  Welcome! Let me help you find the perfect Georgian dishes.")
        print(f"👤 You said: {initial_query}")
        if preferences:
            print(f"📝 Your preferences: {preferences}")
        print("=" * 50)

    async def search_with_context(self, query: str = "", image_path: str = "", limit: int = 10) -> Dict[str, Any]:
        """
        Performs contextual search considering conversation history and user feedback.
        """
        contents = []
        
        try:
            restaurant_data_json = json.dumps(self.restaurant_data, ensure_ascii=False)
            conversation_context = self.conversation.get_conversation_context()
            
            # Handle image if provided
            if image_path:
                image_part = self._process_image(image_path)
                contents.append(image_part)
                
            # Create contextual prompt
            if self.conversation.turn_count == 0:
                # Initial search
                prompt_type = "INITIAL SEARCH"
                search_instruction = f"Find dishes matching the user's initial request: '{query}'"
            else:
                # Refinement search
                prompt_type = "REFINEMENT SEARCH"
                search_instruction = f"Based on user feedback, refine the recommendations. New request: '{query}'"

            full_prompt = f"""
            You are a professional Georgian cuisine expert and waiter. {prompt_type}
            
            {conversation_context}
            
            CURRENT REQUEST: "{query}"
            
            RESTAURANT DATA:
            {restaurant_data_json}

            INSTRUCTIONS:
            1. {search_instruction}
            2. Consider the conversation history and user feedback
            3. NEVER suggest dishes that are in the EXCLUDED DISHES list
            4. If this is a refinement, replace dishes user didn't like with better alternatives
            5. Focus on user preferences and allergies as TOP priority
            6. Act professionally like an experienced waiter who remembers customer preferences
            7. Return maximum {limit} results ranked by relevance
            8. Each dish should be unique (no duplicates)
            9. Ensure all data returned is accurate from the database

            RESPONSE STYLE:
            - Be conversational and helpful
            - Acknowledge previous feedback if this is a refinement
            - Explain why these new suggestions might be better

            OUTPUT FORMAT (JSON ONLY):
            {{
                "conversation_response": "Brief friendly response acknowledging the request...",
                "results": [
                    {{
                        "restaurant_id": "...",
                        "restaurant_name": "...",
                        "dish_name": "...",
                        "dish_price": 0.00,
                        "reason": "Why this dish fits the user's request..."
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
                    temperature=0.2
                )
            )
            
            result = json.loads(response.text)
            
            # Update conversation state
            self.conversation.add_assistant_message(result.get("results", []))
            
            return {"status": "success", "data": result}

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {"status": "error", "message": str(e)}

    def process_feedback(self, feedback: str):
        """Process user feedback and identify dishes to exclude or preferences to note."""
        self.conversation.turn_count += 1
        self.conversation.add_user_message(feedback, "feedback")
        
        # Simple feedback processing - in production, you might want more sophisticated NLP
        feedback_lower = feedback.lower()
        
        # Extract dishes user doesn't like
        for result in self.conversation.current_results:
            dish_name = result["dish_name"].lower()
            restaurant_name = result["restaurant_name"]
            
            # Check if user mentioned this dish negatively
            negative_indicators = ["don't like", "not interested", "no thanks", "remove", "replace"]
            for indicator in negative_indicators:
                if indicator in feedback_lower and dish_name in feedback_lower:
                    self.conversation.exclude_dish(result["dish_name"], restaurant_name)
                    print(f"📝 Noted: You don't want {result['dish_name']} from {restaurant_name}")

    async def continue_conversation(self, user_input: str, limit: int = 10) -> Dict[str, Any]:
        """Continue the conversation with user feedback or new request."""
        
        # Process the feedback
        self.process_feedback(user_input)
        
        # Perform new search with context
        result = await self.search_with_context(user_input, limit=limit)
        
        return result

    def display_results(self, results: List[Dict[str, Any]], conversation_response: str = ""):
        """Display search results in a friendly format."""
        if conversation_response:
            print(f"🤖 {conversation_response}\n")
        
        print(f"🍽️  Here are {len(results)} recommendations for you:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['dish_name']}")
            print(f"   🏪 {result['restaurant_name']}")
            print(f"   💰 ${result['dish_price']}")
            if 'reason' in result:
                print(f"   💭 {result['reason']}")
            print()

    def is_conversation_complete(self, user_input: str) -> bool:
        """Check if user is satisfied and wants to end conversation."""
        satisfaction_indicators = [
            "perfect", "great", "satisfied", "that's good", "looks good",
            "thank you", "thanks", "done", "finished", "order these"
        ]
        return any(indicator in user_input.lower() for indicator in satisfaction_indicators)

    async def run_conversation_loop(self, initial_query: str, preferences: str = "", max_turns: int = 5):
        """Main conversation loop for testing."""
        
        # Start conversation
        self.start_conversation(initial_query, preferences)
        
        # Initial search
        print("🔍 Searching for your preferences...")
        initial_result = await self.search_with_context(initial_query)
        
        if initial_result["status"] == "success":
            data = initial_result["data"]
            self.display_results(data["results"], data.get("conversation_response", ""))
        else:
            print(f"❌ Error: {initial_result['message']}")
            return
        
        # Conversation loop
        for turn in range(1, max_turns + 1):
            print(f"\n{'='*50}")
            print(f"Turn {turn}: What would you like to adjust? (or say 'perfect' if satisfied)")
            print("You can say things like:")
            print("- 'I don't like dish X, suggest something else'")
            print("- 'Too expensive, show cheaper options'")
            print("- 'More vegetarian options please'")
            print("- 'Perfect, I'm satisfied'")
            
            # In real implementation, this would come from user input
            # For testing, you can modify this part
            user_feedback = input("\n👤 Your feedback: ").strip()
            
            if self.is_conversation_complete(user_feedback):
                print("🎉 Great! I'm glad you found dishes you like. Enjoy your meal!")
                break
                
            print(f"\n🔍 Let me find better options based on your feedback...")
            
            # Continue conversation with feedback
            result = await self.continue_conversation(user_feedback)
            
            if result["status"] == "success":
                data = result["data"]
                self.display_results(data["results"], data.get("conversation_response", ""))
            else:
                print(f"❌ Error: {result['message']}")
                
        else:
            print("\n🕐 We've reached the maximum number of refinements. I hope you found something you like!")

# Example usage and testing
async def test_conversation():
    """Test the conversational SUPRA engine."""
    supra = SupraMultiSearchEngine()
    
    # Load data
    if not supra.load_data():
        return
    
    # Test conversation
    await supra.run_conversation_loop(
        initial_query="I want spicy Georgian food for dinner under $20",
        preferences="I'm allergic to nuts and prefer spicy food",
        max_turns=3
    )

if __name__ == "__main__":
    asyncio.run(test_conversation())