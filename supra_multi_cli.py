import asyncio
import argparse
import json
from agents.supra_multi import SupraMultiSearchEngine

class SupraMultiCLI:
    """Simple chat-style CLI for the conversational S.U.P.R.A. engine."""
    
    def __init__(self):
        self.engine = SupraMultiSearchEngine()
        
    def display_results(self, data: dict):
        """Display search results in a friendly format."""
        results = data.get("results", [])
        conversation_response = data.get("conversation_response", "")
        
        if conversation_response:
            print(f"🤖 {conversation_response}\n")
        
        if not results:
            return
            
        print(f"🍽️  Here are {len(results)} recommendations:\n")
        
        # Separate preserved and new dishes
        preserved = [r for r in results if r.get('status') == 'preserved']
        new_dishes = [r for r in results if r.get('status') != 'preserved']
        
        # Show preserved dishes first
        if preserved:
            print("✅ DISHES YOU LIKED (keeping these):")
            for i, result in enumerate(preserved, 1):
                self._print_dish(i, result)
        
        # Show new dishes
        if new_dishes:
            if preserved:
                print("🆕 NEW RECOMMENDATIONS:")
            start_num = len(preserved) + 1
            for i, result in enumerate(new_dishes, start_num):
                self._print_dish(i, result)

    def _print_dish(self, num: int, result: dict):
        """Print a single dish in formatted way."""
        print(f"   {num}. {result['dish_name']}")
        print(f"      🏪 {result['restaurant_name']}")
        print(f"      💰 ${result['dish_price']}")
        if result.get('reason'):
            print(f"      💭 {result['reason']}")
        print()

    async def run_chat(self, preferences: str = ""):
        """Run the chat interface."""
        
        # Load data
        if not self.engine.load_data():
            print("❌ Could not load restaurant data. Exiting.")
            return
            
        # Start conversation
        if preferences:
            self.engine.start_new_conversation(preferences)
            print(f"📝 Your preferences: {preferences}")
            
        print("🍽️  Welcome! I'm your Georgian cuisine assistant.")
        print("💬 Tell me what you'd like to eat, and I'll help you find the perfect dishes!")
        print("📌 Just say what you want - I'll remember what you like and don't like.")
        print("🛑 Say 'thanks' or 'perfect' when you're satisfied.\n")
        print("=" * 60)
        
        while True:
            # Get user input
            try:
                user_input = input("\n👤 You: ").strip()
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Enjoy your meal!")
                break
                
            if not user_input:
                continue
                
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                print("👋 Goodbye! Enjoy your meal!")
                break
            
            print("🔍 Searching...")
            
            # Send to chat engine
            response = await self.engine.chat(user_input)
            
            # Handle different response types
            if response["status"] == "satisfied":
                print(f"🤖 {response['message']}")
                
                # Show final summary
                final_selection = response.get("final_selection", {})
                liked_dishes = final_selection.get("liked_dishes", [])
                if liked_dishes:
                    print(f"\n📋 Your final selection ({len(liked_dishes)} dishes you liked):")
                    for i, dish in enumerate(liked_dishes, 1):
                        print(f"   {i}. {dish['dish_name']} from {dish['restaurant_name']} (${dish['dish_price']})")
                
                print("\n🎉 Conversation completed!")
                break
                
            elif response["status"] == "success":
                self.display_results(response["data"])
                
                # Show conversation state
                state = response.get("conversation_state", {})
                if state.get("liked_dishes_count", 0) > 0 or state.get("excluded_dishes_count", 0) > 0:
                    liked = state.get("liked_dishes_count", 0)
                    excluded = state.get("excluded_dishes_count", 0)
                    print(f"📊 Status: {liked} liked, {excluded} excluded")
                    
            elif response["status"] == "no_response":
                # Silently continue - no need to respond to empty input
                continue
                
            elif response["status"] == "error":
                print(f"❌ Error: {response['message']}")
                
            else:
                print(f"❓ Unexpected response: {response}")

    async def run_single_query(self, query: str, preferences: str = "", user_name: str = ""):
        """Run a single query (for API-like usage)."""
        
        # Load data
        if not self.engine.load_data():
            return {"error": "Could not load restaurant data"}
            
        # Load user preferences if user name provided
        if user_name:
            try:
                with open('data/users.json', 'r') as f:
                    users = json.load(f)
                    
                for user in users:
                    if user['name'] == user_name:
                        preferences = ', '.join(user['preferences'])
                        break
            except Exception as e:
                print(f"Warning: Could not load user preferences: {e}")
        
        # Start conversation with preferences
        if preferences:
            self.engine.start_new_conversation(preferences)
            
        # Send query
        response = await self.engine.chat(query)
        
        # Print results
        if response["status"] == "success":
            self.display_results(response["data"])
            print("\n--- Raw JSON Response ---")
            print(json.dumps(response, indent=2, ensure_ascii=False))
        else:
            print(f"Response: {json.dumps(response, indent=2, ensure_ascii=False)}")
        
        return response

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="S.U.P.R.A. Multi-turn Chat Interface")
    
    parser.add_argument('-q', '--query', type=str, 
                       help="Single query mode - provide one query and exit")
    parser.add_argument('-p', '--preferences', type=str, default="",
                       help="User preferences and dietary restrictions")
    parser.add_argument('-u', '--user', type=str, default="",
                       help="Load preferences from users.json")
    
    args = parser.parse_args()
    
    cli = SupraMultiCLI()
    
    if args.query:
        # Single query mode
        await cli.run_single_query(
            query=args.query,
            preferences=args.preferences,
            user_name=args.user
        )
    else:
        # Interactive chat mode
        await cli.run_chat(preferences=args.preferences)

if __name__ == "__main__":
    asyncio.run(main())