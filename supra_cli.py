import asyncio
import argparse
import json
from agents.supra import SupraSearchEngine

async def main():
    """
    Main function to run the S.U.P.R.A. CLI tool.
    """
    parser = argparse.ArgumentParser(description="S.U.P.R.A. Command-Line Search Tool")
    parser.add_argument('-q', '--query', type=str, help="Text query for your search.", default="")
    parser.add_argument('-i', '--image', type=str, help="Path to an image file for visual search.", default="")
    parser.add_argument('-u', '--user', type=str, help="Enter user name to get personalized results.", default="")

    args = parser.parse_args()
    
    if not args.query and not args.image:
        print("Error: You must provide a --query or an --image path.")
        return

    print("Initializing S.U.P.R.A. CLI...")
    engine = SupraSearchEngine()
    engine.load_data()
    user_preferences = ""

    if args.user:
        with open('data/users.json', 'r') as f:
            users = json.load(f)
            
            for user in users:
                if user['name'] == args.user:
                    user_preferences = ', '.join(user['preferences'])
                    break
    else:
        user_preferences = ""
    
    print(f"Performing search (Query: '{args.query}', Image: '{args.image}')...")
    results = await engine.search(query=args.query, image_path=args.image, preferences = user_preferences)
    
    print("\n--- Search Results ---")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("----------------------\n")

if __name__ == "__main__":
    asyncio.run(main())
