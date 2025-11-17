import argparse
from backend.suggestor import suggest_smart

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str, help="Query text for citation recommendations")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    results = suggest_smart(args.text, top_k=args.top_k)

    for r in results:
        print("=" * 60)
        print(f"Paper ID: {r['paper_id']}")
        print(f"Title: {r['title']}")
        print("\nBibTeX:\n")
        print(r["bibtex"])
        print("\nExplanation:")
        print(r["explanation"])
        print("=" * 60)
        print()

if __name__ == "__main__":
    main()
