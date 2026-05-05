import requests
import time

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
OUTPUT_FILE = "player_names.txt"
BATCH_SIZE = 10000
TARGET = 50_000
HEADERS = {
    "User-Agent": "FootballRAG/1.0 (educational project; fetching footballer names)",
    "Accept": "application/json",
}


def fetch_batch(offset, limit):
    query = f"""
    SELECT ?articleName WHERE {{
      ?player wdt:P106 wd:Q937857 .
      ?article schema:about ?player ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name ?articleName .
    }}
    LIMIT {limit}
    OFFSET {offset}
    """
    response = requests.get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers=HEADERS,
        timeout=120,
    )
    response.raise_for_status()
    bindings = response.json()["results"]["bindings"]
    return [b["articleName"]["value"] for b in bindings]


def main():
    all_names = []
    offset = 0

    print("Fetching football player names from Wikidata...")
    while True:
        print(f"  Fetching batch offset={offset}...")
        try:
            batch = fetch_batch(offset, BATCH_SIZE)
        except Exception as e:
            print(f"  Error at offset {offset}: {e}. Stopping.")
            break

        if not batch:
            break

        all_names.extend(batch)
        print(f"  Got {len(batch)} names (total so far: {len(all_names)})")

        if len(batch) < BATCH_SIZE or len(all_names) >= TARGET:
            break

        offset += BATCH_SIZE
        time.sleep(2)  # be polite to Wikidata

    all_names = sorted(set(all_names))[:TARGET]  # deduplicate and cap at target

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(all_names))

    print(f"\nDone! {len(all_names)} unique player names saved to '{OUTPUT_FILE}'")
    print("Next step: run `python fetch_wikipedia.py` to download their Wikipedia pages.")


if __name__ == "__main__":
    main()
