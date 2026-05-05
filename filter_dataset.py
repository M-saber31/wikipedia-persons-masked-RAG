import re
from datasets import load_dataset

def base_name(s):
    return re.sub(r'\s*\(.*\)\s*$', '', s).strip()

def has_footballer_qualifier(s):
    return bool(re.search(r'\(\s*.*footballer.*\)', s, re.IGNORECASE))

def has_any_qualifier(s):
    return bool(re.search(r'\(.*\)\s*$', s))

# Patterns that appear in the opening sentence of association football articles
INCLUDE_PAT = re.compile(
    r'professional footballer|professional football player|professional football manager|'
    r'football manager and former player|former professional player|former footballer|'
    r'retired footballer|football coach and former|'
    r'plays as a (goalkeeper|midfielder|defender|forward|striker|winger|'
    r'left.back|right.back|centre.back|centre.forward|full.back)|'
    r'played as a (goalkeeper|midfielder|defender|forward|striker|winger|'
    r'left.back|right.back|centre.back|centre.forward|full.back)',
    re.IGNORECASE
)

# Patterns that indicate a different sport — used to reject false positives
EXCLUDE_PAT = re.compile(
    r'American football|National Football League|\bNFL\b|quarterback|linebacker|'
    r'running back|tight end|wide receiver|offensive line|defensive end|placekicker|'
    r'Gaelic football|Gaelic footballer|Australian rules|Australian football|\bAFL\b|'
    r'Canadian football|\bCFL\b|hurler|hurling|basketball|rugby|cricket|gridiron',
    re.IGNORECASE
)

def load_player_set(path):
    with open(path, encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    player_set = set(names)
    player_base_set = set(base_name(n) for n in names)
    return player_set, player_base_set

def is_footballer(example, player_set, player_base_set):
    title = example['title']

    # --- title-based matching ---
    if title in player_set:
        return True
    b = base_name(title)
    if has_footballer_qualifier(title) and b in player_base_set:
        return True
    if not has_any_qualifier(title) and b in player_base_set:
        return True

    # --- text-based matching (opening sentence) ---
    first_line = example['text'].split('\n')[0] if example['text'] else ''
    if INCLUDE_PAT.search(first_line) and not EXCLUDE_PAT.search(first_line):
        return True

    return False

if __name__ == '__main__':
    player_set, player_base_set = load_player_set('./player_names.txt')

    print("Loading dataset...")
    ds = load_dataset('rcds/wikipedia-persons-masked', cache_dir='./dataset')
    train = ds['train']

    print(f"Total entries: {len(train)}")
    print("Filtering football players (title + text patterns)...")

    filtered = train.filter(
        is_footballer,
        fn_kwargs={'player_set': player_set, 'player_base_set': player_base_set},
        num_proc=1
    )

    print(f"Football players found: {len(filtered)}")
    print("\nSample titles:")
    for i in range(min(10, len(filtered))):
        print(f"  {filtered[i]['title']}")

    print("\nSaving filtered dataset...")
    filtered.save_to_disk('./dataset/football_players')
    print("Saved to ./dataset/football_players")
