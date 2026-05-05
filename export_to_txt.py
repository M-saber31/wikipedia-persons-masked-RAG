import os
import re
from datasets import load_from_disk

DATA_DIR = 'wiki_data_filtered'

def safe_filename(title):
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']:
        title = title.replace(ch, '')
    return title.replace(' ', '_') + '.txt'

if __name__ == '__main__':
    print("Loading filtered dataset...")
    ds = load_from_disk('./dataset/football_players')
    print(f"Entries to export: {len(ds)}")

    os.makedirs(DATA_DIR, exist_ok=True)

    skipped = 0
    exported = 0
    for example in ds:
        title = example['title']
        text = example['text'].strip()
        if not text:
            skipped += 1
            continue
        filepath = os.path.join(DATA_DIR, safe_filename(title))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        exported += 1

    print(f"Done! Exported {exported} files to '{DATA_DIR}/'  ({skipped} skipped — empty text)")
