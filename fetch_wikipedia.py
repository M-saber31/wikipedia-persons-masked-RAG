import os
import time
import pickle
import wikipedia
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

DATA_DIR = "wiki_data"
CHECKPOINT_FILE = os.path.join(DATA_DIR, ".checkpoint.pkl")

MAX_WORKERS = 8
SAVE_CHECKPOINT_EVERY = 200

lock = Lock()


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'rb') as f:
            return pickle.load(f)
    return set()


def save_checkpoint(done):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump(done, f)


def safe_filename(title):
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '(', ')']:
        title = title.replace(ch, '')
    return title.replace(' ', '_') + '.txt'


def fetch_single(title):
    try:
        page = wikipedia.page(title, auto_suggest=False)
        filepath = os.path.join(DATA_DIR, safe_filename(title))
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(page.content)
        return title, None
    except Exception as e:
        return title, str(e)


def fetch_pages(pages):
    os.makedirs(DATA_DIR, exist_ok=True)

    done = load_checkpoint()
    remaining = [p for p in pages if p not in done]

    print(f"Total: {len(pages)} | Already fetched: {len(done)} | Remaining: {len(remaining)}")

    failed = []
    counter = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_single, title): title for title in remaining}

        for future in tqdm(as_completed(futures), total=len(futures), unit="page", dynamic_ncols=True):
            title, error = future.result()

            with lock:
                if error:
                    tqdm.write(f"FAILED: {title} — {error}")
                    failed.append(title)
                else:
                    done.add(title)
                    counter += 1
                    if counter % SAVE_CHECKPOINT_EVERY == 0:
                        save_checkpoint(done)

    save_checkpoint(done)

    if failed:
        with open('failed_pages.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(failed))
        print(f"\n{len(failed)} pages failed — saved to failed_pages.txt")

    print(f"\nDone! {len(done)} pages saved to '{DATA_DIR}/'")
    print("Run your app with: streamlit run main.py")


if __name__ == '__main__':
    if os.path.exists('player_names.txt'):
        with open('player_names.txt', 'r', encoding='utf-8') as f:
            pages = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(pages)} names from player_names.txt")
    else:
        print("player_names.txt not found — using default pages.")
        pages = [
            "Artificial intelligence",
            "Machine learning",
            "Deep learning",
            "Neural network",
            "Convolutional neural network",
            "Reinforcement learning",
            "Supervised learning",
            "Unsupervised learning",
            "Natural language processing",
            "Transformer (machine learning model)",
            "ChatGPT",
            "OpenAI",
            "Computer vision",
            "Generative adversarial network",
            "Support vector machine",
            "Decision tree learning",
            "Gradient boosting",
            "Bayesian network",
            "K-nearest neighbors algorithm",
            "Feature engineering",
        ]

    fetch_pages(pages)
