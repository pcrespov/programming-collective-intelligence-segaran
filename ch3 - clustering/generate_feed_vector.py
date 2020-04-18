import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import feedparser
from feedparser import FeedParserDict

current_dir = Path(sys.argv[0] if __name__ == "__main__" else __file__).resolve().parent
data_dir = current_dir / "data"

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)

tag_pattern = re.compile(r"<[^>]+>")
non_alpha_pattern = re.compile(r"[^A-Z^a-z]+")


def zero() -> int:
    return 0


def get_feeds() -> List[str]:
    feeds = []
    with open(data_dir / "feedlist.txt", "rt") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("http"):
                feeds.append(line)
    return feeds


def extract_words(html_text: str) -> List[str]:
    txt = tag_pattern.sub("", html_text)  # deletes tags
    words = non_alpha_pattern.split(txt)  # split by non-alphas
    return [word.lower() for word in words if word.strip()]


def get_word_counts(url: str) -> Tuple[str, Dict[str, int]]:
    log.info("Parsing %s", url)

    data: FeedParserDict = feedparser.parse(url)

    total_wc = Counter()

    for entry in data.entries:
        summary = entry.summary if "summary" in entry else entry.description

        words = extract_words(entry.title + " " + summary)
        wc = Counter(words)
        total_wc.update(wc)

    return data.feed.title, total_wc


def generate():
    # word count per blog
    wc_per_blog: Dict[str, Counter] = {}

    # blog count per word (how many blogs a word appers)
    apcount = Counter()

    feeds = {}
    target_feeds = get_feeds()

    log.info("Parsing %d feeds", len(target_feeds))
    for url in target_feeds:
        try:
            blog_name, wc = get_word_counts(url)
            wc_per_blog[blog_name] = wc
            apcount.update(wc.keys())
            feeds[blog_name] = url
        except Exception:  # pylint: disable=broad-except
            log.error("%s failed", url)

    log.info("%3.1f %% of feeds worked", len(feeds) / len(target_feeds))

    return feeds, wc_per_blog, apcount


def filter_and_pack(
    feeds: Dict[str, str], wc_per_blog: Dict[str, Counter], apcount: Counter
):
    blog_count = len(wc_per_blog)

    log.info("Filtering word list")

    wordlist = []
    for word, bc in apcount.items():
        # filter very repetitive or very strange words
        frac = bc / blog_count
        if 0.1 < frac < 0.5 and len(word) > 3:
            wordlist.append(word)

    log.info("%3.2f %% of words kept", len(wordlist) / len(apcount) * 100)

    log.info("Packing into json ...")
    data = {
        "words": wordlist,
        "blogs": [
            {
                "name": blog,
                "url": feeds[blog],
                "wc": [wc.get(word, 0) for word in wordlist],
            }
            for blog, wc in wc_per_blog.items()
        ],
    }
    return data


def load_data() -> Dict:
    with open(data_dir / "blogdata.json", "rt") as fh:
        data = json.load(fh)
    return data


def main():
    feeds, wc_per_blog, apcount = generate()

    data = filter_and_pack(feeds, wc_per_blog, apcount)

    with open(data_dir / "blogdata.json", "wt") as fh:
        json.dump(data, fh, indent=1)


if __name__ == "__main__":
    main()
