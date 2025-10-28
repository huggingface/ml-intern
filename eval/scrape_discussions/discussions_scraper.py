import sys
import time
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Add parent directory to path to import models
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import Discussion, QuestionAndSolution

BASE_URL = "https://discuss.huggingface.co"


# configure retry decorator for your requests
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type(requests.HTTPError),
)
def safe_get(url, **kwargs):
    resp = requests.get(url, **kwargs)
    if resp.status_code == 422:
        # read retry‐after header if present
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            delay = float(retry_after)
        else:
            # fallback to guess
            delay = 30
        print(f"429 hit — waiting {delay} seconds...")
        time.sleep(delay)
        resp.raise_for_status()
    else:
        resp.raise_for_status()
    return resp


def get_solved_discussions(n_posts: int = 50):
    page = 1
    discussions = []
    while len(discussions) < n_posts:
        url = f"{BASE_URL}/search.json?q=status:solved+order:latest&page={page}"
        resp = safe_get(url)
        topics = resp.json()["topics"]
        if not topics:
            break
        for post in topics:
            discussions.append(
                Discussion(
                    title=post["fancy_title"],
                    url=f"{BASE_URL}/t/{post['slug']}/{post['id']}",
                    topic_id=post["id"],
                    category=post["category_id"],
                    created_at=post["created_at"],
                )
            )
            if len(discussions) >= n_posts:
                break
        page += 1
        time.sleep(0.5)  # simple pacing to avoid bursts
    return discussions


def get_qa_pair(discussions, start_idx: int = 0):
    for discussion in discussions[start_idx:]:
        resp = safe_get(discussion.url + ".json")
        data = resp.json()
        posts = data["post_stream"]["posts"]
        accepted_nr = min(
            max(data["accepted_answer"]["post_number"] - 1, 0), len(posts) - 1
        )
        question = posts[0]["cooked"]
        solution = posts[accepted_nr]["cooked"]
        yield QuestionAndSolution(
            discussion_title=discussion.title,
            discussion_url=discussion.url,
            discussion_topic_id=discussion.topic_id,
            discussion_category=discussion.category,
            discussion_created_at=discussion.created_at,
            question=question,
            solution=solution,
            thread=posts,
        )
        time.sleep(0.5)


if __name__ == "__main__":
    discussions = get_solved_discussions(n_posts=300)
    print(f"Fetched {len(discussions)} discussions")
    with open("qa_pairs.jsonl", "a") as f:
        for qa_pair in get_qa_pair(discussions):
            f.write(qa_pair.model_dump_json() + "\n")
