"""
GitHub Find Examples Tool - Discover examples, tutorials, and guides for any library

Lists all files in a repository and performs deterministic keyword search.
"""

import os
from typing import Any, Dict, List

import requests
from thefuzz import fuzz

from agent.search import TantivyTextIndex, chunk_code
from agent.search.cache import read_json, stable_key, write_json
from agent.tools.types import ToolResult

# In order of priority (lower index = higher priority for sorting)
EXAMPLE_PATTERNS = [
    "scripts",
    # General example patterns (catch-all, lower priority)
    "examples",
    "example",
    # Notebook patterns
    "notebooks",
    "notebook",
    # Tutorial/learning patterns
    "tutorials",
    "tutorial",
    "quickstart",
    "walkthroughs",
    "walkthrough",
    # Cookbook/recipe patterns
    "cookbook",
    "cookbooks",
    "recipes",
    "recipe",
    # Demo/sample patterns
    "demos",
    "demo",
    "samples",
    "sample",
    # Other patterns
    "guides",
    "guide",
    "getting-started",
    "getting_started",
    "playground",
    "howto",
    "how-to",
    "use-cases",
    "usecases",
    "use_cases",
    "sandbox",
    "showcase",
]

CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".md",
    ".mdx",
    ".yaml",
    ".yml",
    ".toml",
}
MAX_INDEXED_EXAMPLE_FILES = 50
MAX_INDEXED_FILE_BYTES = 400_000


def _github_headers(token: str, *, raw: bool = False) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw"
        if raw
        else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _github_get(
    url: str,
    token: str,
    *,
    raw: bool = False,
    **kwargs,
) -> requests.Response:
    response = requests.get(
        url,
        headers=_github_headers(token, raw=raw),
        **kwargs,
    )
    if response.status_code == 401 and token:
        return requests.get(
            url,
            headers=_github_headers("", raw=raw),
            **kwargs,
        )
    return response


def _get_repo_tree(org: str, repo: str, token: str) -> tuple[List[Dict[str, Any]], str]:
    """Get all files in a repository recursively. Returns (files, error_message)"""
    cache_key = stable_key(org, repo)
    cached = read_json("github-trees", cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("files"), list):
        return cached["files"], ""

    full_repo = f"{org}/{repo}"

    # Get default branch
    try:
        response = _github_get(
            f"https://api.github.com/repos/{full_repo}",
            token,
            timeout=10,
        )
        if response.status_code == 404:
            return [], "not_found"
        if response.status_code != 200:
            return [], f"API error: {response.status_code}"

        repo_data = response.json()
        default_branch = repo_data.get("default_branch", "main")
    except Exception as e:
        return [], f"Error fetching repo: {str(e)}"

    # Get repository tree recursively
    try:
        response = _github_get(
            f"https://api.github.com/repos/{full_repo}/git/trees/{default_branch}",
            token,
            params={"recursive": "1"},
            timeout=30,
        )
        if response.status_code != 200:
            return [], f"Error fetching tree: {response.status_code}"

        data = response.json()
        tree = data.get("tree", [])

        # Filter to only include files (not directories)
        files = [
            {
                "path": item["path"],
                "ref": item["sha"],
                "size": item.get("size", 0),
                "branch": default_branch,
                "url": f"https://github.com/{full_repo}/blob/{default_branch}/{item['path']}",
            }
            for item in tree
            if item["type"] == "blob"
        ]

        write_json(
            "github-trees",
            cache_key,
            {"default_branch": default_branch, "files": files},
        )
        return files, ""
    except Exception as e:
        return [], f"Error processing tree: {str(e)}"


def _search_similar_repos(org: str, repo: str, token: str) -> List[Dict[str, Any]]:
    """Search for similar repository names in the organization"""
    # Search for repos in the org with similar name
    query = f"org:{org} {repo}"

    try:
        response = _github_get(
            "https://api.github.com/search/repositories",
            token,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 10},
            timeout=30,
        )

        if response.status_code != 200:
            return []

        data = response.json()
        items = data.get("items", [])

        return [
            {
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count", 0),
                "url": item.get("html_url"),
            }
            for item in items
        ]
    except Exception:
        return []


def _score_against_example_patterns(file_path: str) -> int:
    """Score file against example patterns using token_set_ratio"""
    scores = []
    for pattern in EXAMPLE_PATTERNS:
        score = fuzz.token_set_ratio(pattern.lower(), file_path.lower())
        scores.append(score)
    return max(scores) if scores else 0


def _score_against_keyword(file_path: str, keyword: str) -> int:
    """Calculate fuzzy match score for a file path against a keyword"""
    # Use partial_ratio for substring matching (good for paths)
    # Also check token_set_ratio for word-level matching
    partial_score = fuzz.partial_ratio(keyword.lower(), file_path.lower())
    token_score = fuzz.token_set_ratio(keyword.lower(), file_path.lower())

    # Return the higher of the two
    return max(partial_score, token_score)


def _is_indexable_example_file(file_path: str, size: int) -> bool:
    _, ext = os.path.splitext(file_path.lower())
    return ext in CODE_EXTENSIONS and 0 < size <= MAX_INDEXED_FILE_BYTES


def _rank_index_candidate(
    file: Dict[str, Any], keyword: str
) -> tuple[int, int, int, int, str]:
    in_examples_dir, pattern_priority, path_depth = _get_pattern_priority(file["path"])
    keyword_score = _score_against_keyword(file["path"], keyword) if keyword else 0
    return (-keyword_score, in_examples_dir, pattern_priority, path_depth, file["path"])


def _fetch_file_content_cached(
    org: str,
    repo: str,
    file: Dict[str, Any],
    token: str,
) -> str | None:
    cache_key = stable_key(org, repo, file.get("path"), file.get("ref"))
    cached = read_json("github-files", cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("content"), str):
        return cached["content"]

    url = f"https://api.github.com/repos/{org}/{repo}/contents/{file['path']}"
    params = {"ref": file.get("branch", "HEAD")}
    try:
        response = _github_get(url, token, raw=True, params=params, timeout=20)
        if response.status_code != 200:
            return None
        content = response.text
    except Exception:
        return None

    write_json("github-files", cache_key, {"content": content})
    return content


def _search_example_snippets(
    keyword: str,
    org: str,
    repo: str,
    files: list[Dict[str, Any]],
    token: str,
    *,
    limit: int,
) -> list[Dict[str, Any]]:
    candidates = _get_index_candidates(files, keyword)
    if not candidates:
        return []

    cache_key = stable_key(
        org,
        repo,
        "snippet-docs",
        *[f"{file.get('path')}@{file.get('ref')}" for file in candidates],
    )
    cached_docs = read_json("github-snippet-docs", cache_key)
    if isinstance(cached_docs, list) and all(
        isinstance(item, dict) for item in cached_docs
    ):
        docs = cached_docs
    else:
        docs = _build_example_snippet_docs(org, repo, candidates, token)
        write_json("github-snippet-docs", cache_key, docs)

    if not docs:
        return []

    index = TantivyTextIndex(
        text_fields=["path", "heading", "content"],
        stored_fields=[
            "path",
            "url",
            "ref",
            "size",
            "heading",
            "content",
            "line_start",
            "line_end",
        ],
        field_boosts={"path": 3.0, "heading": 2.0, "content": 1.0},
    )
    index.add_documents(docs)
    hits, _ = index.search(keyword, limit=limit)
    return [
        {
            **hit.fields,
            "score": round(hit.score, 2),
        }
        for hit in hits
    ]


def _build_example_snippet_docs(
    org: str,
    repo: str,
    candidates: list[Dict[str, Any]],
    token: str,
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for file in candidates:
        content = _fetch_file_content_cached(org, repo, file, token)
        if not content:
            continue
        for chunk in chunk_code(content):
            docs.append(
                {
                    "path": file["path"],
                    "url": file["url"],
                    "ref": file["ref"],
                    "size": str(file.get("size", 0)),
                    "heading": chunk.title,
                    "content": chunk.text,
                    "line_start": str(chunk.line_start),
                    "line_end": str(chunk.line_end),
                }
            )
    return docs


def _get_index_candidates(
    files: list[Dict[str, Any]], keyword: str
) -> list[Dict[str, Any]]:
    return sorted(
        [
            file
            for file in files
            if _is_indexable_example_file(file["path"], int(file.get("size", 0)))
        ],
        key=lambda file: _rank_index_candidate(file, keyword),
    )[:MAX_INDEXED_EXAMPLE_FILES]


def _excerpt_around_query(content: str, query: str, *, max_chars: int = 900) -> str:
    if len(content) <= max_chars:
        return content

    terms = [
        term.lower()
        for term in query.replace("_", " ").split()
        if len(term.strip()) >= 3
    ]
    content_lower = content.lower()
    first_match = min(
        (index for term in terms if (index := content_lower.find(term)) >= 0),
        default=0,
    )
    start = max(0, first_match - max_chars // 4)
    end = min(len(content), start + max_chars)
    if end - start < max_chars:
        start = max(0, end - max_chars)

    excerpt = content[start:end]
    if start > 0:
        excerpt = "...\n" + excerpt
    if end < len(content):
        excerpt += "\n..."
    return excerpt


def _get_pattern_priority(file_path: str) -> tuple[int, int, int]:
    """
    Get priority of a file path based on which example pattern directory it's in.

    Returns: (in_examples_dir, pattern_priority, path_depth)
    - in_examples_dir: 0 if in examples/ directory, 1 otherwise (lower is better)
    - pattern_priority: Index in EXAMPLE_PATTERNS (lower is better), or 999 if no match
    - path_depth: Number of path segments (lower is better)

    Note: Prioritizes files in "examples/" directory first, then by most specific pattern match.
    E.g., "examples/scripts/train.py" is better than "scripts/util.py"
    """
    path_lower = file_path.lower()
    path_parts = path_lower.split("/")

    # Check if file is in examples/ directory (highest priority)
    in_examples_dir = 0 if (path_parts[0] in ["examples", "example"]) else 1

    # Find ALL matching patterns and use the best (lowest index) one
    # But prefer deeper matches (more specific) over shallow ones
    best_priority = 999
    best_depth_at_match = -1

    for i, pattern in enumerate(EXAMPLE_PATTERNS):
        # Check if pattern appears as a directory component in the path
        if pattern in path_parts:
            # Find the depth where this pattern appears (rightmost occurrence)
            depth = len(path_parts) - 1 - path_parts[::-1].index(pattern)

            # Prefer deeper matches, or better priority if at same depth
            if depth > best_depth_at_match or (
                depth == best_depth_at_match and i < best_priority
            ):
                best_priority = i
                best_depth_at_match = depth

    return (in_examples_dir, best_priority, len(path_parts))


def _handle_repo_tree_errors(
    all_files: List[Dict[str, Any]],
    error: str,
    org: str,
    repo: str,
    token: str,
) -> ToolResult | None:
    """Handle errors from repo tree fetch. Returns ToolResult if error, None if OK."""
    if error == "not_found":
        similar_repos = _search_similar_repos(org, repo, token)

        if not similar_repos:
            return {
                "formatted": f"Repository '{org}/{repo}' not found and no similar repositories found.",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

        # Format similar repos
        lines = [f"**Repository '{org}/{repo}' not found. Similar repositories:**\n"]
        for i, r in enumerate(similar_repos, 1):
            lines.append(f"{i}. **{r['full_name']}** (⭐ {r['stars']:,} stars)")
            if r["description"]:
                desc = (
                    r["description"][:100] + "..."
                    if len(r["description"]) > 100
                    else r["description"]
                )
                lines.append(f"   {desc}")
            lines.append(f"   {r['url']}\n")

        return {
            "formatted": "\n".join(lines),
            "totalResults": len(similar_repos),
            "resultsShared": len(similar_repos),
            "isError": True,
        }

    if error:
        return {
            "formatted": f"Error accessing repository '{org}/{repo}': {error}",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    if not all_files:
        return {
            "formatted": f"No files found in repository '{org}/{repo}'",
            "totalResults": 0,
            "resultsShared": 0,
        }

    return None


def find_examples(
    keyword: str = "",
    repo: str = "",
    org: str = "huggingface",
    max_results: int = 10,
    min_score: int = 80,
) -> ToolResult:
    """
    Find example files in a repository using fuzzy matching.

    Args:
        keyword: Keyword to fuzzy match against file paths (e.g., "grpo")
        repo: Repository name (e.g., "trl")
        org: GitHub organization (default: "huggingface")
        max_results: Maximum number of results (default 50)
        min_score: Minimum fuzzy match score (0-100, default 60)

    Returns:
        ToolResult with matching files, or similar repos if repo not found
    """
    token = os.environ.get("GITHUB_TOKEN", "")

    if not repo:
        return {
            "formatted": "Error: repo parameter is required",
            "totalResults": 0,
            "resultsShared": 0,
            "isError": True,
        }

    # Get all files in the repository
    all_files, error = _get_repo_tree(org, repo, token)

    # Handle errors (not found, API errors, empty repo)
    if error_result := _handle_repo_tree_errors(all_files, error, org, repo, token):
        return error_result

    # Step 1: Filter files by example patterns (score >= 60)
    example_threshold = 60
    example_files = []
    for file in all_files:
        example_score = _score_against_example_patterns(file["path"])
        if example_score >= example_threshold:
            example_files.append({**file, "example_score": example_score})

    if not example_files:
        return {
            "formatted": f"No example files found in {org}/{repo} (no files match example patterns with score >= {example_threshold}).",
            "totalResults": 0,
            "resultsShared": 0,
        }

    snippet_hits: list[Dict[str, Any]] = []

    # Step 2: If keyword provided, score paths and search file contents.
    if keyword:
        snippet_hits = _search_example_snippets(
            keyword,
            org,
            repo,
            example_files,
            token,
            limit=max(max_results * 2, 10),
        )

        scored_files = []
        for file in example_files:
            keyword_score = _score_against_keyword(file["path"], keyword)
            if keyword_score >= min_score:
                scored_files.append({**file, "score": keyword_score})

        if snippet_hits:
            snippet_scores: dict[str, float] = {}
            for hit in snippet_hits:
                path = hit.get("path", "")
                snippet_scores[path] = max(
                    snippet_scores.get(path, 0.0), float(hit["score"])
                )

            seen_paths = {file["path"] for file in scored_files}
            for file in example_files:
                if file["path"] in snippet_scores and file["path"] not in seen_paths:
                    scored_files.append(
                        {
                            **file,
                            "score": min(100, int(70 + snippet_scores[file["path"]] * 10)),
                            "content_score": snippet_scores[file["path"]],
                        }
                    )
                    seen_paths.add(file["path"])

        if not scored_files:
            return {
                "formatted": f"No files found in {org}/{repo} matching keyword '{keyword}' (min score: {min_score}) among {len(example_files)} example files.",
                "totalResults": 0,
                "resultsShared": 0,
            }

        # Prefer files with content hits, then path similarity.
        scored_files.sort(
            key=lambda x: (float(x.get("content_score", 0.0)), x["score"]),
            reverse=True,
        )
    else:
        # No keyword: prioritize by pattern directory, then path depth
        scored_files = []
        for file in example_files:
            in_examples_dir, pattern_priority, path_depth = _get_pattern_priority(
                file["path"]
            )
            scored_files.append(
                {
                    **file,
                    "score": file["example_score"],
                    "in_examples_dir": in_examples_dir,
                    "pattern_priority": pattern_priority,
                    "path_depth": path_depth,
                }
            )

        if not scored_files:
            return {
                "formatted": f"No example files found in {org}/{repo}.",
                "totalResults": 0,
                "resultsShared": 0,
            }

        # Sort by: 1) files in examples/ dir first, 2) pattern priority (scripts > datasets > etc), 3) path depth, 4) path name
        scored_files.sort(
            key=lambda x: (
                x["in_examples_dir"],
                x["pattern_priority"],
                x["path_depth"],
                x["path"],
            )
        )

    # Limit results
    results = scored_files[:max_results]

    # Format output
    keyword_desc = f" matching '{keyword}'" if keyword else ""
    lines = [f"**Found {len(results)} example files in {org}/{repo}{keyword_desc}:**"]
    if len(scored_files) > max_results:
        lines[0] += f" (showing {max_results} of {len(scored_files)})"
    lines.append("")

    for i, file in enumerate(results, 1):
        lines.append(f"{i}. **{file['path']}**")
        lines.append(f"   Size: {file['size']:,} bytes | Ref: {file['ref'][:7]}")
        lines.append(f"   URL: {file['url']}")

        # Copyable parameters for read_file tool
        read_params = f"{{'repo': '{org}/{repo}', 'path': '{file['path']}'}}"
        lines.append(f"   To read, use: {read_params}")
        lines.append("")

    if snippet_hits:
        lines.append("## Best indexed code snippets")
        lines.append(
            "Use these line ranges with `github_read_file` before reading whole files."
        )
        lines.append("")
        for i, hit in enumerate(snippet_hits[:max_results], 1):
            path = hit["path"]
            line_start = hit["line_start"]
            line_end = hit["line_end"]
            excerpt = _excerpt_around_query(hit["content"], keyword)
            lines.append(f"{i}. **{path}:{line_start}-{line_end}**")
            lines.append(f"   Relevance score: {hit['score']:.2f}")
            lines.append(
                f"   To read exactly: {{'repo': '{org}/{repo}', 'path': '{path}', 'line_start': {line_start}, 'line_end': {line_end}}}"
            )
            lines.append("   ```")
            lines.append(excerpt)
            lines.append("   ```")
            lines.append("")

    return {
        "formatted": "\n".join(lines),
        "totalResults": len(results),
        "resultsShared": len(results),
    }


# Tool specification
GITHUB_FIND_EXAMPLES_TOOL_SPEC = {
    "name": "github_find_examples",
    "description": (
        "Find working example scripts in GitHub repositories (from a list of predetermined directories e.g. examples/, scripts/, tutorials/, etc.). "
        "Uses fuzzy path matching plus Tantivy content search over indexed code snippets when a keyword is provided.\n\n"
        "MANDATORY before writing any ML training, fine-tuning, or inference code. "
        "Your internal knowledge of library APIs is outdated — working examples show current API patterns.\n\n"
        "Sequence: github_find_examples → github_read_file with the returned line_start/line_end ranges → implement based on what you found.\n\n"
        "Skip this only for: simple data queries, status checks, non-code tasks.\n\n"
        "Examples:\n"
        "  {keyword: 'sft', repo: 'trl'} → finds examples/scripts/sft.py\n"
        "  {keyword: 'grpo', repo: 'trl'} → finds GRPO training examples\n"
        "  {repo: 'trl', max_results: 20} → lists all available training method examples"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "keyword": {
                "type": "string",
                "description": "Keyword to search against file paths and indexed code snippets (e.g., 'grpo', 'sft', 'dataset_text_field').",
            },
            "repo": {
                "type": "string",
                "description": "Repository name (e.g., 'trl', 'transformers'). Required.",
            },
            "org": {
                "type": "string",
                "description": "GitHub organization or username. Default: 'huggingface'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 50.",
            },
            "min_score": {
                "type": "integer",
                "description": "Minimum fuzzy match score (0-100). Default: 60.",
            },
        },
        "required": ["repo"],
    },
}


async def github_find_examples_handler(arguments: Dict[str, Any]) -> tuple[str, bool]:
    """Handler for agent tool router"""
    try:
        result = find_examples(
            keyword=arguments.get("keyword", ""),
            repo=arguments["repo"],
            org=arguments.get("org", "huggingface"),
            max_results=arguments.get("max_results", 50),
            min_score=arguments.get("min_score", 60),
        )
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        return f"Error finding examples: {str(e)}", False
