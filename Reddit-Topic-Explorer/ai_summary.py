import os
from dotenv import load_dotenv
from openai import OpenAI

# ======================================================
# OpenAI client initialization
# Uses API key from environment variable
# ======================================================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment")

client = OpenAI(api_key=api_key)


# ======================================================
# Low-level helper for calling the OpenAI Chat API
# Used internally by all summarization functions
# ======================================================
def _call_gpt(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    # Return the generated text response
    return resp.choices[0].message.content.strip()


# ======================================================
# Per-topic summary generation
# Summarizes a single BERTopic topic using keywords
# and a small sample of representative posts
# ======================================================
def summarize_topic(topic_name: str,
                    keywords: list[str],
                    representative_posts: list[dict]) -> str:
    """
    Per-topic summary, given a topic name, its keywords, and a few representative posts.
    representative_posts should have 'title' and 'selftext' fields.
    """

    # Use only the first few posts to keep the prompt concise
    sample_posts = representative_posts[:3]

    # Format representative posts into a readable text block
    posts_block = "\n\n".join(
        f"Title: {p.get('title', '')}\nText: {p.get('selftext', '')}"
        for p in sample_posts
    )

    # User-facing prompt describing the topic and examples
    user_prompt = f"""
Topic name: {topic_name}
Top keywords: {', '.join(keywords)}

Representative posts:
{posts_block}

Write a short paragraph (3–5 sentences) explaining what this topic is about,
what kinds of posts appear here, and what the general tone or sentiment is.
Avoid bullet points. Do not mention that this is from BERTopic or a model.
    """.strip()

    # System prompt constraining tone and role of the assistant
    system_prompt = (
        "You summarize Reddit topics for a research dashboard. "
        "Be concise, neutral, and informative."
    )

    return _call_gpt(system_prompt, user_prompt)


# ======================================================
# Wrapper used by demo.py for the topic_details page
# Converts topic metadata into a concise summary string
# ======================================================
def generate_topic_summary(keywords, rep_docs=None, subreddit=None, max_chars=500) -> str:
    """
    Wrapper used by demo.py for the topic_details page.

    - keywords: list of topic words from BERTopic
    - rep_docs: list of dicts from topic_meta in demo.py, with keys: 'title', 'body', 'url'
    - subreddit: optional subreddit name for extra context
    """

    # Ensure rep_docs is always iterable
    rep_docs = rep_docs or []

    # Normalize representative documents into the format expected by summarize_topic
    representative_posts = [
        {
            "title": d.get("title", ""),
            "selftext": d.get("body", ""),
        }
        for d in rep_docs
    ]

    # Derive a human-readable topic name from the top keywords
    topic_name = ", ".join(keywords[:3]) if keywords else "Unnamed topic"

    # Generate the raw summary text
    raw_summary = summarize_topic(topic_name, keywords, representative_posts)

    # Optionally truncate the summary to a maximum character length
    if max_chars is not None and len(raw_summary) > max_chars:
        raw_summary = raw_summary[:max_chars].rstrip()

    return raw_summary


# ======================================================
# Global subreddit-level summary
# Produces a high-level description for the dashboard
# ======================================================
def summarize_subreddit(subreddit: str,
                        topics: list[dict],
                        top_posts: list[dict]) -> str:
    """
    Global subreddit summary for the analytics dashboard.

    - topics: list of topic_meta dicts from demo.py (each has 'topic_name' and 'keywords')
    - top_posts: high-score raw Reddit posts (title + selftext)
    """

    # Build a compact list of discovered topics and their keywords
    topic_lines = []
    for t in topics[:15]:
        kws = ", ".join(t.get("keywords", [])[:5])
        label = t.get("topic_name") or t.get("name") or "Unnamed topic"
        topic_lines.append(f"- {label} ({kws})")
    topics_block = "\n".join(topic_lines)

    # Include a small sample of high-engagement posts for context
    posts_block = "\n\n".join(
        f"Title: {p.get('title','')}\nText: {p.get('selftext','')[:400]}"
        for p in top_posts[:5]
    )

    # User prompt describing the summarization task
    user_prompt = f"""
    You are analyzing subreddit r/{subreddit}.

    Here are the discovered topics:
    {topics_block}

    Here are some highly upvoted or representative posts:
    {posts_block}

    Task: Write a single cohesive summary (about 2–4 paragraphs) describing:
    - The main themes people talk about
    - The typical tone (e.g., supportive, sarcastic, anxious, etc.)
    - Any recurring conflicts, debates, or questions
    - What someone new to r/{subreddit} should expect to see

    Do NOT list the topics as bullets. Write flowing prose.
    Avoid mentioning that this came from topic modeling or any model.
    """.strip()

    # System prompt guiding style and audience
    system_prompt = (
        "You write high-level summaries of subreddits based on topics and posts. "
        "Be neutral, concise, and readable to non-experts."
    )

    return _call_gpt(system_prompt, user_prompt)


# ======================================================
# Image description helper
# Uses OpenAI vision-capable model to caption images
# ======================================================
def describe_image(image_url: str) -> str:
    """
    Use OpenAI's vision model to produce a short description of an image URL.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",  # same family as your other calls
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are helping with Reddit topic analysis. "
                                "Describe this image in 2–3 sentences. Focus on the main objects, "
                                "people, visible text, and overall setting. Be concise and neutral."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=120,
        )
        return (resp.choices[0].message.content or "").strip()

    # Graceful fallback if the image description fails
    except Exception as e:
        print("Image caption error:", e)
        return ""
