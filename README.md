# Autonomous LinkedIn Content Manager

A five-agent [CrewAI](https://github.com/crewAIInc/crewAI) pipeline that researches, drafts, critiques, optimizes, and schedules a LinkedIn post for any topic.

```
Research  ->  Writing  ->  Critique  ->  Optimization  ->  Scheduling
```

| # | Agent | Tools | Output |
|---|-------|-------|--------|
| 1 | LinkedIn Trend Researcher | SerperDevTool, ScrapeWebsiteTool | Research brief: trends, angles, hashtags, hooks |
| 2 | LinkedIn Content Writer | (LLM only) | 150–300 word draft with hook, body, CTA |
| 3 | Content Quality Critic | (LLM only) | Scored critique with actionable fixes |
| 4 | LinkedIn Post Optimizer | (LLM only) | Final publish-ready post |
| 5 | LinkedIn Publishing Strategist | (LLM only) | Posting time, copy-paste post, hashtag and first-hour strategy |

The crew runs with `Process.sequential`, `verbose=True`, and `memory=True`. Each task also declares explicit `context`, so the optimizer sees both the draft and the critique, and the scheduler sees both the research and the final post.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python linkedin_content_manager.py    # first run creates .env and stops
```

Open the generated `.env` and paste your `OPENAI_API_KEY` and `SERPER_API_KEY` over the placeholders. `.env` is git-ignored, so your keys are never committed. Never rename or force-add it, and never paste keys into a file on GitHub.

## Run

```bash
python linkedin_content_manager.py                    # prompts for a topic
python linkedin_content_manager.py "AI in healthcare" # or pass it directly
```

The final publishing brief prints to the console. A markdown file with all five stage outputs is saved to `outputs/`.

## Configuration (`.env`)

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OPENAI_API_KEY` | yes | | LLM and memory embeddings |
| `SERPER_API_KEY` | yes | | Web search for the researcher |
| `OPENAI_MODEL_NAME` | no | `gpt-4o` | e.g. `gpt-4`, `gpt-4o-mini` |
| `LLM_TEMPERATURE` | no | `0.7` | |
| `MAX_RETRIES` | no | `3` | Retries on rate limits, timeouts, and 5xx errors |
| `OUTPUT_DIR` | no | `outputs` | Where content packages are saved |

## Error handling

- Missing keys, or placeholder keys left in `.env`, are reported before any API call.
- Authentication, quota, and unknown-model errors fail fast with a clear message.
- Rate limits, timeouts, network errors, and 5xx responses are retried with exponential backoff (2s, 4s, ...).
