#!/usr/bin/env python3
"""
Autonomous LinkedIn Content Manager
===================================

A five-agent CrewAI pipeline that turns a topic/niche into a polished,
publish-ready LinkedIn post with scheduling recommendations.

Pipeline (strictly sequential):

    Research  ->  Writing  ->  Critique  ->  Optimization  ->  Scheduling

Each task's output is passed to the next agent as context (CrewAI's
sequential process does this automatically, and we also declare explicit
`context=` links where an agent needs more than the immediately preceding
output). Crew memory is enabled so agents can recall earlier findings.

Usage:
    1. pip install -r requirements.txt
    2. Open .env (created on first run) and paste your OPENAI_API_KEY and
       SERPER_API_KEY over the placeholders
    3. python linkedin_content_manager.py                 # prompts for a topic
       python linkedin_content_manager.py "AI in healthcare"  # topic as argument
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ENV_FILE = SCRIPT_DIR / ".env"

# Written to .env on first run so the user has a file to paste keys into.
# .env is git-ignored, so keys entered there are never committed.
ENV_TEMPLATE = """\
# Paste your API keys below, replacing the placeholder values.
# This file is git-ignored: keep it that way and never commit real keys.

# Required: OpenAI API key (https://platform.openai.com/api-keys)
OPENAI_API_KEY=your_openai_api_key_here

# Required: Serper.dev API key for Google search (https://serper.dev)
SERPER_API_KEY=your_serper_api_key_here

# Optional overrides
# OPENAI_MODEL_NAME=gpt-4o        # or gpt-4, gpt-4o-mini
# LLM_TEMPERATURE=0.7
# MAX_RETRIES=3                   # retries for rate limits / transient API errors
# OUTPUT_DIR=outputs
"""

# Create .env from the template if it does not exist yet.
if not ENV_FILE.exists():
    try:
        ENV_FILE.write_text(ENV_TEMPLATE, encoding="utf-8")
    except OSError:
        pass  # read-only checkout; keys can still come from the shell

# Load variables from .env (next to this script first, then the CWD) before
# anything reads the environment. Existing shell variables are not overridden.
load_dotenv(ENV_FILE)
load_dotenv()

REQUIRED_ENV_VARS = {
    "OPENAI_API_KEY": "OpenAI API key (https://platform.openai.com/api-keys)",
    "SERPER_API_KEY": "Serper.dev API key for web search (https://serper.dev)",
}

# Model and run settings, overridable from .env.
OPENAI_MODEL = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", SCRIPT_DIR / "outputs"))

# Human-readable stage names, in pipeline order. Used for console separators.
STAGES = ["Research", "Writing", "Critique", "Optimization", "Scheduling"]

SEPARATOR_WIDTH = 78


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def print_banner(title: str, char: str = "=") -> None:
    """Print a full-width section separator with a centred title."""
    line = char * SEPARATOR_WIDTH
    print(f"\n{line}\n{title.center(SEPARATOR_WIDTH)}\n{line}\n", flush=True)


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    print(f"\n[ERROR] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def validate_environment() -> None:
    """Exit with a helpful message if any required API key is missing.

    Placeholder values left in the generated .env are treated as missing
    so the user gets a clear error instead of a 401 mid-run.
    """
    missing = []
    for var, description in REQUIRED_ENV_VARS.items():
        value = os.getenv(var, "").strip()
        if not value or value.startswith("your_") or value.endswith("_here"):
            missing.append(f"  - {var}: {description}")

    if missing:
        print_error(
            "Missing required environment variables:\n"
            + "\n".join(missing)
            + f"\n\nOpen {ENV_FILE} and replace the placeholder values with "
            "your keys, then run the script again."
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def build_agents(llm):
    """Create the five specialised agents.

    Only the Trend Researcher has tools (web search + scraping); the other
    four are LLM-only and work from the context passed down the pipeline.
    """
    from crewai import Agent
    from crewai_tools import ScrapeWebsiteTool, SerperDevTool

    # Web search via Serper.dev (reads SERPER_API_KEY from the environment)
    # and a generic scraper for reading articles the search surfaces.
    search_tool = SerperDevTool()
    scrape_tool = ScrapeWebsiteTool()

    # 1. Trend Researcher: the only agent with external tools.
    trend_researcher = Agent(
        role="LinkedIn Trend Researcher",
        goal=(
            "Research the latest trending topics, hashtags, and content themes "
            "on LinkedIn for a given niche"
        ),
        backstory=(
            "You are an expert social media researcher who monitors LinkedIn "
            "trends, viral posts, and industry news every day. You know what "
            "drives engagement on LinkedIn: which angles spark debate, which "
            "hooks earn the 'see more' click, and which hashtags actually reach "
            "an audience. You back your recommendations with current sources."
        ),
        tools=[search_tool, scrape_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=15,
    )

    # 2. Content Writer: turns the research brief into a first draft.
    content_writer = Agent(
        role="LinkedIn Content Writer",
        goal="Write engaging, high-quality LinkedIn posts based on the research provided",
        backstory=(
            "You are a seasoned LinkedIn ghostwriter for industry leaders. You "
            "understand the LinkedIn algorithm inside out and are an expert in "
            "hook writing, storytelling, and CTA placement. Your voice is "
            "conversational yet professional: posts read like a smart colleague "
            "sharing hard-won insight, never like a press release."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # 3. Content Critic: scores the draft and lists concrete fixes.
    content_critic = Agent(
        role="Content Quality Critic",
        goal=(
            "Review LinkedIn posts and provide detailed, constructive feedback on "
            "engagement potential, tone, structure, clarity, hook strength, and "
            "CTA effectiveness"
        ),
        backstory=(
            "You are a harsh but fair editor who has reviewed thousands of "
            "LinkedIn posts. You can tell at a glance whether a post will get "
            "10 likes or 10k+ impressions, and you explain exactly why. Your "
            "feedback is always specific and actionable: you quote the weak "
            "line and propose the stronger alternative."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # 4. Content Optimizer: rewrites the draft using the critique.
    content_optimizer = Agent(
        role="LinkedIn Post Optimizer",
        goal="Rewrite posts incorporating critic feedback to maximize LinkedIn engagement",
        backstory=(
            "You are a LinkedIn growth expert and conversion copywriter. You "
            "are a master of LinkedIn formatting: short lines, strategic line "
            "breaks, purposeful emoji, hashtag optimisation, proven hook "
            "patterns, and mobile-first readability. You never lose the "
            "author's message while making it impossible to scroll past."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    # 5. Scheduling Agent: decides when to post and packages the final output.
    scheduling_agent = Agent(
        role="LinkedIn Publishing Strategist",
        goal=(
            "Determine the optimal posting time, finalize formatting with "
            "hashtags, and create publishing-ready output with scheduling "
            "recommendations"
        ),
        backstory=(
            "You are a LinkedIn analytics expert. You understand optimal "
            "posting times by industry, audience timezone, and day of week, "
            "and you know how the first hour after publishing shapes a post's "
            "reach. You hand creators a brief they can act on immediately."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return (
        trend_researcher,
        content_writer,
        content_critic,
        content_optimizer,
        scheduling_agent,
    )


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def build_tasks(agents):
    """Create the five sequential tasks, one per agent.

    `{topic}` placeholders are filled in by CrewAI from the inputs passed to
    `crew.kickoff(inputs={"topic": ...})`.
    """
    from crewai import Task

    (
        trend_researcher,
        content_writer,
        content_critic,
        content_optimizer,
        scheduling_agent,
    ) = agents

    # 1. Research
    research_task = Task(
        name="Research",
        description=(
            "Research the latest trends, viral content patterns, and hot topics "
            "on LinkedIn for the niche: {topic}.\n"
            "Identify 3-5 trending angles, relevant hashtags, and content hooks "
            "that are currently performing well. Use web search to find recent "
            "news, discussions, and data points (prefer sources from the last "
            "few months) and cite the sources you rely on."
        ),
        expected_output=(
            "A structured research brief in markdown with these sections:\n"
            "1. Trending Topics (3-5, each with a one-line why-it's-hot)\n"
            "2. Suggested Post Angles (3-5)\n"
            "3. Top-Performing Hashtags (8-12, ranked by relevance)\n"
            "4. Content Hook Ideas (at least 5 ready-to-use opening lines)\n"
            "5. Key Facts & Sources (bullet list with URLs)"
        ),
        agent=trend_researcher,
    )

    # 2. Writing (receives the research brief)
    writing_task = Task(
        name="Writing",
        description=(
            "Using the research brief, write a compelling LinkedIn post about "
            "{topic}.\n"
            "Requirements:\n"
            "- A strong hook in the first 2 lines (this is all readers see "
            "before 'see more')\n"
            "- A storytelling or value-driven body\n"
            "- A clear call to action (CTA)\n"
            "- 150-300 words\n"
            "- Use the trending angles and hooks from the research"
        ),
        expected_output=(
            "A complete LinkedIn post draft with a hook, body, and CTA, "
            "followed by a line of suggested hashtags and the word count."
        ),
        agent=content_writer,
        context=[research_task],
    )

    # 3. Critique (receives the draft)
    critique_task = Task(
        name="Critique",
        description=(
            "Critically review the LinkedIn post draft. Evaluate:\n"
            "- Hook strength (will people click 'see more'?)\n"
            "- Storytelling quality\n"
            "- Engagement potential\n"
            "- CTA effectiveness\n"
            "- Tone consistency\n"
            "- LinkedIn formatting\n"
            "- Viral potential\n"
            "Provide an overall score out of 10 and specific improvement "
            "suggestions, quoting the lines you would change."
        ),
        expected_output=(
            "A detailed critique in markdown containing: an overall score out "
            "of 10, a score per criterion above, key strengths, key weaknesses, "
            "and a numbered list of specific, actionable improvement "
            "suggestions (with rewritten examples where useful)."
        ),
        agent=content_critic,
        context=[writing_task],
    )

    # 4. Optimization (receives both the draft and the critique)
    optimization_task = Task(
        name="Optimization",
        description=(
            "Take the original LinkedIn post and the critic's feedback. Rewrite "
            "the post incorporating ALL of the feedback:\n"
            "- Improve the hook\n"
            "- Tighten the copy\n"
            "- Optimize formatting (short lines, line breaks, strategic emoji)\n"
            "- Strengthen the CTA\n"
            "- Optimize hashtags (3-5 highly relevant ones)\n"
            "Produce the final publish-ready version about {topic}, still "
            "within 150-300 words."
        ),
        expected_output=(
            "The final, polished, publish-ready LinkedIn post with optimized "
            "formatting, hashtags, and CTA, followed by a short bullet list of "
            "the changes made in response to the critique."
        ),
        agent=content_optimizer,
        context=[writing_task, critique_task],
    )

    # 5. Scheduling (receives the optimized post and the original research)
    scheduling_task = Task(
        name="Scheduling",
        description=(
            "Analyze the final post content and the target audience for "
            "{topic}. Recommend the best day and time to publish (with "
            "timezone) and explain why. Provide the final formatted post ready "
            "for LinkedIn copy-paste, and include a brief with the hashtag "
            "strategy and first-hour engagement tips."
        ),
        expected_output=(
            "A complete publishing brief in markdown with these sections:\n"
            "1. Recommended Posting Time (primary + backup slot, day, time, "
            "timezone, rationale)\n"
            "2. Final Post (exact copy-paste text, plain text, no markdown "
            "syntax inside the post itself)\n"
            "3. Hashtag List & Strategy\n"
            "4. First-Hour Engagement Strategy (actionable checklist)"
        ),
        agent=scheduling_agent,
        context=[research_task, optimization_task],
    )

    return [
        research_task,
        writing_task,
        critique_task,
        optimization_task,
        scheduling_task,
    ]


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

# Index of the stage currently running; reset at the start of every kickoff.
_stage_index = 0


def on_task_complete(output) -> None:
    """Print a separator after each pipeline stage.

    CrewAI calls `task_callback` with the TaskOutput of every finished task,
    in order, which lets us mark the transition between stages clearly. It is
    a module-level function (not a closure) so CrewAI can serialise the crew.
    """
    global _stage_index
    i = _stage_index
    stage = STAGES[i] if i < len(STAGES) else (output.name or "Task")
    print_banner(f"STAGE {i + 1}/{len(STAGES)} COMPLETE: {stage.upper()}")
    if i + 1 < len(STAGES):
        print_banner(f"STAGE {i + 2}/{len(STAGES)}: {STAGES[i + 1].upper()}", char="-")
    _stage_index += 1


def build_crew():
    """Assemble the LLM, agents, tasks, and the sequential crew."""
    from crewai import LLM, Crew, Process

    # One shared OpenAI LLM for every agent. The key is read from
    # OPENAI_API_KEY; the model defaults to gpt-4o (override via
    # OPENAI_MODEL_NAME in .env, e.g. "gpt-4" or "gpt-4o-mini").
    llm = LLM(
        model=OPENAI_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=os.environ["OPENAI_API_KEY"],
    )

    agents = build_agents(llm)
    tasks = build_tasks(agents)

    return Crew(
        name="linkedin-content-manager",
        agents=list(agents),
        tasks=tasks,
        process=Process.sequential,  # research -> write -> critique -> optimize -> schedule
        verbose=True,
        memory=True,  # uses OpenAI embeddings via OPENAI_API_KEY
        task_callback=on_task_complete,
    )


# ---------------------------------------------------------------------------
# Error classification & retries
# ---------------------------------------------------------------------------

def classify_error(exc: Exception) -> tuple[str, bool]:
    """Map an exception to (user-facing message, is_retryable).

    CrewAI may wrap provider errors, so we inspect the class name and message
    of the whole exception chain rather than relying on exact types.
    """
    chain = []
    current: BaseException | None = exc
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__

    text = " ".join(f"{type(e).__name__}: {e}" for e in chain).lower()

    if "authentication" in text or "invalid api key" in text or "incorrect api key" in text or "401" in text:
        return "Authentication failed. Check OPENAI_API_KEY / SERPER_API_KEY in .env.", False
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return "OpenAI quota exhausted. Check your plan and billing details.", False
    if "model_not_found" in text or ("does not exist" in text and "model" in text):
        return f"Model '{OPENAI_MODEL}' is not available to your OpenAI account. Set OPENAI_MODEL_NAME in .env.", False
    if "ratelimit" in text or "rate limit" in text or "429" in text:
        return "Rate limited by the API.", True
    if "timeout" in text or "timed out" in text:
        return "The API request timed out.", True
    if "connection" in text or "network" in text:
        return "Network error while contacting the API.", True
    if any(code in text for code in ("500", "502", "503", "504", "internalserver", "serviceunavailable")):
        return "The API returned a server error.", True
    return f"Unexpected error: {exc}", False


def run_with_retries(topic: str):
    """Run the crew, retrying transient API failures with exponential backoff."""
    global _stage_index
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _stage_index = 0
            crew = build_crew()
            print_banner(f"STAGE 1/{len(STAGES)}: {STAGES[0].upper()}", char="-")
            return crew.kickoff(inputs={"topic": topic})
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            message, retryable = classify_error(exc)
            if not retryable or attempt == MAX_RETRIES:
                raise RuntimeError(message) from exc
            delay = 2 ** attempt
            print_error(
                f"{message} Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})..."
            )
            time.sleep(delay)
    raise RuntimeError("Crew did not run")  # unreachable; satisfies type checkers


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(topic: str, result) -> Path:
    """Write every stage's output plus the final brief to a markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")[:50] or "post"
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}.md"

    parts = [f"# LinkedIn Content Package: {topic}\n",
             f"_Generated {datetime.now():%Y-%m-%d %H:%M} with {OPENAI_MODEL}_\n"]
    for stage, task_output in zip(STAGES, result.tasks_output):
        parts.append(f"\n---\n\n## {stage}\n\n{task_output.raw}\n")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def get_topic(cli_topic: str | None) -> str:
    """Return the topic from the CLI argument or prompt the user for it."""
    if cli_topic and cli_topic.strip():
        return cli_topic.strip()

    while True:
        try:
            topic = input("Enter the topic/niche for your LinkedIn post: ").strip()
        except EOFError:
            print_error("No topic provided.")
            sys.exit(1)
        if len(topic) >= 3:
            return topic
        print("Please enter a topic of at least 3 characters (e.g. 'AI in healthcare').")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Autonomous LinkedIn Content Manager (CrewAI, 5 agents)."
    )
    parser.add_argument("topic", nargs="*", help="Topic/niche (prompted if omitted)")
    args = parser.parse_args()

    print_banner("AUTONOMOUS LINKEDIN CONTENT MANAGER")
    print("Pipeline: " + "  ->  ".join(STAGES))
    print(f"Model:    {OPENAI_MODEL}\n")

    validate_environment()

    # Import CrewAI only after validation so a missing dependency is reported
    # clearly instead of as a raw traceback.
    try:
        import crewai  # noqa: F401
        import crewai_tools  # noqa: F401
    except ImportError as exc:
        print_error(f"Missing dependency ({exc.name}). Run: pip install -r requirements.txt")
        return 1

    topic = get_topic(" ".join(args.topic))
    print(f"\nTopic: {topic}")

    started = time.monotonic()
    try:
        result = run_with_retries(topic)
    except KeyboardInterrupt:
        print_error("Interrupted by user.")
        return 130
    except RuntimeError as exc:
        print_error(str(exc))
        return 1

    elapsed = time.monotonic() - started

    # Final, copy-paste-ready output from the Scheduling agent.
    print_banner("FINAL PUBLISHING BRIEF")
    print(result.raw)

    try:
        path = save_results(topic, result)
        print_banner("DONE")
        print(f"Full content package (all 5 stages) saved to: {path}")
    except OSError as exc:
        print_error(f"Could not save results to {OUTPUT_DIR}: {exc}")

    if result.token_usage:
        print(f"Total tokens used: {result.token_usage.total_tokens:,}")
    print(f"Completed in {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
