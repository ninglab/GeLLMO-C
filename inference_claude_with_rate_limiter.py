import os
import time
import random
import asyncio
import json
import argparse

from anthropic import Anthropic, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from prompter import Prompter
from config import DEFAULT_MAX_TOKENS

# ── Setup Anthropic client ─────────────────────────────────────────────────────
API_KEY = os.environ("ANTHROPIC_API_KEY")
client = Anthropic(api_key=API_KEY)

# ── Token-bucket rate limiter ───────────────────────────────────────────────────
class TokenBucket:
    def __init__(self, calls_per_minute: int):
        self.capacity = calls_per_minute
        self.tokens = calls_per_minute
        self.refill_rate = calls_per_minute / 60.0  # tokens per second
        self.timestamp = time.monotonic()

    async def wait(self):
        now = time.monotonic()
        elapsed = now - self.timestamp
        # refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.timestamp = now
        if self.tokens < 1:
            # sleep just enough to accrue 1 token
            await asyncio.sleep((1 - self.tokens) / self.refill_rate)
            self.tokens = 0
        self.tokens -= 1

# ── API-call with retry and rate-limit handling ────────────────────────────────
@retry(wait=wait_exponential(min=4, max=30), stop=stop_after_attempt(5))
async def get_completion_with_retry(
    prompt: str,
    system_prompt: str,
    bucket: TokenBucket,
) -> dict:
    """
    Calls Claude via Anthropic client, honoring token-bucket rate limits
    and retrying on 429s using the Retry-After header.
    """
    # wait for our next token
    await bucket.wait()

    # run the blocking client call in a thread
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=DEFAULT_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return {"input": prompt, "output": response.content[0].text}

    except RateLimitError as e:
        retry_after = float(e.headers.get("retry-after", 1.0))
        await asyncio.sleep(retry_after)
        # re-raise so tenacity will retry
        raise

# ── Orchestrate concurrent processing ──────────────────────────────────────────
async def process_prompts(
    prompts: list[str],
    system_prompt: str,
    calls_per_minute: int = 50,
    concurrency: int = 5
) -> list[dict]:
    """
    Schedules all prompts under a shared TokenBucket and Semaphore
    to respect both rate limits and bounded concurrency.
    """
    bucket = TokenBucket(calls_per_minute)
    sem = asyncio.Semaphore(concurrency)
    tasks = []
    for prompt in prompts:
        async def worker(p=prompt):
            async with sem:
                try:
                    return await get_completion_with_retry(p, system_prompt, bucket)
                except Exception as e:
                    # you can log e here if desired
                    return None
        tasks.append(asyncio.create_task(worker()))

    results = await asyncio.gather(*tasks)
    # filter out failures
    return [r for r in results if r is not None]

# ── Entry point ────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path", required=True)
    parser.add_argument("--icl_data_path",  required=True)
    parser.add_argument("--task_name",      required=True)
    parser.add_argument("--opt_type",       required=True)
    parser.add_argument("--num_shots",      type=int, required=True)
    parser.add_argument("--output_path",    required=True)
    args = parser.parse_args()

    # generate prompts
    prompter = Prompter(opt_type=args.opt_type)
    prompts = prompter.generate_prompt_for_general_purpose_LLMs(
        args.test_data_path,
        args.icl_data_path,
        task=args.task_name,
        prompt_type="icl",
        model_id="claude",
        sampling="random",
        num_shots=args.num_shots,
        prompt_explain=False
    )
    system_prompt = prompter.system_prompt

    # call API
    responses = await process_prompts(
        prompts,
        system_prompt,
        calls_per_minute=50,   # tier-1 limit for Claude 3.5 Sonnet
        concurrency=5          # adjust based on your comfort with parallelism
    )

    # save
    out_dir = os.path.join(args.output_path, "claude-3.5", f"{args.opt_type}-{args.num_shots}", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.task_name}_response.json")
    with open(out_file, "w") as f:
        json.dump(responses, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
