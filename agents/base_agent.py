"""
Base class for all AI agents.
Uses Claude Code CLI by default, falls back to Anthropic API.
"""
import re
import json
import os
import asyncio
import structlog

# Load env vars first
from config import settings  # noqa: F401

log = structlog.get_logger()

# Default models
ANALYST_MODEL = "claude-opus-4-6"
FAST_MODEL = "claude-haiku-4-5"

# Anthropic models (fallback when Claude Code CLI is unavailable)
ANTHROPIC_FAST_MODEL = "claude-haiku-4-5"
ANTHROPIC_ANALYST_MODEL = "claude-sonnet-4-6"


class BaseAgent:
    """
    Shared infrastructure for all trading agents.
    Uses Google Gemini API directly.
    """

    def __init__(self, name: str, model: str = ANALYST_MODEL):
        self.name = name
        self.model = model

    async def analyze(self, prompt: str, system: str | None = None, max_turns: int = 3) -> str:
        """
        Run a single analysis and return the text result.
        Tries Claude Code CLI first, falls back to Anthropic API.
        """
        # If running inside Claude Code already, skip CLI and go straight to Anthropic
        inside_claude_code = os.environ.get("CLAUDECODE") == "1"

        if not inside_claude_code:
            try:
                return await self._analyze_with_claude_code(prompt, system, max_turns)
            except Exception as e:
                log.warning("claude_code_failed_trying_anthropic", agent=self.name, error=str(e))

        # Fall back to Anthropic API
        try:
            return await self._analyze_with_anthropic(prompt, system)
        except Exception as e2:
            log.warning("anthropic_failed_using_heuristic", agent=self.name, error=str(e2))
            return self._heuristic_fallback(prompt, system)

    async def _analyze_with_claude_code(self, prompt: str, system: str | None = None, max_turns: int = 3) -> str:
        """Analyze using Claude Code CLI via subprocess to avoid conflicts."""
        import sys
        import subprocess
        import tempfile
        import concurrent.futures

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(full_prompt)
            prompt_file = f.name

        # Build clean env - strip keys that cause the SDK to use API instead of local CLI
        clean_env = {k: v for k, v in os.environ.items() if k not in (
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "CLAUDE_CODE_API_KEY_HELPER_TTY",
            "ANTHROPIC_API_KEY",  # Prevents SDK from routing to API instead of local CLI
        )}

        code = f"""
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

async def run():
    with open({repr(prompt_file)}) as f:
        prompt = f.read()
    result = ""
    async for msg in query(
        prompt=prompt,
        options=ClaudeAgentOptions(allowed_tools=[], model={repr(self.model)}, max_turns={max_turns}),
    ):
        if isinstance(msg, ResultMessage):
            result = msg.result
    print(result)

asyncio.run(run())
"""

        def run_subprocess():
            return subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=clean_env,
                timeout=120
            )

        try:
            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, run_subprocess)

            if result.returncode != 0:
                raise RuntimeError(f"Claude Code subprocess failed: {result.stderr[:200]}")

            output = result.stdout.strip()
            log.debug("agent_analyze_claude_code", agent=self.name, model=self.model, chars=len(output))
            return output
        finally:
            os.unlink(prompt_file)

    async def _analyze_with_anthropic(self, prompt: str, system: str | None = None) -> str:
        """Analyze using Anthropic API."""
        from anthropic import AsyncAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = AsyncAnthropic(api_key=api_key)

        model = ANTHROPIC_FAST_MODEL if self.model == FAST_MODEL else ANTHROPIC_ANALYST_MODEL

        messages = []
        if system:
            messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=messages,
        )

        # Handle different content block types
        content = response.content[0]
        if hasattr(content, 'text'):
            result = content.text
        elif hasattr(content, 'type') and content.type == 'text':
            result = content.text
        else:
            result = str(content)
        
        log.debug("agent_analyze_anthropic", agent=self.name, model=model, chars=len(result))
        return result

    def _heuristic_fallback(self, prompt: str, system: str | None = None) -> str:
        """Simple heuristic fallback when API quota is exceeded."""
        import re
        import random
        
        system = system or ""
        
        # Check if this is a probability analyst request
        if "true_prob_pct" in system or "calibrated probability" in system.lower():
            # Extract market price from prompt if possible
            market_price = 50
            match = re.search(r"Current YES price:\s*(\d+)", prompt)
            if match:
                market_price = float(match.group(1))
            
            # Return a neutral probability estimate
            return json.dumps({
                "true_prob_pct": market_price,
                "confidence": 0.3,
                "rationale": "API quota exceeded - using heuristic fallback. Recommend manual analysis.",
                "base_rate": "N/A - API quota exceeded",
                "key_risks": ["API quota exceeded - no AI analysis"],
                "information_needs": ["Retry when API quota available"],
                "recommended_side": None,
                "recommended_size_pct": 0.0
            })
        
        # Otherwise, scanner-style response
        tickers = re.findall(r'"ticker":\s*"([^"]+)"', prompt)
        if not tickers:
            return json.dumps([{"score": 7, "reason": "Heuristic fallback - API quota exceeded"}])
        
        return json.dumps([{"ticker": t, "score": 7, "reason": "Heuristic fallback"} for t in tickers])

    def _extract_json(self, text: str) -> dict:
        """Extract a JSON object from text that may contain markdown fences or prose."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)

    def _extract_json_array(self, text: str) -> list:
        """Extract a JSON array from text."""
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
