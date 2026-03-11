"""
Base class for all AI agents.
Supports multiple providers: 'claude' (default) and 'grok'.
Claude agents use Claude Code CLI -> Anthropic API -> Grok as fallback chain.
Grok agents go directly to xAI Grok API -> Anthropic -> heuristic as fallback.
Includes centralized cost tracking across all agents.
"""
import re
import json
import os
import asyncio
import time
from datetime import datetime
import structlog

# Load env vars first
from config import settings  # noqa: F401

log = structlog.get_logger()

# Claude models (for high-stakes reasoning: probability analysis, risk review)
ANALYST_MODEL = "claude-opus-4-6"
FAST_MODEL = "claude-haiku-4-5"

# Anthropic models (fallback when Claude Code CLI is unavailable)
ANTHROPIC_FAST_MODEL = "claude-haiku-4-5"
ANTHROPIC_ANALYST_MODEL = "claude-opus-4-6"

# xAI Grok models (for high-volume tasks: scanning, triage, monitoring)
GROK_FAST_MODEL = "grok-3-fast"
GROK_ANALYST_MODEL = "grok-4-1-fast-reasoning"

# Legacy alias
XAI_MODEL = GROK_ANALYST_MODEL

# ── Estimated cost per AI call by model (input + output, conservative) ──────
# These are rough per-call estimates assuming ~2000 input + ~500 output tokens.
# Used for budget tracking, not billing — actual costs come from provider dashboards.
_MODEL_COST_PER_CALL_USD: dict[str, float] = {
    # Claude
    ANALYST_MODEL: 0.068,              # Opus: $15/M in + $75/M out
    FAST_MODEL: 0.003,                 # Haiku: $0.80/M in + $4/M out
    ANTHROPIC_FAST_MODEL: 0.003,
    ANTHROPIC_ANALYST_MODEL: 0.068,
    # Grok
    GROK_FAST_MODEL: 0.010,            # grok-3-fast: ~$5/M in + $15/M out
    GROK_ANALYST_MODEL: 0.015,         # grok-4-1-fast-reasoning: ~$5/M in + $25/M out
}


class _CostTracker:
    """Singleton that tracks AI spend across all agents for the current day."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._daily_cost_usd: float = 0.0
        self._cost_date: str = ""  # YYYY-MM-DD
        self._per_agent: dict[str, float] = {}  # agent_name -> cumulative USD today

    async def record(self, agent_name: str, model: str) -> None:
        """Record one AI call's estimated cost."""
        cost = _MODEL_COST_PER_CALL_USD.get(model, 0.01)  # default 1¢ if unknown
        today = datetime.utcnow().strftime("%Y-%m-%d")
        async with self._lock:
            if today != self._cost_date:
                # New day — reset
                log.info("cost_tracker_daily_reset",
                         prev_date=self._cost_date,
                         prev_total=round(self._daily_cost_usd, 3),
                         per_agent=self._per_agent)
                self._daily_cost_usd = 0.0
                self._per_agent = {}
                self._cost_date = today
            self._daily_cost_usd += cost
            self._per_agent[agent_name] = self._per_agent.get(agent_name, 0.0) + cost

    @property
    def daily_total_usd(self) -> float:
        return self._daily_cost_usd

    @property
    def per_agent(self) -> dict[str, float]:
        return dict(self._per_agent)

    def check_limit(self, limit_usd: float) -> bool:
        """Return True if under limit, False if exceeded."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        if today != self._cost_date:
            return True  # new day, always under
        return self._daily_cost_usd < limit_usd


# Global singleton — shared across all agent instances
cost_tracker = _CostTracker()


class BaseAgent:
    """
    Shared infrastructure for all trading agents.
    Supports 'claude' and 'grok' as primary providers.
    Each provider has a fallback chain to ensure resilience.
    """

    def __init__(self, name: str, model: str = ANALYST_MODEL, provider: str = "claude"):
        self.name = name
        self.model = model
        self.provider = provider  # "claude" or "grok"

    async def analyze(self, prompt: str, system: str | None = None, max_turns: int = 3) -> str:
        """
        Run a single analysis and return the text result.
        Routes to the agent's configured provider first, then falls back.
        Automatically tracks estimated cost.
        """
        if self.provider == "grok":
            result = await self._analyze_grok_first(prompt, system)
        else:
            result = await self._analyze_claude_first(prompt, system, max_turns)
        # Track cost for this call
        await cost_tracker.record(self.name, self.model)
        return result

    async def _analyze_claude_first(self, prompt: str, system: str | None, max_turns: int) -> str:
        """Claude-first fallback chain: Claude Code CLI -> Anthropic API -> Grok -> heuristic."""
        inside_claude_code = os.environ.get("CLAUDECODE") == "1"

        if not inside_claude_code:
            try:
                return await self._analyze_with_claude_code(prompt, system, max_turns)
            except Exception as e:
                log.warning("claude_code_failed_trying_anthropic", agent=self.name, error=str(e))

        from config import settings
        if settings.ANTHROPIC_API_KEY:
            try:
                return await self._analyze_with_anthropic(prompt, system)
            except Exception as e2:
                log.warning("anthropic_failed_trying_grok", agent=self.name, error=str(e2))

        try:
            return await self._analyze_with_grok(prompt, system)
        except Exception as e3:
            log.warning("grok_failed_using_heuristic", agent=self.name, error=str(e3))
            return self._heuristic_fallback(prompt, system)

    async def _analyze_grok_first(self, prompt: str, system: str | None) -> str:
        """Grok-first fallback chain: Grok -> Anthropic API -> heuristic."""
        try:
            return await self._analyze_with_grok(prompt, system)
        except Exception as e:
            log.warning("grok_failed_trying_anthropic", agent=self.name, error=str(e))

        from config import settings
        if settings.ANTHROPIC_API_KEY:
            try:
                return await self._analyze_with_anthropic(prompt, system)
            except Exception as e2:
                log.warning("anthropic_fallback_failed_using_heuristic", agent=self.name, error=str(e2))

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
        """Analyze using Anthropic API. Uses prefilled assistant response for JSON mode."""
        from anthropic import AsyncAnthropic

        api_key = settings.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        client = AsyncAnthropic(api_key=api_key)

        model = ANTHROPIC_FAST_MODEL if self.model == FAST_MODEL else ANTHROPIC_ANALYST_MODEL

        messages = []
        if system:
            messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        # Prefill assistant response with "{" or "[" to force JSON output
        # This is Anthropic's recommended approach for structured output
        wants_json = self._wants_json(system)
        if wants_json:
            # Detect if the prompt expects an array or object
            if system and ("json array" in system.lower() or "return only a valid json array" in system.lower()):
                messages.append({"role": "assistant", "content": "["})
            else:
                messages.append({"role": "assistant", "content": "{"})

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

        # Prepend the prefilled character back since Anthropic continues from it
        if wants_json:
            if system and ("json array" in system.lower() or "return only a valid json array" in system.lower()):
                result = "[" + result
            else:
                result = "{" + result

        log.debug("agent_analyze_anthropic", agent=self.name, model=model, chars=len(result))
        return result

    @staticmethod
    def _wants_json(system: str | None) -> bool:
        """Check if the system prompt expects JSON output.
        Used to enable structured JSON mode on providers that support it."""
        if not system:
            return False
        s = system.lower()
        return ("return only" in s and "json" in s) or "return only valid json" in s or "output format (valid json" in s

    def _resolve_grok_model(self) -> str:
        """Pick the right Grok model based on the agent's configured model tier."""
        if self.model in (FAST_MODEL, GROK_FAST_MODEL):
            return GROK_FAST_MODEL
        return GROK_ANALYST_MODEL

    async def _analyze_with_grok(self, prompt: str, system: str | None = None) -> str:
        """Analyze using xAI Grok API (OpenAI-compatible). Enables JSON mode when appropriate."""
        import httpx

        from config.settings import XAI_API_KEY

        if not XAI_API_KEY:
            raise ValueError("XAI_API_KEY not set")

        grok_model = self._resolve_grok_model()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        request_body: dict = {
            "model": grok_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": 1024,
        }
        # Enable JSON mode when the system prompt requests JSON output
        if self._wants_json(system):
            request_body["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]

        log.debug("agent_analyze_grok", agent=self.name, model=grok_model, chars=len(result))
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
        # Try markdown code block first
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find balanced JSON object
        try:
            # Find first { and try progressively longer matches
            start = text.find("{")
            if start == -1:
                return json.loads(text)
            
            # Try each possible end position from shortest to longest
            for end in range(start + 1, len(text) + 1):
                try:
                    result = json.loads(text[start:end])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    continue
            
            # Fallback: try the whole text
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: return empty dict
            return {}

    def _extract_json_array(self, text: str) -> list:
        """Extract a JSON array from text."""
        # Try markdown code block first
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        
        # Try to find balanced JSON array
        try:
            start = text.find("[")
            if start == -1:
                return json.loads(text)
            
            for end in range(start + 1, len(text) + 1):
                try:
                    result = json.loads(text[start:end])
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue
            
            return json.loads(text)
        except json.JSONDecodeError:
            return []
