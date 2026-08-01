"""Local AI Explainer — Ollama operator advisory for the digital twin.

Uses ``httpx`` against a local Ollama instance (default ``http://localhost:11434``)
with model ``qwen2.5:7b``. On timeout / offline, falls back to structured
template text so the control room never blocks on the LLM.

Safety
------
System prompt forbids direct hardware commands. Output is advisory only
(≤3 sentences).

Run
---
    python -m arip_backend.ai_explainer
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

import httpx

from arip_backend.schemas.explainer import OperatorAdvisory

DEFAULT_OLLAMA_BASE = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT_S = 1.5

SYSTEM_PROMPT = (
    "You are an industrial operator advisory tool for a nitroxylene hydrogenation "
    "batch reactor digital twin. Do NOT issue direct hardware control commands. "
    "Explain the process state, cause of warnings, and current reaction trajectory "
    "in 3 sentences or fewer. Be concise, factual, and use plant language "
    "(temperature, pressure, conversion, interlocks). Never invent setpoints or "
    "valve positions that were not provided in the inputs."
)


@dataclass
class LocalAIExplainer:
    """Async Ollama client that turns fused plant state into operator notes."""

    base_url: str = DEFAULT_OLLAMA_BASE
    model: str = DEFAULT_MODEL
    timeout_s: float = DEFAULT_TIMEOUT_S
    _last_error: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def build_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_user_prompt(
        self,
        *,
        stage: str,
        fused_state: Mapping[str, float],
        ekf_confidence: float,
        active_interlocks: Sequence[str],
        optimization_advice: Sequence[str],
        safety_status: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        """Pack numerical inputs into a tight operator briefing prompt."""
        T = fused_state.get("T_reactor", fused_state.get("T_reactor_c", fused_state.get("T")))
        P = fused_state.get("P_headspace", fused_state.get("P_headspace_bar", fused_state.get("P")))
        C_n = fused_state.get("C_nitro", fused_state.get("C_NX"))
        C_x = fused_state.get("C_xylidine", fused_state.get("C_amine", fused_state.get("C_XYL")))

        def _fmt(v: Any, unit: str = "", digits: int = 2) -> str:
            if v is None:
                return "n/a"
            try:
                return f"{float(v):.{digits}f}{unit}"
            except (TypeError, ValueError):
                return str(v)

        interlocks = ", ".join(active_interlocks) if active_interlocks else "none"
        advice = "; ".join(optimization_advice) if optimization_advice else "none"
        status = safety_status or ("CRITICAL" if active_interlocks else "NOMINAL")

        lines = [
            "Plant briefing — draft a ≤3 sentence operator note.",
            f"Current stage: {stage}",
            f"Safety status: {status}",
            f"Fused state: T={_fmt(T, '°C')}, P={_fmt(P, ' bar')}, "
            f"C_nitro={_fmt(C_n, ' kmol/m³', 3)}, C_xylidine={_fmt(C_x, ' kmol/m³', 3)}",
            f"EKF confidence: {float(ekf_confidence):.1f}%",
            f"Active safety interlocks: {interlocks}",
            f"Optimization advice: {advice}",
        ]
        if extras:
            for k, v in extras.items():
                lines.append(f"{k}: {v}")
        lines.append(
            "Respond with plain text only (no markdown bullets, no command verbs like OPEN/CLOSE/SET)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Template fallback (Ollama offline / timeout)
    # ------------------------------------------------------------------
    def template_advisory(
        self,
        *,
        stage: str,
        fused_state: Mapping[str, float],
        ekf_confidence: float,
        active_interlocks: Sequence[str],
        optimization_advice: Sequence[str],
        safety_status: str | None = None,
    ) -> str:
        """Deterministic ≤3 sentence note when the local LLM is unavailable."""
        T = fused_state.get("T_reactor", fused_state.get("T_reactor_c", fused_state.get("T")))
        P = fused_state.get("P_headspace", fused_state.get("P_headspace_bar", fused_state.get("P")))
        C_n = fused_state.get("C_nitro", fused_state.get("C_NX", 0.0))
        C_x = fused_state.get("C_xylidine", fused_state.get("C_amine", fused_state.get("C_XYL", 0.0)))
        status = safety_status or ("CRITICAL" if active_interlocks else "NOMINAL")

        s1 = (
            f"Stage {stage} is {status}: reactor at "
            f"{float(T) if T is not None else float('nan'):.1f}°C / "
            f"{float(P) if P is not None else float('nan'):.2f} bar with "
            f"C_nitro={float(C_n):.3f} and C_xylidine={float(C_x):.3f} kmol/m³ "
            f"(EKF confidence {float(ekf_confidence):.0f}%)."
        )

        if active_interlocks:
            s2 = (
                "Active interlocks ("
                + ", ".join(active_interlocks)
                + ") explain the elevated status; follow the DCS trip logic already in force—this note is advisory only."
            )
        else:
            s2 = (
                "No hard safety interlocks are active; the fused trajectory is consistent with "
                "continued monitored hydrogenation under existing TCU/MFC loops."
            )

        if optimization_advice:
            s3 = "Process note: " + optimization_advice[0]
        else:
            s3 = (
                "Reaction trajectory remains on the physics+EKF estimate; "
                "recheck confidence if sensor residuals grow."
            )

        return " ".join([s1, s2, s3])

    # ------------------------------------------------------------------
    # Ollama HTTP
    # ------------------------------------------------------------------
    def _chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/chat"

    def _generate_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/generate"

    async def _ollama_chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0.2,
                "num_predict": 120,
            },
        }
        timeout = httpx.Timeout(self.timeout_s, connect=self.timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self._chat_url(), json=payload)
            resp.raise_for_status()
            data = resp.json()
        # Ollama chat schema: {"message": {"content": "..."}}
        msg = data.get("message") or {}
        content = msg.get("content") or data.get("response") or ""
        text = str(content).strip()
        if not text:
            raise RuntimeError("Ollama returned empty advisory content")
        return text

    async def ping(self) -> bool:
        """Return True if Ollama responds within the connection timeout."""
        timeout = httpx.Timeout(self.timeout_s, connect=self.timeout_s)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(f"{self.base_url.rstrip('/')}/api/tags")
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate_operator_advisory(
        self,
        *,
        stage: str,
        fused_state: Mapping[str, float],
        ekf_confidence: float,
        active_interlocks: Sequence[str] | None = None,
        optimization_advice: Sequence[str] | None = None,
        safety_status: str | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> OperatorAdvisory:
        """Build prompt → call Ollama (1.5 s timeout) → fallback template if needed."""
        interlocks = list(active_interlocks or [])
        advice = list(optimization_advice or [])
        system = self.build_system_prompt()
        user = self.build_user_prompt(
            stage=stage,
            fused_state=fused_state,
            ekf_confidence=ekf_confidence,
            active_interlocks=interlocks,
            optimization_advice=advice,
            safety_status=safety_status,
            extras=extras,
        )

        timed_out = False
        reachable = False
        self._last_error = None

        try:
            text = await self._ollama_chat(system, user)
            reachable = True
            return OperatorAdvisory(
                advisory_text=self._clip_to_three_sentences(text),
                source="ollama",
                model_name=self.model,
                stage=stage,
                ekf_confidence=float(ekf_confidence),
                safety_status=safety_status,
                timed_out=False,
                ollama_reachable=True,
                metadata={"base_url": self.base_url},
            )
        except httpx.TimeoutException as exc:
            timed_out = True
            self._last_error = f"Ollama timeout ({self.timeout_s}s): {exc}"
        except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
            self._last_error = f"Ollama unavailable: {exc}"

        fallback = self.template_advisory(
            stage=stage,
            fused_state=fused_state,
            ekf_confidence=ekf_confidence,
            active_interlocks=interlocks,
            optimization_advice=advice,
            safety_status=safety_status,
        )
        return OperatorAdvisory(
            advisory_text=fallback,
            source="template_fallback",
            model_name=None,
            stage=stage,
            ekf_confidence=float(ekf_confidence),
            safety_status=safety_status,
            timed_out=timed_out,
            ollama_reachable=reachable,
            metadata={
                "base_url": self.base_url,
                "fallback_reason": self._last_error,
                "attempted_model": self.model,
            },
        )

    def generate_operator_advisory_sync(self, **kwargs: Any) -> OperatorAdvisory:
        """Sync wrapper for scripts / non-async callers."""
        return asyncio.run(self.generate_operator_advisory(**kwargs))

    @staticmethod
    def _clip_to_three_sentences(text: str) -> str:
        cleaned = " ".join(text.replace("\n", " ").split())
        parts: list[str] = []
        buf = ""
        for ch in cleaned:
            buf += ch
            if ch in ".!?" and len(buf.strip()) > 1:
                parts.append(buf.strip())
                buf = ""
                if len(parts) >= 3:
                    break
        if len(parts) < 3 and buf.strip():
            parts.append(buf.strip())
        return " ".join(parts[:3]) if parts else cleaned


def _demo() -> None:
    explainer = LocalAIExplainer(timeout_s=1.5)
    print("Local AI Explainer — Ollama Operator Advisory")
    print(f"Target: {explainer.model} @ {explainer.base_url}")
    print("=" * 60)

    cases = [
        dict(
            stage="HYDROGENATION / Stage 3",
            fused_state={
                "T_reactor_c": 88.5,
                "P_headspace_bar": 10.0,
                "C_nitro": 1.15,
                "C_xylidine": 1.55,
            },
            ekf_confidence=91.2,
            active_interlocks=[],
            optimization_advice=[
                "Increase Agitator Speed by +30 RPM to enhance k_L a.",
            ],
            safety_status="NOMINAL",
        ),
        dict(
            stage="HYDROGENATION / Stage 3",
            fused_state={
                "T_reactor_c": 108.2,
                "P_headspace_bar": 11.0,
                "C_nitro": 0.80,
                "C_xylidine": 1.90,
            },
            ekf_confidence=78.0,
            active_interlocks=["IL_T_REACTOR_HIGH", "ALARM_CRITICAL"],
            optimization_advice=[],
            safety_status="CRITICAL",
        ),
        dict(
            stage="Stage 3 → digest check",
            fused_state={
                "T_reactor_c": 85.0,
                "P_headspace_bar": 10.0,
                "C_nitro": 0.012,
                "C_xylidine": 2.75,
            },
            ekf_confidence=94.5,
            active_interlocks=[],
            optimization_advice=[
                "Transition batch to Stage 4 (Isothermal Digest).",
            ],
            safety_status="NOMINAL",
        ),
    ]

    async def _run() -> None:
        reachable = await explainer.ping()
        print(f"Ollama reachable: {reachable}\n")
        for i, case in enumerate(cases, 1):
            adv = await explainer.generate_operator_advisory(**case)
            print(f"[{i}] source={adv.source} timed_out={adv.timed_out}")
            print(f"    {adv.advisory_text}\n")

    asyncio.run(_run())


if __name__ == "__main__":
    _demo()
