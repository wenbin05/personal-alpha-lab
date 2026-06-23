from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.extractions.models import LLMExtraction
from src.extractions.prompting import (
    OPENAI_EXTRACTION_SCHEMA,
    OPENAI_PROMPT_VERSION,
    build_openai_extraction_input,
    verify_evidence_snippets,
)
from src.extractions.quality import apply_quality_calibration
from src.extractions.validation import extraction_from_payload, normalize_extraction_payload, safe_json_list


OPENAI_PROVIDER_KEY = "openai_compatible"


class OpenAIProviderError(RuntimeError):
    """Sanitized provider error safe to display in the UI."""


@dataclass
class OpenAIProviderConfig:
    provider: str = ""
    model: str = ""
    max_input_chars: int = 12_000
    timeout_seconds: int = 60
    api_key: str | None = field(default=None, repr=False)

    @property
    def provider_requested(self) -> bool:
        return self.provider.strip().lower() in {"openai", "openai_compatible"}

    @property
    def enabled(self) -> bool:
        return self.provider_requested and bool(self.api_key) and bool(self.model.strip())

    @property
    def disabled_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.provider_requested:
            reasons.append("Set LLM_PROVIDER=openai to enable the OpenAI provider.")
        if not self.api_key:
            reasons.append("OPENAI_API_KEY is not configured.")
        if not self.model.strip():
            reasons.append("LLM_MODEL is not configured.")
        return reasons


@dataclass(frozen=True)
class OpenAIProviderStatus:
    enabled: bool
    model: str
    warnings: list[str]


def build_openai_config(settings: Any | None = None, environ: dict[str, str] | None = None) -> OpenAIProviderConfig:
    env = environ if environ is not None else os.environ
    llm_settings = getattr(settings, "llm", None)
    provider = str(getattr(llm_settings, "provider", "") or env.get("LLM_PROVIDER", "") or "").strip()
    model = str(getattr(llm_settings, "model", "") or env.get("LLM_MODEL", "") or "").strip()
    max_chars = getattr(llm_settings, "max_input_chars", env.get("LLM_MAX_INPUT_CHARS", 12_000))
    timeout = getattr(llm_settings, "timeout_seconds", env.get("LLM_TIMEOUT_SECONDS", 60))

    try:
        max_input_chars = max(500, int(max_chars))
    except Exception:
        max_input_chars = 12_000
    try:
        timeout_seconds = max(5, int(timeout))
    except Exception:
        timeout_seconds = 60

    return OpenAIProviderConfig(
        provider=provider,
        model=model,
        max_input_chars=max_input_chars,
        timeout_seconds=timeout_seconds,
        api_key=env.get("OPENAI_API_KEY") or None,
    )


def openai_provider_status(settings: Any | None = None, environ: dict[str, str] | None = None) -> OpenAIProviderStatus:
    config = build_openai_config(settings, environ=environ)
    return OpenAIProviderStatus(
        enabled=config.enabled,
        model=config.model,
        warnings=[] if config.enabled else config.disabled_reasons,
    )


def redact_secret(text: str, api_key: str | None = None) -> str:
    redacted = str(text or "")
    if api_key:
        redacted = redacted.replace(api_key, "[REDACTED]")
    redacted = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", redacted)
    return redacted


def sanitize_openai_error(exc: Exception, api_key: str | None = None) -> str:
    class_name = type(exc).__name__
    message = redact_secret(str(exc), api_key=api_key)
    if class_name in {"AuthenticationError"}:
        return "OpenAI authentication failed. Check OPENAI_API_KEY."
    if class_name in {"PermissionDeniedError", "NotFoundError"}:
        return f"OpenAI model access failed. Check LLM_MODEL and account permissions. ({message})"
    if class_name in {"APITimeoutError", "TimeoutError"}:
        return "OpenAI request timed out. Try a smaller document or a longer LLM_TIMEOUT_SECONDS value."
    if class_name in {"RateLimitError"}:
        return "OpenAI rate limit or quota was reached. No extraction was stored."
    if class_name in {"APIConnectionError", "ConnectionError"}:
        return "OpenAI connection failed. Check network access and try again."
    return f"OpenAI extraction failed: {message}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe({k: v for k, v in vars(value).items() if not k.startswith("_")})
    return str(value)


def _extract_response_text(response: Any) -> str:
    output_text = str(getattr(response, "output_text", "") or "").strip()
    if output_text:
        return output_text

    output = getattr(response, "output", None) or []
    text_parts: list[str] = []
    refusals: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if isinstance(item, dict):
            content = item.get("content")
        for part in content or []:
            if isinstance(part, dict):
                if part.get("type") == "refusal" or part.get("refusal"):
                    refusals.append(str(part.get("refusal") or "Model refused the request."))
                if part.get("text"):
                    text_parts.append(str(part["text"]))
            else:
                if getattr(part, "type", "") == "refusal" or getattr(part, "refusal", None):
                    refusals.append(str(getattr(part, "refusal", None) or "Model refused the request."))
                text = getattr(part, "text", None)
                if text:
                    text_parts.append(str(text))
    if refusals:
        raise OpenAIProviderError("OpenAI returned a refusal. No extraction was stored.")
    return "\n".join(text_parts).strip()


def _response_metadata(response: Any) -> dict[str, Any]:
    return {
        "response_id": getattr(response, "id", None),
        "usage": _json_safe(getattr(response, "usage", None)),
    }


def _create_default_client(api_key: str, timeout_seconds: int) -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise OpenAIProviderError("The openai package is not installed. Run pip install -r requirements.txt.") from exc
    return OpenAI(api_key=api_key, timeout=timeout_seconds)


class OpenAIExtractionProvider:
    def __init__(
        self,
        config: OpenAIProviderConfig,
        client_factory: Callable[[str, int], Any] | None = None,
    ) -> None:
        self.config = config
        self.client_factory = client_factory or _create_default_client

    @classmethod
    def from_settings(
        cls,
        settings: Any | None = None,
        environ: dict[str, str] | None = None,
        client_factory: Callable[[str, int], Any] | None = None,
    ) -> "OpenAIExtractionProvider":
        return cls(build_openai_config(settings, environ=environ), client_factory=client_factory)

    def status(self) -> OpenAIProviderStatus:
        return OpenAIProviderStatus(
            enabled=self.config.enabled,
            model=self.config.model,
            warnings=[] if self.config.enabled else self.config.disabled_reasons,
        )

    def extract(self, document: dict[str, Any], extraction_type: str = "general_document_review") -> LLMExtraction:
        if not self.config.enabled:
            raise OpenAIProviderError("OpenAI provider is disabled. " + " ".join(self.config.disabled_reasons))

        prepared = build_openai_extraction_input(
            document,
            extraction_type=extraction_type,
            max_input_chars=self.config.max_input_chars,
        )
        try:
            client = self.client_factory(str(self.config.api_key), self.config.timeout_seconds)
            response = client.responses.create(
                model=self.config.model,
                input=[
                    {"role": "system", "content": prepared.system_prompt},
                    {"role": "user", "content": prepared.user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "personal_alpha_lab_llm_extraction",
                        "schema": OPENAI_EXTRACTION_SCHEMA,
                        "strict": True,
                    }
                },
                store=False,
            )
        except OpenAIProviderError:
            raise
        except Exception as exc:
            raise OpenAIProviderError(sanitize_openai_error(exc, api_key=self.config.api_key)) from exc

        output_text = _extract_response_text(response)
        if not output_text:
            raise OpenAIProviderError("OpenAI returned an empty response. No extraction was stored.")
        try:
            provider_payload = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise OpenAIProviderError("OpenAI returned invalid structured JSON. No extraction was stored.") from exc

        provider_payload, evidence_warnings = verify_evidence_snippets(provider_payload, prepared.submitted_text)
        provider_payload, quality_warnings = apply_quality_calibration(provider_payload, prepared.submitted_text)
        model_warnings = safe_json_list(provider_payload.get("extraction_warnings"))
        all_warnings = [*model_warnings, *prepared.warnings, *evidence_warnings, *quality_warnings]
        provider_payload["extraction_warnings"] = " | ".join(dict.fromkeys(w for w in all_warnings if w))

        normalized_output = normalize_extraction_payload(
            {
                **provider_payload,
                "document_id": document.get("document_id") or 0,
                "catalyst_id": document.get("catalyst_id"),
                "ticker": document.get("ticker") or "UNKNOWN",
                "provider": OPENAI_PROVIDER_KEY,
                "model_name": self.config.model,
                "extraction_type": extraction_type,
                "review_status": "pending_review",
                "prompt_version": OPENAI_PROMPT_VERSION,
            }
        )
        raw_metadata = {
            "provider": "openai",
            "model_name": self.config.model,
            "prompt_version": OPENAI_PROMPT_VERSION,
            "original_chars": prepared.original_chars,
            "submitted_chars": prepared.submitted_chars,
            "truncated": prepared.truncated,
            "validated_output": {
                key: normalized_output[key]
                for key in [
                    "event_type_detected",
                    "sentiment_label",
                    "catalyst_strength",
                    "risk_severity",
                    "confidence",
                    "document_relevance",
                    "evidence_sufficiency",
                    "time_horizon",
                    "key_positive_points",
                    "key_risks",
                    "evidence_snippets",
                    "short_summary",
                    "detailed_summary",
                    "proposed_score_effect",
                    "extraction_warnings",
                ]
            },
            **_response_metadata(response),
        }
        normalized_output["raw_llm_response_json"] = json.dumps(_json_safe(raw_metadata), ensure_ascii=False)
        normalized_output["review_status"] = "pending_review"
        return extraction_from_payload(normalized_output)
