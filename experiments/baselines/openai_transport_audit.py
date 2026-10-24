"""Count actual HTTP attempts without retaining request content or credentials."""

from __future__ import annotations

from typing import Any


_AUDIT_INDEX = "campus_development_audit_index"


class OpenAITransportAudit:
    """Attach synchronous hooks to the HTTP client passed into OpenAI."""

    def __init__(self) -> None:
        self._attempts: list[dict[str, int | None]] = []

    @property
    def event_hooks(self) -> dict[str, list]:
        return {"request": [self.request_hook], "response": [self.response_hook]}

    def request_hook(self, request: Any) -> None:
        request.extensions[_AUDIT_INDEX] = len(self._attempts)
        self._attempts.append({"http_status": None})

    def response_hook(self, response: Any) -> None:
        index = response.request.extensions.get(_AUDIT_INDEX)
        if isinstance(index, int) and 0 <= index < len(self._attempts):
            self._attempts[index]["http_status"] = int(response.status_code)

    def summary(self, *, logical_requests: int) -> dict[str, Any]:
        attempts = [dict(record) for record in self._attempts]
        complete = (logical_requests > 0 and len(attempts) >= logical_requests
                    and all(isinstance(item["http_status"], int) for item in attempts))
        return {
            "provider_http_attempt_count": len(attempts),
            "provider_http_attempts": attempts,
            "provider_transport_complete": complete,
        }
