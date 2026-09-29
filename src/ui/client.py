"""The cockpit's view of the engine: one HTTP client, no imports from the graph.

Before Phase 6c the UI imported the orchestrator and ran audits **inside the Streamlit process**.
That was convenient and wrong in three ways worth naming, because each is a thing that would have
been discovered in production rather than here:

* **Two execution paths.** An audit run from the UI and one run through the API went through
  different code, so anything true of one was only probably true of the other.
* **No serialisation.** Two analysts pressing the button at once meant two concurrent audits
  contending for one vector store and one rate limit. The API's single worker is the answer, and a
  UI that bypasses it does not get that answer.
* **A Streamlit rerun could bill money.** Streamlit re-executes the whole script on every widget
  interaction. Guarding against that with a session-state key works until someone adds a widget
  above the guard.

So the cockpit is now a client of the same API a bank's own system would call, and this module is
the whole of its access to the engine -- which is also what makes it testable: every test drives it
against the real FastAPI app over an ASGI transport, with no network and no server.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import get_settings
from src.models import ComplianceReport, ValidationReport

# Generous, because the thing on the other end is an audit. A dev batch is under a minute and the
# 10,000-message batch is not; `submit(wait=True)` has the API's own bounded wait behind it, so this
# only has to outlast that rather than guess at a run length.
DEFAULT_TIMEOUT = 360.0


class ApiError(RuntimeError):
    """A non-2xx answer, carrying the status so the UI can say something specific.

    A 401 means the token is wrong, a 503 means the service or its corpus is not ready, and a 409
    means the review action conflicts with where the finding already is. Those are three different
    things for an analyst to do, so they must not arrive as one stack trace.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class FinGuardClient:
    """Everything the cockpit can ask of the engine."""

    base_url: str | None = None
    token: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    # An already-open client to use instead of building one. The suite passes FastAPI's own
    # TestClient here, so every test below drives the *real* endpoints -- routing, auth, the
    # lifespan, the single worker -- with no server and no network. Unused in production.
    session: httpx.Client | None = None

    def __post_init__(self) -> None:
        settings = get_settings()
        self.base_url = (self.base_url or settings.api_base_url).rstrip("/")
        self.token = self.token if self.token is not None else settings.api_auth_token

    # --- plumbing -------------------------------------------------------------------------

    @contextmanager
    def _session(self):
        """An injected client is borrowed, never closed -- its owner opened it and will close it."""
        if self.session is not None:
            yield self.session
            return
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            yield client

    def _request(self, method: str, path: str, **kwargs) -> Any:
        # Per request rather than on the client, so an injected session needs no configuring and
        # cannot be left holding a stale token.
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            with self._session() as client:
                response = client.request(method, path, headers=headers, **kwargs)
        except httpx.HTTPError as error:
            # A connection refused is the common case in development -- the API is simply not
            # running -- and it deserves that sentence rather than a transport traceback.
            raise ApiError(0, f"cannot reach the API at {self.base_url}: {error}") from error

        if response.is_success:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, str(detail))

    # --- reading --------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def audit(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/audits/{job_id}")

    def audits(self, *, limit: int = 50) -> list[dict[str, Any]]:
        return self._request("GET", "/audits", params={"limit": limit})

    def reports(self, *, period: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if period:
            params["period"] = period
        return self._request("GET", "/reports", params=params)

    def report(self, report_id: str) -> ComplianceReport:
        """The frozen report joined to its findings' current review statuses."""
        return ComplianceReport(**self._request("GET", f"/reports/{report_id}"))

    def filed(self, report_id: str) -> ComplianceReport:
        """The same report exactly as the engine produced it, with no review applied."""
        return ComplianceReport(**self._request("GET", f"/reports/{report_id}/filed"))

    def validation(self, report_id: str) -> ValidationReport | None:
        """The ingestion record, or None when this report predates it being kept."""
        try:
            return ValidationReport(**self._request("GET", f"/reports/{report_id}/validation"))
        except ApiError as error:
            if error.status == 404:
                return None
            raise

    # --- writing --------------------------------------------------------------------------

    def submit(
        self, filename: str, payload: bytes, *, wait: bool = False, force: bool = False
    ) -> dict[str, Any]:
        """Hand a batch to the API. Returns the 202 body, or the finished result when `wait`."""
        params: dict[str, Any] = {}
        if wait:
            params["wait"] = "true"
        if force:
            params["force"] = "true"
        return self._request(
            "POST", "/audits", files={"batch": (filename, payload)}, params=params or None
        )

    def review(
        self, finding_id: str, action: str, *, reviewer: str, note: str = ""
    ) -> dict[str, Any]:
        return self._request(
            "POST", f"/findings/{finding_id}/review",
            json={"action": action, "reviewer": reviewer, "note": note},
        )
