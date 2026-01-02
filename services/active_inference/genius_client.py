"""
VERSES Genius API Client for Active Inference.

This client interfaces with the VERSES Genius service for:
- Bayesian network inference
- POMDP-based action selection
- Structure learning with variational free energy minimization
- Belief updates and policy selection

The Genius API provides a proper active inference engine using
Factor Graphs (VFG) for probabilistic reasoning.

NOTE: This client uses ONLY Python standard library (no httpx/requests).
Authentication uses x-api-key header (not Bearer token).
Graph writes (POST/PUT/DELETE) require If-Match: * header.
"""

import json
import os
import time
import logging
import urllib.request
import urllib.error
from typing import Optional, Any, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GeniusConfig:
    """
    Configuration for VERSES Genius API.

    All sensitive values (URL, API key) are read from environment variables.
    No defaults are provided for security - set these in .env file:
        GENIUS_API_URL=https://agent-xxx.agents.genius.verses.ai
        GENIUS_API_KEY=your-api-key
    """
    api_url: str = field(default_factory=lambda: os.getenv("GENIUS_API_URL", ""))
    api_key: str = field(default_factory=lambda: os.getenv("GENIUS_API_KEY", ""))
    agent_id: str = field(default_factory=lambda: os.getenv("GENIUS_AGENT_ID", ""))
    license_key: str = field(default_factory=lambda: os.getenv("GENIUS_LICENSE_KEY", ""))
    timeout: float = 60.0
    poll_interval: float = 1.0
    max_polls: int = 60


class GeniusClient:
    """
    Client for VERSES Genius Active Inference API.

    Supports:
    - Graph management (create/update VFG structures)
    - Inference (compute posteriors given evidence)
    - Action selection (POMDP-based policy selection)
    - Learning (update parameters from observations)
    - Simulation (multi-step rollouts)

    Uses only Python standard library (urllib) - no external dependencies.
    """

    def __init__(self, config: Optional[GeniusConfig] = None):
        self.config = config or GeniusConfig()

        if not self.config.api_url:
            raise ValueError("GENIUS_API_URL environment variable not set")
        if not self.config.api_key:
            raise ValueError("GENIUS_API_KEY environment variable not set")

        self._graph_etag: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # No cleanup needed for urllib

    def close(self):
        """Close the HTTP client (no-op for urllib)."""
        pass

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, str], Any]:
        """
        Make HTTP request using urllib (stdlib only).

        Returns: (status_code, response_headers, parsed_json_or_text_or_none)
        """
        url = f"{self.config.api_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
        }
        if extra_headers:
            headers.update(extra_headers)

        data = None
        if json_data is not None:
            data = json.dumps(json_data).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                status = resp.status
                resp_headers = dict(resp.headers.items())
                body_bytes = resp.read()
                if not body_bytes:
                    return status, resp_headers, None

                body_text = body_bytes.decode("utf-8")
                try:
                    return status, resp_headers, json.loads(body_text)
                except json.JSONDecodeError:
                    return status, resp_headers, body_text

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8") if hasattr(e, "read") else ""
            # Re-raise with context for caller to handle
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {err_body}") from None

    # -------------------------------------------------------------------------
    # License Management
    # -------------------------------------------------------------------------

    def activate_license(self) -> dict:
        """Activate the license key."""
        if not self.config.license_key:
            logger.warning("No license key configured")
            return {"status": "no_license"}

        try:
            status, _, body = self._request(
                "POST",
                "/license/update",
                json_data={"license_key": self.config.license_key}
            )
            logger.info("Genius license activated successfully")
            return {"status": "activated"}
        except RuntimeError as e:
            logger.error(f"License activation failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_license_metadata(self) -> dict:
        """Get license metadata and limits."""
        try:
            status, _, body = self._request("GET", "/license/metadata")
            return body if isinstance(body, dict) else {}
        except RuntimeError as e:
            logger.error(f"Failed to get license metadata: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Health & Version
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the Genius API is available."""
        try:
            status, _, _ = self._request("GET", "/")
            return status == 200
        except RuntimeError:
            return False

    def get_version(self) -> dict:
        """Get API version information."""
        try:
            status, _, body = self._request("GET", "/version")
            return body if isinstance(body, dict) else {}
        except RuntimeError as e:
            logger.error(f"Failed to get version: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Graph Management
    # -------------------------------------------------------------------------

    def get_graph(self) -> Optional[dict]:
        """Retrieve the current VFG graph."""
        try:
            status, headers, body = self._request("GET", "/graph")
            if status == 404:
                return None

            # Store ETag for conditional updates
            self._graph_etag = headers.get("ETag") or headers.get("etag")

            return body if isinstance(body, dict) else None
        except RuntimeError as e:
            if "404" in str(e):
                return None
            logger.error(f"Failed to get graph: {e}")
            return None

    def set_graph(self, vfg: dict) -> dict:
        """
        Set or update the VFG graph.

        Args:
            vfg: Variable Factor Graph structure with:
                - version: "0.4.0" or "0.5.0"
                - variables: Named discrete variables
                - factors: Distribution tables
                - metadata: Model type info

        Returns:
            Updated graph response
        """
        # Always include If-Match: * for graph writes (required by Genius API)
        extra_headers = {"If-Match": "*"}
        if self._graph_etag:
            extra_headers["If-Match"] = self._graph_etag

        try:
            status, headers, body = self._request(
                "POST",  # Use POST instead of PUT for loading graph
                "/graph",
                json_data={"vfg": vfg},
                extra_headers=extra_headers
            )

            self._graph_etag = headers.get("ETag") or headers.get("etag")
            return body if isinstance(body, dict) else {"status": "ok"}
        except RuntimeError as e:
            logger.error(f"Failed to set graph: {e}")
            raise

    def load_graph(self, vfg_model: dict) -> Tuple[int, Optional[str]]:
        """
        Load a VFG model (POST /graph with If-Match: *).

        This is the recommended way to load models per the new Genius API approach.

        Args:
            vfg_model: Complete VFG model dict with 'vfg' key

        Returns:
            Tuple of (status_code, etag)
        """
        try:
            status, headers, _ = self._request(
                "POST",
                "/graph",
                json_data=vfg_model,
                extra_headers={"If-Match": "*"}
            )
            etag = headers.get("ETag") or headers.get("etag")
            self._graph_etag = etag
            return status, etag
        except RuntimeError as e:
            logger.error(f"Failed to load graph: {e}")
            raise

    def unload_graph(self) -> int:
        """
        Unload the current graph (DELETE /graph with If-Match: *).

        Returns:
            HTTP status code
        """
        try:
            status, _, _ = self._request(
                "DELETE",
                "/graph",
                extra_headers={"If-Match": "*"}
            )
            self._graph_etag = None
            return status
        except RuntimeError as e:
            # It's ok if there was no graph to delete
            if "404" in str(e):
                self._graph_etag = None
                return 404
            logger.error(f"Failed to unload graph: {e}")
            raise

    def delete_graph(self) -> bool:
        """Delete the current graph (legacy method, use unload_graph)."""
        try:
            self.unload_graph()
            return True
        except RuntimeError as e:
            logger.error(f"Failed to delete graph: {e}")
            return False

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def infer(
        self,
        variables: list[str],
        evidence: Optional[dict[str, Any]] = None,
        wait: bool = True
    ) -> dict:
        """
        Run inference to compute posterior probabilities.

        Args:
            variables: Variables to query
            evidence: Observed variable values
            wait: If True, poll until complete

        Returns:
            Inference result with posteriors
        """
        request = {
            "variables": variables,
            "library": "pgmpy",
            "fail_on_warning": False,
        }

        if evidence:
            request["evidence"] = evidence

        try:
            status, _, body = self._request("POST", "/infer", json_data=request)

            if status == 202 and isinstance(body, dict):
                # Async - need to poll
                task_id = body.get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            return body if isinstance(body, dict) else {}

        except RuntimeError as e:
            logger.error(f"Inference failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Action Selection (POMDP)
    # -------------------------------------------------------------------------

    def select_action(
        self,
        observations: Optional[dict[str, Any]] = None,
        options: Optional[dict[str, Any]] = None,
        wait: bool = True
    ) -> dict:
        """
        Select optimal action using active inference (POMDP).

        This uses Expected Free Energy (EFE) minimization to select
        actions that balance:
        - Epistemic value (information gain, reducing uncertainty)
        - Pragmatic value (achieving preferred outcomes)

        Args:
            observations: Current observations as evidence
            options: Optional control parameters (policy_len, beta, etc.)
            wait: If True, poll until complete

        Returns:
            Action selection result with:
            - belief_state: Current beliefs
            - policy_belief: Probability over policies
            - efe_components: Breakdown of EFE
            - action: Selected action
        """
        request = {}
        if observations:
            request["observations"] = observations
        if options:
            request["options"] = options

        try:
            status, _, body = self._request("POST", "/actionselection", json_data=request)

            if status == 202 and isinstance(body, dict):
                task_id = body.get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            return body if isinstance(body, dict) else {}

        except RuntimeError as e:
            logger.error(f"Action selection failed: {e}")
            raise

    def action_selection(self, payload: dict) -> dict:
        """
        Raw action selection call (matches the new API approach).

        Args:
            payload: Full payload dict with observations and options

        Returns:
            Action selection response
        """
        try:
            status, _, body = self._request("POST", "/actionselection", json_data=payload)
            return body if isinstance(body, dict) else {}
        except RuntimeError as e:
            logger.error(f"Action selection failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------

    def learn(
        self,
        data: list[dict[str, Any]],
        wait: bool = True
    ) -> dict:
        """
        Update model parameters from observed data.

        Uses variational inference to update factor distributions
        based on observed data, minimizing free energy.

        Args:
            data: List of observation dictionaries
            wait: If True, poll until complete

        Returns:
            Learning result with:
            - updated_vfg: New graph with learned parameters
            - history: Learning metrics (JS divergence, log likelihood)
        """
        request = {"data": data}

        try:
            status, _, body = self._request("POST", "/learn", json_data=request)

            if status == 202 and isinstance(body, dict):
                task_id = body.get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            return body if isinstance(body, dict) else {}

        except RuntimeError as e:
            logger.error(f"Learning failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        steps: int = 10,
        initial_state: Optional[dict[str, Any]] = None,
        wait: bool = True
    ) -> dict:
        """
        Run multi-step simulation with the current model.

        Args:
            steps: Number of simulation steps
            initial_state: Starting state
            wait: If True, poll until complete

        Returns:
            Simulation trajectory
        """
        request = {"steps": steps}
        if initial_state:
            request["initial_state"] = initial_state

        try:
            status, _, body = self._request("POST", "/simulate", json_data=request)

            if status == 202 and isinstance(body, dict):
                task_id = body.get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            return body if isinstance(body, dict) else {}

        except RuntimeError as e:
            logger.error(f"Simulation failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_graph(self, vfg: dict) -> dict:
        """Validate a VFG structure without setting it."""
        try:
            status, _, body = self._request("POST", "/validate", json_data={"vfg": vfg})
            return body if isinstance(body, dict) else {}
        except RuntimeError as e:
            logger.error(f"Validation failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Task Polling
    # -------------------------------------------------------------------------

    def _poll_task(self, task_id: str) -> dict:
        """Poll a task until completion."""
        polls = 0

        while polls < self.config.max_polls:
            try:
                status, _, body = self._request("GET", f"/status/{task_id}")

                if isinstance(body, dict):
                    task_status = body.get("status", "")

                    if task_status == "completed":
                        return body.get("result", body)
                    elif task_status == "failed":
                        error = body.get("error", "Unknown error")
                        raise RuntimeError(f"Task failed: {error}")

                # Still pending
                time.sleep(self.config.poll_interval)
                polls += 1

            except RuntimeError as e:
                logger.error(f"Failed to poll task {task_id}: {e}")
                raise

        raise TimeoutError(f"Task {task_id} did not complete in time")

    def get_all_tasks(self) -> list[dict]:
        """Get status of all tasks."""
        try:
            status, _, body = self._request("GET", "/status")
            return body if isinstance(body, list) else []
        except RuntimeError as e:
            logger.error(f"Failed to get tasks: {e}")
            return []


# -----------------------------------------------------------------------------
# VFG Builder Helpers
# -----------------------------------------------------------------------------

class VFGBuilder:
    """
    Helper for building Variable Factor Graph (VFG) structures.

    VFG is the native format for Genius active inference models.
    """

    def __init__(self, version: str = "0.5.0"):
        self.vfg = {
            "version": version,
            "variables": {},
            "factors": {},
            "metadata": {
                "model_type": "bayesian_network"
            }
        }

    def add_variable(
        self,
        name: str,
        elements: list[str],
        description: str = ""
    ) -> "VFGBuilder":
        """
        Add a discrete variable to the graph.

        Args:
            name: Variable name
            elements: Possible values (states)
            description: Human-readable description
        """
        self.vfg["variables"][name] = {
            "type": "discrete",
            "elements": elements,
            "cardinality": len(elements),
        }
        if description:
            self.vfg["variables"][name]["description"] = description
        return self

    def add_categorical_factor(
        self,
        name: str,
        variable: str,
        probabilities: list[float]
    ) -> "VFGBuilder":
        """
        Add a categorical (prior) distribution.

        Args:
            name: Factor name
            variable: Variable this factor defines
            probabilities: Prior probabilities for each state
        """
        self.vfg["factors"][name] = {
            "type": "categorical",
            "variable": variable,
            "probabilities": probabilities
        }
        return self

    def add_conditional_factor(
        self,
        name: str,
        child: str,
        parents: list[str],
        cpt: list[list[float]]
    ) -> "VFGBuilder":
        """
        Add a conditional probability table (CPT).

        Args:
            name: Factor name
            child: Child variable
            parents: Parent variables
            cpt: Conditional probability table
        """
        self.vfg["factors"][name] = {
            "type": "conditional",
            "child": child,
            "parents": parents,
            "cpt": cpt
        }
        return self

    def set_model_type(self, model_type: str) -> "VFGBuilder":
        """
        Set the model type.

        Options: bayesian_network, pomdp, markov_random_field, factor_graph
        """
        self.vfg["metadata"]["model_type"] = model_type
        return self

    def build(self) -> dict:
        """Return the constructed VFG."""
        return self.vfg


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

def create_genius_client() -> GeniusClient:
    """Create a Genius client with default configuration."""
    return GeniusClient()


def test_genius_connection() -> dict:
    """Test connection to Genius API."""
    try:
        with GeniusClient() as client:
            healthy = client.health_check()
            version = client.get_version() if healthy else {}
            license_info = client.get_license_metadata() if healthy else {}

            return {
                "healthy": healthy,
                "version": version,
                "license": license_info
            }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test connection
    result = test_genius_connection()
    print(f"Genius API Status: {result}")

    if result.get("healthy"):
        # Test building a simple VFG
        vfg = (
            VFGBuilder()
            .add_variable("category", ["luxury", "budget", "boutique", "hostel"])
            .add_variable("traveler_type", ["business", "leisure", "digital_nomad", "backpacker"])
            .add_variable("satisfaction", ["low", "medium", "high"])
            .add_categorical_factor(
                "category_prior",
                "category",
                [0.25, 0.25, 0.25, 0.25]  # Uniform prior
            )
            .set_model_type("bayesian_network")
            .build()
        )

        print(f"\nSample VFG structure:")
        import json
        print(json.dumps(vfg, indent=2))
