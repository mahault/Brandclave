"""
VERSES Genius API Client for Active Inference.

This client interfaces with the VERSES Genius service for:
- Bayesian network inference
- POMDP-based action selection
- Structure learning with variational free energy minimization
- Belief updates and policy selection

The Genius API provides a proper active inference engine using
Factor Graphs (VFG) for probabilistic reasoning.
"""

import os
import time
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GeniusConfig:
    """Configuration for VERSES Genius API."""
    api_url: str = field(default_factory=lambda: os.getenv(
        "GENIUS_API_URL",
        "https://agent-3271175-2724605-8366d751a2c4dc.agents.genius.verses.ai"
    ))
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
    """

    def __init__(self, config: Optional[GeniusConfig] = None):
        self.config = config or GeniusConfig()

        if not self.config.api_key:
            raise ValueError("GENIUS_API_KEY environment variable not set")

        self.client = httpx.Client(
            base_url=self.config.api_url,
            timeout=self.config.timeout,
            headers={
                "x-api-key": self.config.api_key,
                "Content-Type": "application/json",
            }
        )

        self._graph_etag: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

    # -------------------------------------------------------------------------
    # License Management
    # -------------------------------------------------------------------------

    def activate_license(self) -> dict:
        """Activate the license key."""
        if not self.config.license_key:
            logger.warning("No license key configured")
            return {"status": "no_license"}

        try:
            response = self.client.post(
                "/license/update",
                json={"license_key": self.config.license_key}
            )
            response.raise_for_status()
            logger.info("Genius license activated successfully")
            return {"status": "activated"}
        except httpx.HTTPError as e:
            logger.error(f"License activation failed: {e}")
            return {"status": "error", "error": str(e)}

    def get_license_metadata(self) -> dict:
        """Get license metadata and limits."""
        try:
            response = self.client.get("/license/metadata")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get license metadata: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Health & Version
    # -------------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check if the Genius API is available."""
        try:
            response = self.client.get("/")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def get_version(self) -> dict:
        """Get API version information."""
        try:
            response = self.client.get("/version")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to get version: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Graph Management
    # -------------------------------------------------------------------------

    def get_graph(self) -> Optional[dict]:
        """Retrieve the current VFG graph."""
        try:
            response = self.client.get("/graph")
            if response.status_code == 404:
                return None
            response.raise_for_status()

            # Store ETag for conditional updates
            self._graph_etag = response.headers.get("etag")

            return response.json()
        except httpx.HTTPError as e:
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
        headers = {}
        if self._graph_etag:
            headers["if-match"] = self._graph_etag

        try:
            response = self.client.put(
                "/graph",
                json={"vfg": vfg},
                headers=headers
            )
            response.raise_for_status()

            self._graph_etag = response.headers.get("etag")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to set graph: {e}")
            raise

    def delete_graph(self) -> bool:
        """Delete the current graph."""
        try:
            response = self.client.delete("/graph")
            response.raise_for_status()
            self._graph_etag = None
            return True
        except httpx.HTTPError as e:
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
            response = self.client.post("/infer", json=request)

            if response.status_code == 202:
                # Async - need to poll
                task_id = response.json().get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Inference failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Action Selection (POMDP)
    # -------------------------------------------------------------------------

    def select_action(
        self,
        observations: Optional[dict[str, Any]] = None,
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

        try:
            response = self.client.post("/actionselection", json=request)

            if response.status_code == 202:
                task_id = response.json().get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
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
            response = self.client.post("/learn", json=request)

            if response.status_code == 202:
                task_id = response.json().get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
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
            response = self.client.post("/simulate", json=request)

            if response.status_code == 202:
                task_id = response.json().get("task_id")
                if wait and task_id:
                    return self._poll_task(task_id)
                return {"task_id": task_id, "status": "pending"}

            response.raise_for_status()
            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Simulation failed: {e}")
            raise

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_graph(self, vfg: dict) -> dict:
        """Validate a VFG structure without setting it."""
        try:
            response = self.client.post("/validate", json={"vfg": vfg})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
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
                response = self.client.get(f"/status/{task_id}")
                response.raise_for_status()

                result = response.json()
                status = result.get("status", "")

                if status == "completed":
                    return result.get("result", result)
                elif status == "failed":
                    error = result.get("error", "Unknown error")
                    raise RuntimeError(f"Task failed: {error}")

                # Still pending
                time.sleep(self.config.poll_interval)
                polls += 1

            except httpx.HTTPError as e:
                logger.error(f"Failed to poll task {task_id}: {e}")
                raise

        raise TimeoutError(f"Task {task_id} did not complete in time")

    def get_all_tasks(self) -> list[dict]:
        """Get status of all tasks."""
        try:
            response = self.client.get("/status")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
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
