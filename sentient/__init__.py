"""Sentient: a space that infers what the people in it need and recommends an adjustment,
within limits humans set.

This top-level package is the first two-factor prototype (kept for reference and still tested)
plus the pieces both models share: the sealed recommendation ledger, the baselines, and the
sensor-feed interface. The current model is in `sentient/rich/`.

Simulation-first. The same generative model that simulates the room infers its state.
A real sensor feed (Home Assistant) drops in behind the SensorFeed protocol with no model
change. See docs/SENTIENT_PROTOTYPE.md.
"""

import os

# Keep JAX on CPU and quiet for the prototype; the model is tiny.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
