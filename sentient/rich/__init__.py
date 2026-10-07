"""Sentient, current model: eight hidden factors, ten sensor channels, ten actions.

Same loop as the first prototype (sense, infer, recommend with a sealed forecast, score),
over a model with enough structure to tell loud-and-happy from loud-and-stressed, to know
when the air has gone stale, and to infer that the cooling unit has stopped working from
how the room responds to its own commands. The inference itself is scored against the
simulator's hidden truth. See docs/SENTIENT_PROTOTYPE.md.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
