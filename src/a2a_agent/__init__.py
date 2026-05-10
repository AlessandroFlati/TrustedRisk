"""TrustedRisk A2A agent -- hybrid LlmAgent + SequentialAgent composer.

Per design doc §3.4 Option C (hybrid): the root agent interprets the request
and routes to decide_then_validate (Sequential) or validate-only, but NEVER
skips validate when a clinical recommendation is being produced.
"""

from .agent import root_agent, to_a2a_app

__all__ = ["root_agent", "to_a2a_app"]
