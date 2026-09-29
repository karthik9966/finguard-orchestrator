"""Evaluation: the golden datasets, the metric runners, and the baseline they are compared against.

Kept out of `src/` because it is not part of the service. Nothing under `src/` imports anything here,
which is the property that stops an evaluation fixture from quietly becoming production behaviour.
"""
