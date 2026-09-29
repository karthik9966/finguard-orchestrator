"""Metric runners, split the way Evaluation Design §2 splits the tiers.

`deterministic` needs no model and no network: detector recall against the baseline, context
precision (local embeddings and a local cross-encoder), the clean-batch guarantee, malformed-input
handling and schema conformance. It is what the PR gate runs, and it is free.

`live` needs a model: system-level recall, the faithfulness gate, triage ranking, prompt-injection
resistance and the advisory narrative judge. It is what the nightly gate runs, and it costs money.
"""
