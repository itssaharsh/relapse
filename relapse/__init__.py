"""Relapse: does unlearned knowledge stay gone when a model is stored at lower precision?

Pipeline (Lexsi Labs open-source stack on Cohere Tiny Aya):
CuratorKIT builds the probe sets -> CircuitKIT locates where the knowledge flows ->
SafeTune removes it (RMU / NPO, plus our quantization-aware RMU) -> AuditKIT scores
every storage format with paired confidence intervals.
"""
__version__ = "0.1.0"
