"""Re-implementation of the ViDDAR evaluation pipeline (Xiu et al., IEEE TVCG 2025).

Covers the dataset-based evaluation of obstruction and information manipulation
attack detection (Sections 5.1 and 5.2 of the paper) with locally hosted VLMs.
"""

import os

# Never let transformers try to import TensorFlow/Flax that may exist elsewhere on the machine.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Use the operating system's certificate store for HTTPS (model downloads). Needed on machines
# where antivirus/proxy software re-signs TLS traffic with a certificate unknown to `certifi`.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass
