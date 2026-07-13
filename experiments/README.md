# Experiment registry

`registry.csv` is an append-only local record of substantive runs. Each row
links a run ID to its configuration hash, Git commit, dataset checksum, seed,
hardware, status, runtime, and result location. Failed and invalidated runs are
retained with explanatory notes.
