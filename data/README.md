# Dataset Setup

The raw dataset is intentionally excluded from the repository because of its size.

Place the toxic-comment training dataset in:

```text
data/
└── raw/
    └── train.csv
```

The project uses a labelled Wikipedia toxic-comment dataset containing more than 159,000 comments.

If the notebook's dataset-loading cell points to a different local filename/path, update that cell to reference `data/raw/train.csv` before running the notebook from the repository root.
