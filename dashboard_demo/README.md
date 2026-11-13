# Improved dashboard demo

This separate Streamlit app reads the existing saved artifacts. It does not modify
the original dashboard, experiment engine, configuration, or results. It reuses
the existing manifest and result validators without invoking the original app.

From the repository root, run:

```bash
cd dashboard_demo
../.venv/bin/streamlit run app.py --server.port 8502
```

Open http://localhost:8502. Starting from this directory applies the demo's isolated
theme. The demo uses the repository's existing dependencies; no install is needed
if its virtual environment is already set up.

To use a different saved results directory, append
`-- --results /absolute/path/to/results` to the command. With no saved batches,
the demo displays an empty state and never generates results automatically.

## Suggested improvements demonstrated

- Put the leading set, score, score margin and consistency ratio above the fold.
- Use named alternatives and criterion labels in ranking and applied-weight charts.
- Preserve ties and explain that TOPSIS scores are relative, not probabilities.
- Separate passed, failed, deferred and blocked checks, with status filtering.
- Make consistency issues visible while retaining the original judgments.
- Plot saved sensitivity points and expose saved winner-change intervals.
- Show detailed calculation stages on demand, rendering only the selected view.
- Distinguish selected-run downloads from whole-batch exports.
- Use native accessible widgets, chart descriptions and an isolated teal theme.

The data is synthetic and the checks validate the independent reference engine.
PoliInsight comparison and AI interpretation evaluation remain deferred.

The demo is a design concept, not a replacement for every original dashboard view.
Resolved inputs and complete saved traces remain available in expanders/JSON.
