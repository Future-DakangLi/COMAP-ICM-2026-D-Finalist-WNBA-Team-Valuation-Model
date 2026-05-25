# Honest Reproduction Note

This version intentionally removes all direct result targets from the executable workflow.

Removed design patterns:

- no file containing reported paper outputs is loaded by the code;
- no hard-coded paper values such as published P10/P50/P90 simulation outputs;
- no scaling predicted attendance to a known paper total;
- no scaling ticket revenue to a known paper revenue;
- no quantile-matching simulation trick.

The trade-off is that some generated numbers will naturally differ from the PDF. That is acceptable for this repository because the priority is a clean and believable modelling process.
