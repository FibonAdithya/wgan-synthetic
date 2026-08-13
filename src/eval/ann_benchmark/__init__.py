"""GPU ANN-algorithm benchmark: index build time, recall and QPS.

Deliberately exports nothing. Every cuVS import in this package is inside a
function body, and re-exporting from here would undo that by dragging the
device modules into any `import src.eval.ann_benchmark` on a CPU-only box.
"""
