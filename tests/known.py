"""Values pinned from earlier steps (see the results docs), shared by several test files."""

from __future__ import annotations

from uno import Level

# best responses to the step 8.6 Heuristic on TOY (the `symmetry` experiment, checked against uno.reference.solve)
HEURISTIC_TOY = {(0, Level.L1): 0.7378, (0, Level.L2): 0.7730, (0, Level.L3): 0.7798, (1, Level.L1): 0.4361, (1, Level.L2): 0.4774, (1, Level.L3): 0.4786}
