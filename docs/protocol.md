# Worker protocol, version 1

Worker: `python -m e57cls run --config <run.json>`. Standard output carries UTF-8
JSON Lines. Non-JSON output from third-party libraries is treated as technical
log text. Standard error is captured separately. English technical logs and all
protocol events are retained under the run output directory.

Every event has `version: 1`, `event`, UTC timestamp and `elapsed_seconds`.
Events: `state`, `phase_started`, `phase_finished`, `progress`, `scene`, `warning`,
`error`, `finished`. Progress includes phase and fraction, and scan/image/view
counters when available. `finished` carries output and report paths. Errors remain
visible in the GUI; worker exit does not close its window.

Commands on standard input, one JSON object per line:

```json
{"version":1,"command":"pause"}
{"version":1,"command":"resume"}
{"version":1,"command":"cancel"}
```

Pause and cancel are cooperative at bounded chunk/tile/view boundaries. States:
`running`, `paused`, `cancelling`, `cancelled`. Cancel exits with status 130, errors
with nonzero status. Resume in the same process clears pause. Resume after exit
restarts the saved configuration: completed mask entries are hash-verified and
per-scan votes are committed in a SQLite transaction. The GUI keeps the original
configuration for resume; edited settings create a fresh run directory.

Closing a GUI-owned active worker requests cancellation, waits for its safe exit,
and then closes the window. `--watch <run folder>` is a read-only event follower
for a worker started separately; closing that window does not affect the worker.

The owner removed the inspection confirmation gate and output-folder selector.
File selection immediately starts the worker; checks only produce reports/logs.
Output defaults to `klassifiziert/run_YYYYMMDD_HHMMSS` next to the input file.
