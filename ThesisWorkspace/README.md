# Thesis workspace

Research training scripts and saved GraphNeT model configurations. Training
databases, event selections, credentials, model weights, and event-level results
are not distributed with this workspace.

## Setup and paths

Install GraphNeT and its training dependencies using the repository's
[getting-started guide](../GETTING_STARTED.md). These scripts use PyTorch,
PyTorch Lightning, NumPy, and pandas. Use a compatible GraphNeT environment;
the historical experiments have not been validated against newer dependencies.

Run a script from the repository root, supplying your own database and event
selection. For example:

```bash
python ThesisWorkspace/training_models/MuonAnalysis/RegressionModels/Position_Reco/position.py \
  --path /path/to/training.db \
  --events /path/to/event_numbers.csv \
  --output-dir ThesisWorkspace/outputs/position
```

- `--path` is required and points to a SQLite database with the pulsemap,
  truth table, features, and target labels expected by the selected script.
- `--events` is required. Regression scripts expect a CSV with an `event_no`
  column. The two classifier scripts expect a text file of integer event
  numbers, one per line. The first 80% train the model; the remainder validate it.
- `--output-dir` defaults to an ignored `outputs/` folder beside the script.
  Predictions, weights, and generated model configurations are written there.
- Training defaults to CPU. Add `--gpus 0` (or other device indices) to use GPUs.
- Use `--help` to inspect the remaining options. Event selections are loaded
  inside `main`, after argument parsing.

## Optional experiment tracking

Weights & Biases imports, initialization, and uploads are commented out, and
the Lightning trainer uses `logger=False`. To enable tracking, install W&B,
uncomment the indicated import and initialization block, choose your project,
and change the trainer argument to `logger=wandb_logger`. Authenticate outside
the repository. Model artifact uploads remain disabled with `log_model=False`.

## Existing research-script limitations

Model definitions were preserved during publication cleanup. Directory and
script names do not always match the implemented task:

| Script | Implemented task |
| --- | --- |
| `RegressionModels/Position_Reco/position.py` | Position reconstruction |
| `RegressionModels/Energy_Reco/energy.py` | Position reconstruction (duplicate template) |
| `RegressionModels/Zenith_Reco/zenith.py` | Zenith reconstruction using DeepCore features |
| `RegressionModels/Azimuth_Reco/azimuth.py` | Zenith reconstruction (duplicate template) |
| `StoppedThroughClassifier/StoppedThrough.py` | Stopped-muon classification |
| `RealVsSim/RealVsSim.py` | Stopped-muon classification (duplicate template) |

Paths in this table are relative to `training_models/MuonAnalysis/`. The angle
scripts were configured for neutrino selections despite their directory name.
Review the task, loss, target labels, and detector before training; changing
`--target` alone does not change the reconstruction task or loss.

The checked-in YAML files describe historical model architectures. Some differ
from the neighboring scripts and are not automatically loaded by them.
`training_models/NeutrinoAnalysis/` contains saved configurations, without
training scripts or model weights.

## Publication exclusions

The workspace `.gitignore` excludes local tokens, environment files, datasets,
CSV results, weights, checkpoints, and logs. Existing results and credentials
can remain on disk without being included in a commit. Keep event-selection
text files under the ignored `ThesisWorkspace/data/` directory if you store
them in this workspace.

The publication cleanup was checked for Python syntax, argument wiring, and
remaining private paths. Full training requires the external data and training
dependencies and was not run as part of this cleanup.
