"""
Things to do when making a new model:
change database path
change event_no csv path
make sure you are importing and using the right loss function
truth = TRUTH.ICECUBE86 + ['is_data'] if variable is not already in TRUTH.ICECUBE86
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pytorch_lightning.callbacks import EarlyStopping
# Optional experiment tracking; see ThesisWorkspace/README.md.
# from pytorch_lightning.loggers import WandbLogger
from  pytorch_lightning import Trainer

import torch
import numpy as np
import pandas as pd
from torch.optim.adam import Adam

#from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector import IceCube86, IceCubeDeepCore
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses

from graphnet.models.task.reconstruction import ZenithReconstructionWithKappa, ZenithReconstruction
from graphnet.training.callbacks import ProgressBar, PiecewiseLinearLR
from graphnet.training.loss_functions import VonMisesFisher2DLoss
from graphnet.training.utils import make_dataloader
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger


print('All imported')
logger = Logger()

# Constants
features = FEATURES.DEEPCORE

truth = TRUTH.DEEPCORE[:-1]
#truth = ["energy",
 #       "energy_track",
  #      "energy_cascade",
   #     "position_x",
    #    "position_y",
     #   "position_z",
      #  "azimuth",
       # "zenith",]

# Outputs stay beside this script unless --output-dir is supplied.
OUTPUT_DIR = str(Path(__file__).resolve().parent / "outputs")


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    event_no_path: str,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Train using local data and save artifacts in the output directory."""
    # Load event selections only when training is explicitly requested.
    event_nos = pd.read_csv(event_no_path)["event_no"].tolist()
    n_train = int(len(event_nos) * 0.8)
    train_events = [int(event) for event in event_nos[:n_train]]
    val_events = [int(event) for event in event_nos[n_train:]]
    # External tracking is disabled for the public example. To opt in,
    # uncomment the import and block below, then set Trainer(logger=wandb_logger).
    # wandb_dir = os.path.join(output_dir, "wandb")
    # os.makedirs(wandb_dir, exist_ok=True)
    # wandb_logger = WandbLogger(
    #     project="thesis-workspace",
    #     name=Path(__file__).stem,
    #     save_dir=wandb_dir,
    #     log_model=False,
    # )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")


    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            #"devices": 2,
            "max_epochs": max_epochs,
        },
    }

    archive = os.path.join(output_dir, "Zenith_Reco")
    run_name = "{}_Neutrinos".format(config["target"])


 # Define graph representation
    graph_definition = KNNGraph(detector = IceCubeDeepCore(),
                                nb_nearest_neighbours = 8,
                                node_definition=NodesAsPulses(),   # nearest neighbors and node definition was added
                                )


    training_dataloader = make_dataloader(
        db=config["path"],
        graph_definition=graph_definition,
        pulsemaps=config["pulsemap"],
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        shuffle=True,
        selection=train_events,
        num_workers=config["num_workers"],
        truth_table=truth_table,
    )

    validation_dataloader = make_dataloader(
        db=config["path"],
        pulsemaps=config["pulsemap"],
        graph_definition=graph_definition,
        features=features,
        truth=truth,
        batch_size=config["batch_size"],
        shuffle=False,
        selection=val_events,
        num_workers=config["num_workers"],
        truth_table=truth_table,
    )

    # Building model
    gnn = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = ZenithReconstructionWithKappa(
        hidden_size=gnn.nb_outputs,
        target_labels=config["target"],
        loss_function=VonMisesFisher2DLoss(),
        )

    model = StandardModel(
        graph_definition=graph_definition,
        gnn=gnn,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(training_dataloader) / 2,
                len(training_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-2, 1, 1e-02],
        },
        scheduler_config={
            "interval": "step",
        },
    )

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=config["early_stopping_patience"],
        ),
        ProgressBar(),
    ]

    # Configure Trainer ########## CHECK THIS ###########
    trainer = Trainer(
        max_epochs=config["fit"]["max_epochs"],
        callbacks=callbacks,
        log_every_n_steps=50,
        devices=gpus if gpus else 1,
        logger=False,
        accelerator="gpu" if gpus else "cpu",

        )

    # Train model
    trainer.fit(
        model,
        training_dataloader,
        validation_dataloader,
    )

    # Get predictions
    #prediction_columns = config["target"] + "_pred"
    prediction_columns  = ["zenith_pred", "kappa_pred"]
    additional_attributes = [config["target"]]

    additional_attributes = model.target_labels
    assert isinstance(additional_attributes, list)  # mypy

    results = model.predict_as_dataframe(
        validation_dataloader,
        prediction_columns=prediction_columns,
        additional_attributes=additional_attributes + ["event_no"],
    )

    # Save predictions and model to file
    result_path = os.path.join(archive, run_name)

    # Log information
    logger.info(f"Writing results to {result_path}")

    # Ensure the directory exists
    os.makedirs(result_path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{result_path}/results.csv")

    # Save model state dictionary
    model.save_state_dict(f"{result_path}/state_dict.pth")

    # Save model configuration as a YAL file
    model.save_config(f"{result_path}/model_config.yml")


if __name__ == "__main__":

    torch.multiprocessing.set_sharing_strategy("file_system")
    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
        Train GNN model without the use of config files.
        """
    )

    parser.add_argument(
        "--path",
        help="Path to your local SQLite training database.",
        required=True,
    )

    parser.add_argument(
        "--events",
        required=True,
        help="CSV file with an event_no column.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory for generated artifacts (default: %(default)s)",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="SplitInIcePulses",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default=["zenith"],
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 50),
        ("early-stopping-patience", 3),
        ("batch-size", 512),
        ("num-workers", 25)
    )


    args = parser.parse_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.events,
        args.output_dir,
    )
