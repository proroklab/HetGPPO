from pathlib import Path
from typing import Union

import torch

from utils import EvaluationUtils


def export(
    checkpoint_path: Union[str, Path],
):

    config, trainer, env = EvaluationUtils.get_config_trainer_and_env_from_checkpoint(
        checkpoint_path
    )

    model_path = (
        Path(checkpoint_path).parent
        / "a_range_1_u_range_0_5_[3_2_0_002]_0_05_dt_0_1_friction_0_dt_delay_option_5.pt"
    )

    model = trainer.get_policy().model.gnn

    torch.save(model, model_path)


if __name__ == "__main__":
    checkpoint_path = "/Users/Matteo/Downloads/MultiPPOTrainer_give_way_deploy_7ec2a_00000_0_2022-10-18_11-13-22/checkpoint_000267/checkpoint-267"
    export(checkpoint_path)
