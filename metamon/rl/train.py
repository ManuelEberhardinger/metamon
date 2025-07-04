import os
from functools import partial

import wandb

import amago
from amago import cli_utils

from metamon.env import get_metamon_teams
from metamon.interface import (
    TokenizedObservationSpace,
    ALL_OBSERVATION_SPACES,
    ALL_REWARD_FUNCTIONS,
    ALL_ACTION_SPACES,
)
from metamon.tokenizer import get_tokenizer
from metamon.data import ParsedReplayDataset
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGODataset,
    make_baseline_env,
    make_placeholder_env,
)
from metamon import baselines


WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")


def add_cli(parser):
    # fmt: off
    parser.add_argument("--run_name", required=True, help="Give the run a name to identify logs and checkpoints.")
    parser.add_argument("--obs_space", type=str, default="DefaultObservationSpace")
    parser.add_argument("--reward_function", type=str, default="DefaultShapedReward")
    parser.add_argument("--action_space", type=str, default="DefaultActionSpace")
    parser.add_argument("--parsed_replay_dir", type=str, default=None, help="Path to the parsed replay directory. Defaults to the official huggingface version.")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to save checkpoints. Find checkpoints under {ckpt_dir}/{run_name}/ckpts/")
    parser.add_argument("--ckpt", type=int, default=None, help="Resume training from an existing run with this run_name. Provide the epoch checkpoint to load.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for. In offline RL model, an epoch is an arbitrary interval (here: 25k) of training steps on a fixed dataset.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=12, help="Batch size per GPU. Total batch size is batch_size_per_gpu * num_gpus.")
    parser.add_argument("--grad_accum", type=int, default=1, help="Number of gradient accumulations per update.")
    parser.add_argument("--il", action="store_true", help="Overrides amago settings to use imitation learning.")
    parser.add_argument("--model_gin_config", type=str, required=True, help="Path to a gin config file (that might edit the model architecture). See provided rl/configs/models/)")
    parser.add_argument("--train_gin_config", type=str, required=True, help="Path to a gin config file (that might edit the training or hparams).")
    parser.add_argument("--tokenizer", type=str, default="DefaultObservationSpace-v0", help="The tokenizer to use for the text observation space. See metamon.tokenizer for options.")
    parser.add_argument("--dloader_workers", type=int, default=10, help="Number of workers for the data loader.")
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
    # fmt: on
    return parser


live_opponents = [
    baselines.heuristic.basic.PokeEnvHeuristic,
    baselines.heuristic.basic.Gen1BossAI,
    baselines.heuristic.basic.Grunt,
    baselines.heuristic.basic.GymLeader,
    baselines.heuristic.kaizo.EmeraldKaizo,
]


def configure(args):
    """
    Setup gin configuration with overrides for command line args or anything else
    """
    config = {
        "MetamonTstepEncoder.tokenizer": get_tokenizer(args.tokenizer),
        "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
    }
    if args.il:
        # NOTE: would break for a custom agent, but ultimately just creates some wasted params that aren't trained
        config.update(
            {
                "amago.agent.Agent.use_multigamma": False,
                "amago.agent.MultiTaskAgent.use_multigamma": False,
                "amago.agent.Agent.fake_filter": True,
                "amago.agent.MultiTaskAgent.fake_filter": True,
            }
        )
    cli_utils.use_config(config, [args.model_gin_config, args.train_gin_config])


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_cli(parser)
    args = parser.parse_args()
    configure(args)

    # metamon dataset
    obs_space = TokenizedObservationSpace(
        ALL_OBSERVATION_SPACES[args.obs_space](), get_tokenizer(args.tokenizer)
    )
    reward_function = ALL_REWARD_FUNCTIONS[args.reward_function]()
    action_space = ALL_ACTION_SPACES[args.action_space]()
    parsed_replay_dataset = ParsedReplayDataset(
        dset_root=args.parsed_replay_dir,
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        # amago will handle sequence lengths
        max_seq_len=None,
        verbose=True,
        # FIXME
        formats=["gen1ou"],
    )
    amago_dataset = MetamonAMAGODataset(
        dset_name="Metamon Parsed Replays",
        parsed_replay_dset=parsed_replay_dataset,
    )

    # validation environments (evaluated throughout training)
    make_envs = [
        partial(
            make_baseline_env,
            battle_format=f"gen{i}ou",
            observation_space=obs_space,
            action_space=action_space,
            reward_function=reward_function,
            player_team_set=get_metamon_teams(f"gen{i}ou", "competitive"),
            opponent_type=opponent,
        )
        for i in range(
            1, 4
        )  # TODO: gen4ou teams are illegal after updating to latest Showdown due to tiering change.
        for opponent in live_opponents
    ]
    experiment = MetamonAMAGOExperiment(
        ## required ##
        run_name=args.run_name,
        ckpt_base_dir=args.ckpt_dir,
        # max_seq_len = should be set in the gin file
        dataset=amago_dataset,
        # tstep_encoder_type = should be set in the gin file
        # traj_encoder_type = should be set in the gin file
        # agent_type = should be set in the gin file
        val_timesteps_per_epoch=300,  # per actor
        ## environment ##
        make_train_env=partial(make_placeholder_env, obs_space, action_space),
        make_val_env=make_envs,
        env_mode="async",
        async_env_mp_context="spawn",
        parallel_actors=len(make_envs),
        # no exploration
        exploration_wrapper_type=None,
        sample_actions=True,
        force_reset_train_envs_every=None,
        ## logging ##
        log_to_wandb=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
        verbose=True,
        log_interval=300,
        ## replay ##
        padded_sampling="none",
        dloader_workers=args.dloader_workers,
        ## learning schedule ##
        epochs=args.epochs,
        # entirely offline RL
        start_learning_at_epoch=0,
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        train_batches_per_epoch=25_000 * args.grad_accum,
        val_interval=1,
        ckpt_interval=2,
        ## optimization ##
        batch_size=args.batch_size_per_gpu,
        batches_per_update=args.grad_accum,
        mixed_precision="no",
    )

    experiment.start()
    if args.ckpt is not None:
        experiment.load_checkpoint(args.ckpt)
    experiment.learn()
    wandb.finish()
