import argparse
import sys

from scripts.run_experiment import run_experiment
from scripts.evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Run SMACv2 experiments (train/test) with TorchRL-based DQN, VDN or QMIX."
    )

    subparsers = parser.add_subparsers(dest = "command", required = True)

    train_parser = subparsers.add_parser("train", help = "Run training.")
    train_parser.add_argument(
        "--alg",
        choices = ["dqn", "vdn", "qmix"],
        default = "dqn",
        help = "Which algorithm to run: DQN, VDN or QMIX."
    )

    test_parser = subparsers.add_parser("test", help = "Run evaluation.")
    test_parser.add_argument(
        "--alg",
        choices = ["dqn", "vdn", "qmix"],
        default = "dqn",
        help = "Which algorithm to test: DQN, VDN or QMIX"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    config_path = f"configs/{args.alg}.yaml"
    checkpoint_path = f"checkpoints/{args.alg}_checkpoint.pth"

    if args.command == "train":
        print(f"Running training of {args.alg} algorithm...")
        run_experiment(args.alg, config_path)
    elif args.command == "test":
        print(f"Running evaluation of {args.alg} algorithm...")
        evaluate(args.alg, config_path, checkpoint_path)

if __name__ == "__main__":
    main()
