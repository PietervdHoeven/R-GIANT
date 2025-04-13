# src/cli.py

import argparse
from batch_scripts import batch_clean, batch_connectomes, batch_nodes

def main():
    parser = argparse.ArgumentParser(description="R-GIANT Batch CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add subcommands
    subparsers.add_parser("clean", help="Run batch_clean.py")
    subparsers.add_parser("connectomes", help="Run batch_connectomes.py")
    subparsers.add_parser("nodes", help="Run batch_nodes.py")

    args = parser.parse_args()

    # Route to the correct batch script
    if args.command == "clean":
        batch_clean.main()
    elif args.command == "connectomes":
        batch_connectomes.main()
    # elif args.command == "nodes":
    #     batch_nodes.main()
