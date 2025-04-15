import argparse
from rgiant.preprocessing.cleaning import run_cleaning_pipeline
from rgiant.preprocessing.connectomes import run_connectome_pipeline
from rgiant.preprocessing.graphs import build_pyg_data
from rgiant.preprocessing.nodes import extract_node_features

def parse_args():
    parser = argparse.ArgumentParser(description="Run R-GIANT pipelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Cleaning pipeline
    cleaning_parser = subparsers.add_parser("clean", help="Run the cleaning pipeline.")
    cleaning_parser.add_argument("--participant-id", required=True, help="Participant ID.")
    cleaning_parser.add_argument("--session-id", required=True, help="Session ID.")
    cleaning_parser.add_argument("--data-dir", required=True, help="Directory containing input data.")
    cleaning_parser.add_argument("--clear-temp", action="store_true", help="Clear temporary files after processing.")
    cleaning_parser.add_argument("--log-dir", default="logs/default", help="Directory for log files.")
    cleaning_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Connectomes pipeline
    connectomes_parser = subparsers.add_parser("connectome", help="Run the connectomes pipeline.")
    connectomes_parser.add_argument("--participant-id", required=True, help="Participant ID.")
    connectomes_parser.add_argument("--session-id", required=True, help="Session ID.")
    connectomes_parser.add_argument("--data-dir", required=True, help="Directory containing input data.")
    connectomes_parser.add_argument("--log-dir", default="logs/default", help="Directory for log files.")
    connectomes_parser.add_argument("--plot-dir", default="plots/matrices", help="Directory for saving plots.")
    connectomes_parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Graphs pipeline
    graphs_parser = subparsers.add_parser("graph", help="Build PyG data.")
    graphs_parser.add_argument("--participant-id", required=True, help="Participant ID.")
    graphs_parser.add_argument("--session-id", required=True, help="Session ID.")
    graphs_parser.add_argument("--data-dir", required=True, help="Directory containing input data.")

    # Nodes pipeline
    nodes_parser = subparsers.add_parser("nodes", help="Extract node features.")
    nodes_parser.add_argument("--participant-id", required=True, help="Participant ID.")
    nodes_parser.add_argument("--session-id", required=True, help="Session ID.")
    nodes_parser.add_argument("--data-dir", required=True, help="Directory containing input data.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "clean":
        # Calling the cleaning pipeline.
        # Note: cleaning pipeline does not use plot_dir.
        run_cleaning_pipeline(
            patient_id=args.participant_id,
            session_id=args.session_id,
            data_dir=args.data_dir,
            clear_temp=args.clear_temp,
            log_dir=args.log_dir, 
            stream=args.verbose,
        )
    elif args.command == "connectome":
        # Connectomes pipeline uses an additional argument for plot directory.
        run_connectome_pipeline(
            patient_id=args.participant_id,
            session_id=args.session_id,
            data_dir=args.data_dir,
            log_dir=args.log_dir,           # e.g., defaults to "logs/default" unless user sets it or via SLURM job script overrides
            plot_dir=args.plot_dir,         # Only connectomes receives this argument
            stream=args.verbose,
        )
    elif args.command == "graph":
        build_pyg_data(
            patient_id=args.participant_id,
            session_id=args.session_id,
            data_dir=args.data_dir,
        )
    elif args.command == "nodes":
        extract_node_features(
            patient_id=args.participant_id,
            session_id=args.session_id,
            data_dir=args.data_dir,
        )

if __name__ == "__main__":
    main()
