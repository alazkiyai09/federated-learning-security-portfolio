#!/bin/bash
# Run Experiments Script for Flower Fraud Detection

set -e

echo "=========================================="
echo "Flower Fraud Detection - Run Experiments"
echo "=========================================="
echo ""

# Function to run experiment
run_experiment() {
    local name=$1
    local strategy=$2
    local data=${3:-"iid"}
    local extra_args=${@:4}

    echo ""
    echo "------------------------------------------"
    echo "Running: $name"
    echo "Strategy: $strategy"
    echo "Data: $data"
    echo "------------------------------------------"
    python main.py strategy=$strategy data=$data $extra_args
}

# Parse command line arguments
EXPERIMENT=${1:-"all"}

case $EXPERIMENT in
    "fedavg-iid")
        run_experiment "FedAvg on IID" "fedavg" "iid"
        ;;
    "fedavg-noniid")
        run_experiment "FedAvg on Non-IID" "fedavg" "non_iid"
        ;;
    "fedprox-noniid")
        run_experiment "FedProx on Non-IID" "fedprox" "non_iid"
        ;;
    "fedadam-noniid")
        run_experiment "FedAdam on Non-IID" "fedadam" "non_iid"
        ;;
    "all")
        echo "Running full experiment suite..."
        run_experiment "FedAvg on IID" "fedavg" "iid"
        run_experiment "FedAvg on Non-IID" "fedavg" "non_iid"
        run_experiment "FedProx on Non-IID" "fedprox" "non_iid"
        run_experiment "FedAdam on Non-IID" "fedadam" "non_iid"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo ""
        echo "Available experiments:"
        echo "  fedavg-iid       - FedAvg on IID data"
        echo "  fedavg-noniid    - FedAvg on Non-IID data"
        echo "  fedprox-noniid   - FedProx on Non-IID data"
        echo "  fedadam-noniid   - FedAdam on Non-IID data"
        echo "  all              - Run all experiments"
        echo ""
        echo "Usage: ./run_experiments.sh [experiment]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "View results with:"
echo "  tensorboard --logdir logs"
echo ""
