import numpy as np
import logging
from typing import List, Dict, Tuple

# Import your aggregator class here: aggregator = ...

def simulate_exploration_aggregator(aggregator):
    """Demonstration of aggregator usage in a single round."""

    # create "previous_global"
    previous_global = [
        np.random.randn(64, 784),  # fc1.weight
        np.random.randn(64),       # fc1.bias
        np.random.randn(32, 64),   # fc2.weight
        np.random.randn(32),       # fc2.bias
        np.random.randn(10, 32),   # fc3.weight
        np.random.randn(10),       # fc3.bias
    ]

    # build some client updates, 7 benign + 3 malicious
    client_updates = {}
    for client_idx in range(10):
        if client_idx < 7:
            cid = f"client_{client_idx}"
            # add small noise
            param_list = []
            for layer in previous_global:
                # add small noise for perturbations between benign clients
                noise = 0.01 * np.random.randn(*layer.shape)
                param_list.append(layer + noise)
            metadata = {
                "num_examples": np.random.randint(5, 100),
                "client_id": cid
            }
            client_updates[cid] = (param_list, metadata)
        else:
            # malicious
            cid = "malicious_client" + str(client_idx)
            param_list = []
            for layer in previous_global:
                # large noise to simulate malicious updates
                noise = 10 * np.random.randn(*layer.shape)
                param_list.append(layer + noise)
            metadata = {
                "num_examples": np.random.randint(5, 100),
                "client_id": cid
            }
            client_updates[cid] = (param_list, metadata)

    # aggregator does the actual model update
    new_global = aggregator.aggregate(previous_global, client_updates)

    print("Used clients this round:")
    print(aggregator.used_clients_per_round[aggregator.round])

    print("=== Aggregation Complete ===")
    for i, layer in enumerate(new_global):
        print(f" Layer {i} shape: {layer.shape}")
        if np.isnan(layer).any():
            print("   --> NaNs found!")
        else:
            print(f"   --> OK, mean={layer.mean():.4f}, std={layer.std():.4f}")

    return new_global


# import your aggregator class here
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Import you aggregator class HERE! 
    # aggregator = # ExploreExploitTrimmedMean()

    global_params = None
    for round_num in range(25):
        print(f"\n=== Simulation Round {round_num + 1} ===")

        # Example usage: we gather some clients, let aggregator decide who to select
        all_clients = [f"client_{i}" for i in range(3)] + ["malicious_client"]
        selected = aggregator.client_selection(all_clients)
        print(f"Selected clients for round {round_num+1}: {selected}")
        aggregator.client_settings(global_params)
        global_params = simulate_exploration_aggregator(aggregator)