from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  
        self.lr = 0.1  # Initial learning rate
        self.used_clients_per_round = {} # for tracking which clients were used in each round

    def client_selection(self, client_ids: List[str]) -> List[str]:        
        X = 5  # number of initial rounds to select only benign or malicious
        if self.round < X:
            # Filter out any client IDs that contain 'malicious'
            benign_clients = [cid for cid in client_ids if "malicious" not in cid.lower()]
            logger.info(f"Round {self.round}: Selecting only benign clients: {benign_clients}")
            return benign_clients
        else:
            logger.info(f"Round {self.round}: Selecting all clients: {client_ids}")
            return client_ids
        
    
    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        # (Optional) Adjust the learning rate every 10 rounds
        if self.round % 10 == 0 and self.round > 0:
            self.lr *= 0.1
        self.round += 1
        return {"learning_rate": self.lr}

    def aggregate(self, previous_global, client_updates):

        if len(client_updates) == 0:
            return previous_global

        dim_proportion = 0.3
        all_2D_layers = [i for i in range(len(previous_global)) 
                        if previous_global[i].ndim >= 2]
        
        # take out the 2D layers
        if len(all_2D_layers) == 0:
            logger.warning("No 2D layers found. Returning previous global.")
            return previous_global

        # sub-sample only 2D layers
        nr_dimension = len(all_2D_layers)
        subsample_size = max(1, int(dim_proportion * nr_dimension))

        # outlier scores
        dnc_for_this_round = {cid: 0.0 for cid in client_updates}

        # select random 2D layers from 'all_2D_layers'
        chosen_layer_indices = random.sample(all_2D_layers, subsample_size)

        for layer_idx in chosen_layer_indices:
            for client_id, (client_params, _) in client_updates.items():
                param_layer = client_params[layer_idx]
                U, S, Vt = np.linalg.svd(param_layer, full_matrices=False)
                v1 = Vt[0]
                projection = param_layer @ v1
                mean_proj = np.mean(projection)
                outlier_score = np.abs(projection - mean_proj)
                dnc_for_this_round[client_id] += np.mean(outlier_score)

        # average across the chosen layers
        avg_scores = {}
        for cid in dnc_for_this_round:
            avg_scores[cid] = dnc_for_this_round[cid] / float(subsample_size)

        # threshold for incl. into fedavg
        # mean + 1 * std
        all_scores = list(avg_scores.values())
        if all_scores:
            mean_outlier = np.mean(all_scores)
            std_outlier = np.std(all_scores)
            threshold = mean_outlier + 1 * std_outlier
        else:
            threshold = float("inf")

        trusted_clients = [
            cid for cid, score in avg_scores.items()
            if score < threshold
        ]

        if not trusted_clients:
            logger.warning("No clients below threshold; returning previous global.")
            return previous_global

        self.used_clients_per_round[self.round] = trusted_clients

        # FedAvg trusted clients
        trusted_clients_num_examples = {
            cid: client_updates[cid][1]["num_examples"]
            for cid in trusted_clients
        }
        total_weight = sum(trusted_clients_num_examples.values())
        if total_weight == 0:
            return previous_global

        weighted_sum = [np.zeros_like(param) for param in previous_global]
        for cid in trusted_clients:
            num_examples = trusted_clients_num_examples[cid]
            for i, param in enumerate(client_updates[cid][0]):
                weighted_sum[i] += param * num_examples

        logger.info(f"Aggregating from {len(trusted_clients)} trusted clients this round.")
        x_agg = [w / total_weight for w in weighted_sum]

        for round in self.used_clients_per_round:
            logger.info(f"Used clients Round {round}: {self.used_clients_per_round[round]}")
        
        return x_agg
        


# def simulate_trmean_aggregator(aggregator):

#     previous_global = [
#         np.random.randn(64, 784),   # fc1.weight
#         np.random.randn(64),        # fc1.bias
#         np.random.randn(32, 64),    # fc2.weight
#         np.random.randn(32),        # fc2.bias
#         np.random.randn(10, 32),    # fc3.weight
#         np.random.randn(10),        # fc3.bias
#     ]

#     client_updates = {}
#     for client_idx in range(4):
#         client_id = f"client_{client_idx}"
#         if client_idx < 3:
#             client_params = []
#             for layer in previous_global:
#                 noise = 0.01 * np.random.randn(*layer.shape)
#                 client_params.append(layer + noise)

#             metadata = {"num_examples": np.random.randint(5, 100), "client_id": client_id}  
#             client_updates[client_id] = (client_params, metadata)
#         else: 
#             client_params = []
#             for layer in previous_global:
#                 noise = 100 * np.random.randn(*layer.shape)
#                 client_params.append(layer + noise)

#             metadata = {"num_examples": np.random.randint(5, 100), "client_id": client_id}  
#             client_updates[client_id] = (client_params, metadata)

#     new_global = aggregator.aggregate(previous_global, client_updates)

#     print("used clients per round:")
#     print(aggregator.used_clients_per_round)

#     print("=== Aggregation Complete ===")
#     for i, layer in enumerate(new_global):
#         print(f"Layer {i} shape: {layer.shape}")
#         # Check if any NaNs
#         if np.isnan(layer).any():
#             print(f"Layer {i} has NaNs!")
#         else:
#             print(f"Layer {i} is OK, mean={layer.mean():.4f}, std={layer.std():.4f}")

#     # Return updated global for chaining in next round
#     return new_global


# if __name__ == "__main__":
#     aggregator = ServerFunctions()

#     global_params = None
#     for round_num in range(2):
#         print(f"\n=== Simulation Round {round_num + 1} ===")
#         global_params = simulate_trmean_aggregator(aggregator)