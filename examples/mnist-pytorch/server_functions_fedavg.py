from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  # Keep track of training rounds
        self.lr = 0.1  # Initial learning rate
        self.used_clients_per_round = {}
        self.exclude_every_kth_round = 5
        self.session_length = 30
        self.lr_decay_period = 10
        self.session_counter = 0  

    def client_selection(self, client_ids: List[str]) -> List[str]:        
        
        round_in_session = (self.round % self.session_length) + 1

        needed_rounds = max(1, self.exclude_every_kth_round -
                               (1 if self.session_counter > 0 else 0))
        
        if round_in_session < needed_rounds:
            # Filter out any client IDs that contain 'malicious'
            benign_clients = [cid for cid in client_ids if "malicious" not in cid.lower()]
            logger.info(f"Round {self.round}: Selecting only benign clients: {benign_clients}")
            return benign_clients
        else:
            logger.info(f"Round {self.round}: Selecting all clients: {client_ids}")
            return client_ids

    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        # Adjust the learning rate every 10 rounds
        round_in_session = (self.round % self.session_length)

        if round_in_session == 0 and self.round != 0:
            self.session_counter += 1
            self.lr = 0.1
            logger.info(f"Round {self.round}: New session! Reset LR to {self.lr}")

        elif self.round > 0 and self.round % self.lr_decay_period == 0:
            self.lr *= 0.1
            logger.info(f"Round {self.round}: Decaying learning rate. New lr: {self.lr}")

        self.round += 1
        return {"learning_rate": self.lr}

    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # Implement a weighted FedAvg aggregation
        client_ids = list(client_updates.keys())
        self.used_clients_per_round[self.round] = client_ids
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0
        for client_id, (client_parameters, metadata) in client_updates.items():
            num_examples = metadata.get("num_examples", 1)
            total_weight += num_examples
            for i, param in enumerate(client_parameters):
                weighted_sum[i] += param * num_examples
        logger.info("Models aggregated")
        for round in self.used_clients_per_round:
            logger.info(f"Used client for Round {round}: {self.used_clients_per_round[round]}")
        return [param / total_weight for param in weighted_sum]
    



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