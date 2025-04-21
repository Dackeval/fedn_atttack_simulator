from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  # Keep track of training rounds
        self.lr = 0.1  # Initial learning rate
        self.used_clients_per_round = {}

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
        # Adjust the learning rate every 10 rounds
        if self.round % 10 == 0:
            self.lr *= 0.1
        self.round += 1
        return {"learning_rate": self.lr}

    def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # KRUM aggregation        
        # f is for the tolerated malicious clients
        f = 1
        # Multi k-Krum 
        k = 3   
        num_clients = len(client_updates)
        
        # add the client ids to a list
        client_ids = list(client_updates.keys())
        distance_sums = {}
        # calculate the euclidean distance sum for each client
        for client_id in client_ids:
            distances = []
            for other_client_id in client_ids:
                if client_id == other_client_id:
                    continue
                # extract params from each client
                params1 = client_updates[client_id][0]
                params2 = client_updates[other_client_id][0]

                if len(params1) != len(params2):
                    continue  # continue if they have diff. num. layers

                layer_distances = []
                for layer1, layer2 in zip(params1, params2):
                    if layer1.shape != layer2.shape:
                        continue # continue if layers are of different shapes

                    layer_distances.append(np.linalg.norm(layer1 - layer2))

                total_distance = sum(layer_distances)  # sum all layers
                distances.append(total_distance)
            # sort the clients by the sum of distances to other clients
            distances.sort()
            # sum the tolerated clients distances
            distances_to_sort = max(1, num_clients - f - 1)
            sum_distances = sum(distances[:distances_to_sort])
            # append distance sum to the dictionary with client id as key
            distance_sums[client_id] = sum_distances

        used_clients = set()
        # if num clients are below 2, return the first available model
        if num_clients == 0:
            return previous_global
        elif num_clients == 1:
            logger.warning("Not enough clients for aggregation. Returning first available model.")
            first_client_id = list(client_updates.keys())[0]
            self.used_clients_per_round[self.round] = [first_client_id]
            logger.info(f"Used clients Round {self.round}: {first_client_id}")
            return list(client_updates.values())[0][0]
        # if k is greater than the number of clients - f, then select one or run KRUM depending on k
        elif k > num_clients - f and num_clients < 2:
            selected_id = min(distance_sums, key=distance_sums.get)
            x_agg = client_updates[selected_id][0]        
        # else run multi k-krum with fedavg on remaining clients
        else: 
            # select the k smallest euclidean distance sum updates
            k_smallest_clients = sorted(distance_sums, key=distance_sums.get)[:k]
            k_smallest_num_examples = {client_id: client_updates[client_id][1]["num_examples"] for client_id in k_smallest_clients}
            # fedAvg for the remaining clients 
            weighted_sum = [np.zeros_like(param) for param in previous_global]
            total_weight = 0
            # for loop over the k-smallest euclidean distance sum updates
            for client_id in k_smallest_clients:
                num_examples = k_smallest_num_examples.get(client_id, 1)
                total_weight += num_examples
                # fetch params from the client
                client_parameters = client_updates[client_id][0]  
                used_clients.add(client_id)
                for i, param in enumerate(client_parameters):
                    weighted_sum[i] += param * num_examples
            
            self.used_clients_per_round[self.round] = list(used_clients)
            logger.info("Models aggregated")
            x_agg = [param / total_weight for param in weighted_sum]

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