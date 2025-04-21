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
        # trimming factor and number of clients for TrMean
        trimming_fraction = 0.1
        num_clients = len(client_updates)
        # for tracking the participating clients
        client_ids = list(client_updates.keys())
        # create list to store parameters for each layer
        layerwise_params = [[] for _ in range(len(previous_global))]
        sample_counts = []


        # if less than 3 fallback to FedAvg
        if num_clients < 3:
            logger.info("Not enough clients for TrMean (need >=3). Falling back to FedAvg.")
            self.used_clients_per_round[self.round] = client_ids
            return self.fedavg(previous_global, client_updates)

        # for tracking 
        for cid in client_ids:
            client_parameters, metadata = client_updates[cid]
            w = metadata.get("num_examples", 1)
            sample_counts.append(float(w))

            for layer_idx, layer_param in enumerate(client_parameters):
                layerwise_params[layer_idx].append(layer_param)

        # for tracking the used clients
        final_used_clients = set()

        # coordinate-wise TrMean for each layer
        new_global = []

        for layer_idx in range(len(previous_global)):

            stacked = np.stack(layerwise_params[layer_idx], axis=0)
            original_shape = stacked.shape[1:]
            flattened = stacked.reshape(num_clients, -1) # shape: (num_clients, NxM)
            
            aggregated_flat = np.zeros(flattened.shape[1], dtype=flattened.dtype)

            for col in range(flattened.shape[1]):
                layer_triplett = []
                for i in range(num_clients):
                    params = flattened[i, col]
                    layer_triplett.append((params, client_ids[i], sample_counts[i]))
                
                layer_triplett.sort(key=lambda x: x[0]) # sort by parameter value

                temp_number_of_trimmed_clients = int(trimming_fraction * num_clients)
                max_trimmed = (num_clients - 1) // 2
                number_of_trimmed_clients = max(1, min(temp_number_of_trimmed_clients, max_trimmed))

                if number_of_trimmed_clients * 2 >= num_clients:
                    # If even after min() it's too big, skip trimming entirely
                    trimmed_triplets = layer_triplett
                else:
                    trimmed_triplets = layer_triplett[number_of_trimmed_clients:-number_of_trimmed_clients]

                total_weight = sum(tr[2] for tr in trimmed_triplets)  # sum of sample_counts
                if total_weight == 0:
                    aggregated_value = np.mean([tr[0] for tr in trimmed_triplets]) if trimmed_triplets else 0.0
                else:
                    weighted_sum = sum(tr[0] * tr[2] for tr in trimmed_triplets)
                    aggregated_value = weighted_sum / total_weight

                aggregated_flat[col] = aggregated_value

                # Mark those clients as "used" for this coordinate
                for val, c_id, w in trimmed_triplets:
                    final_used_clients.add(c_id)

            # reshape to original layer shape
            new_layer = aggregated_flat.reshape(original_shape)
            new_global.append(new_layer)

        # log which clients contributed to aggregation
        logger.info(f"Round {self.round} - TrMean used clients (union across coordinates): {sorted(final_used_clients)}")
        for round in self.used_clients_per_round:
            logger.info(f"Round {round}: {self.used_clients_per_round[round]}")

        return new_global

    def fedavg(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        """ Fallback FedAvg aggregator, used when not enough clients for TrMean. """
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0.0
        for client_id, (params, metadata) in client_updates.items():
            w = metadata.get("num_examples", 1)
            total_weight += w
            for i, p in enumerate(params):
                weighted_sum[i] += p * w

        if total_weight == 0:
            # fallback to previous global if no clients reported any examples
            return previous_global
        
        logger.info(f"Round {self.round} - TrMean used clients (union across coordinates): {client_updates.keys()}")

        return [p / total_weight for p in weighted_sum]
    

