from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  # Keep track of training rounds
        self.lr = 0.1  # Initial learning rate

    def client_selection(self, client_ids: List[str]) -> List[str]:
        # Select up to 10 random clients
        return random.sample(client_ids, min(len(client_ids), 10))

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

        # if lezs than 3 fallback to FedAvg
        if num_clients < 3:
            logger.info("Not enough clients for TrMean (need >=3). Falling back to FedAvg.")
            return self.fedavg(previous_global, client_updates)

        # coordinate-wise TrMean for each layer
        new_global = []
        for layer_idx in range(len(previous_global)):
            # gather all parameters for each layer + their sample counts
            layer_params = []
            sample_counts = []
            for client_id, (client_parameters, metadata) in client_updates.items():
                w = metadata.get("num_examples", 1)
                sample_counts.append(float(w))

                # stack the parameters in layer_params per client
                layer_params.append(client_parameters[layer_idx])

            # reshape from (num_clients, (M,N)) to (num_clients, M, N) for easier manipulation 
            # or (num_clients, D) for a vector, or (num_clients,) for a scalar/bias
            stacked = np.stack(layer_params, axis=0)
            # for reshaping back later.
            original_shape = stacked.shape[1:] 
            # reshape to (num_client, N*M) 
            flattened = stacked.reshape(num_clients, -1)  
            sample_counts = np.array(sample_counts)       # (num_clients,) to match with parameters

            # We'll build a new array for the aggregated layer in flat form
            aggregated_flat = np.zeros(flattened.shape[1], dtype=flattened.dtype)

            # looping over each coordinate ie layer
            for col_idx in range(flattened.shape[1]):
                # values is the params for the current clients 
                values = flattened[:, col_idx]  

                # Pair them with sample counts, e.g. [(params, num_examples), ...]
                pairs = list(zip(values, sample_counts))

                # sort by the parameter values
                pairs.sort(key=lambda x: x[0])

                trim_amount = int(trimming_fraction * num_clients)
                max_trim = (num_clients - 1) // 2  # ensures at least 1 remains
                trim_amount = max(1, min(trim_amount, max_trim))
                
                if trim_amount * 2 >= num_clients:
                    # If even after min() it's too big, skip trimming
                    trimmed_pairs = pairs
                else:
                    trimmed_pairs = pairs[trim_amount : -trim_amount]

                # fedavg on the remaining 
                total_weight = sum(w for _, w in trimmed_pairs)
                if total_weight == 0:
                    # fallback to mean if total num_examples is zero
                    aggregated_value = np.mean([val for val, _ in trimmed_pairs])
                else:
                    weighted_sum = sum(val * w for val, w in trimmed_pairs)
                    aggregated_value = weighted_sum / total_weight

                aggregated_flat[col_idx] = aggregated_value

            # reshape back to the original layer shape (e.g. (M, N))
            new_layer = aggregated_flat.reshape(original_shape)
            new_global.append(new_layer)

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

        return [p / total_weight for p in weighted_sum]
    

