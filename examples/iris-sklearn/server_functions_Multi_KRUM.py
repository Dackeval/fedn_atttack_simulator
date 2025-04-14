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
        # KRUM aggregation        
        # f is for the tolerated malicious clients
        f = 0
        # Multi k-Krum 
        k = 2   
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

        # if num clients are below 2, return the first available model
        if num_clients == 0:
            return previous_global
        elif num_clients == 1:
            logger.warning("Not enough clients for aggregation. Returning first available model.")
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
                
                for i, param in enumerate(client_parameters):
                    weighted_sum[i] += param * num_examples
            
            logger.info("Models aggregated")
            x_agg = [param / total_weight for param in weighted_sum]

        
        return x_agg


