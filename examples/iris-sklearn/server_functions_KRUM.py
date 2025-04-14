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
        num_clients = len(client_updates)

        if num_clients < 2:
            logger.warning("Not enough clients for aggregation. Returning first available model.")
            return list(client_updates.values())[0][0]  

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

        for client_id, distance_sum in distance_sums.items():
            logger.warning(f"Client: {client_id}, Distance Sum: {distance_sum}")
        
        selected_id = min(distance_sums, key=distance_sums.get)
        x_agg = client_updates[selected_id][0]
        return x_agg


