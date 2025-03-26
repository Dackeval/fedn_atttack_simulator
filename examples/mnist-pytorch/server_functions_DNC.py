from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  # Keep track of training rounds
        self.lr = 0.1  # Initial learning rate
        self.dnc = {}  
        self.trusted_clients = set()
        self.client_rounds = {}

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
        
        dim_proportion = 0.3
        num_clients = len(client_updates)

        if num_clients == 0:
            return previous_global

        nr_dimension = len(previous_global)
        subsample_size = max(1, int(dim_proportion * nr_dimension))  # atleast 1 dim.

        # loop over nr dimensions subselected 
        for _ in range(subsample_size):
            random_dimension = random.choice(range(nr_dimension))

            # one client update at a time is calculated.
            for client_id, (client_parameters, _) in client_updates.items():
                if client_id not in self.dnc:
                    self.dnc[client_id] = 0  # Initialize cumulative score
                    self.client_rounds[client_id] = 0  # track participation

                param_layer = client_parameters[random_dimension]  # Extract specific dimension
                if param_layer.ndim < 2:
                    # for bias, skip 
                    continue
                else: 
                    # Perform SVD per client
                    U, S, Vt = np.linalg.svd(param_layer, full_matrices=False)
                    v1 = Vt[0]
                    # Calculate the projection of the layer on the first singular vector
                    projection = param_layer @ v1
                    mean_proj = np.mean(projection)
                    outlier_score = np.abs(projection - mean_proj)

                    # += to the sum of outlier scores over rounds
                    self.dnc[client_id] += np.mean(outlier_score)
                    self.client_rounds[client_id] += 1  # tracking how many times the client been eval.

        avg_scores = {}
        for client_id in self.dnc:
            if self.client_rounds[client_id] > 0:
                avg_scores[client_id] = self.dnc[client_id] / self.client_rounds[client_id]


        # setting threshold dynamically with std and mean
        all_scores = list(avg_scores.values())
        if all_scores:
            mean_outlier = np.mean(all_scores)
            std_outlier = np.std(all_scores)
            threshold = mean_outlier + 2 * std_outlier
        else:
            threshold = float('inf')

        # select trusted clients based on long-term behavior
        trusted_clients = [
            client_id for client_id, avg_score in avg_scores.items()
            if avg_score < threshold
        ]
        self.trusted_clients = set(trusted_clients)

        trusted_clients_num_examples = {
            client_id: client_updates[client_id][1]["num_examples"]
            for client_id in trusted_clients if client_id in client_updates
        }

        if not trusted_clients_num_examples:
            return previous_global  # avoid div. by zero - no trusted clients

        # fedAvg with trusted clients
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = sum(trusted_clients_num_examples.values())

        for client_id in trusted_clients_num_examples:
            num_examples = trusted_clients_num_examples[client_id]
            client_parameters = client_updates[client_id][0]  

            for i, param in enumerate(client_parameters):
                weighted_sum[i] += param * num_examples

        # if total_weight == 0:
        #     logger.warning("All trusted clients reported 0 examples. Returning previous_global.")
        #     return previous_global

        logger.info(f"Models aggregated from {len(trusted_clients)} trusted clients.")
        x_agg = [param / total_weight for param in weighted_sum]

        return x_agg


                