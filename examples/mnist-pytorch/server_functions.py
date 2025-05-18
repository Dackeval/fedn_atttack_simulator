from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random


class ServerFunctions(ServerFunctionsBase):
    def __init__(self):
        super().__init__()

        self.aggregator_type  = "fedavg"

        #  late-client state 
        self.late_delay       = 0                  # skip rounds
        self.late_set         = set()              # indices

        #  learning-rate / session control 
        self.round            = 0                  # global round counter
        self.lr               = 0.1
        self.lr_decay_period  = 10                 # decay every 10 rounds
        self.session_length   = 30                 # 30-round “epochs”
        self.session_counter  = 0                  # how many 30-round blocks seen
        self.selected_clients_per_round = {}
        self.used_clients_per_round = {}

    #  helper: parse client-id into idx / defense / delay / late list
    def _quick_parse(self, cid: str):
        parts = cid.split("_")
        if len(parts) < 4 or parts[1] != "client":
            raise ValueError
        idx       = int(parts[2])
        defense    = parts[3]
        delay      = 0
        late_list  = []
        for p in parts[4:]:
            if p.startswith("delay"):
                delay = int(p[5:])               # after "delay"
            elif p.startswith("late"):
                late_list = [int(x) for x in p[4:].split("-") if x]
        return idx, defense, delay, late_list


    def client_selection(self, client_ids: List[str]) -> List[str]:
        for cid in client_ids:
            try:
                _, defense, delay, late_list = self._quick_parse(cid)
                self.aggregator_type = defense
                self.late_delay      = delay
                self.late_set        = set(late_list)
                break
            except Exception:
                continue
        logger.info(f"[CFG] aggregator={self.aggregator_type} "
                    f"late_delay={self.late_delay} late_set={sorted(self.late_set)}")

        allowed = []
        for cid in client_ids:
            try:
                idx, *_ = self._quick_parse(cid)
            except Exception:                # id not in our pattern
                allowed.append(cid)
                continue

            if idx in self.late_set and self.round < self.late_delay:
                logger.info(f"[{self.aggregator_type}] exclude {cid}")
            else:
                allowed.append(cid)
        
        # log the selected clients for this round
        logger.info(f"Round {self.round} - selected clients: {allowed}")

        return allowed


    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        round_in_session = self.round % self.session_length

        # start of a new 30-round “session”  (but not round 0)
        if round_in_session == 0 and self.round != 0:
            self.session_counter += 1
            self.lr = 0.1
            logger.info(f"Round {self.round}: new session → LR reset to {self.lr}")

        # every 10 rounds (except round 0)  → decay LR
        elif self.round > 0 and self.round % self.lr_decay_period == 0:
            self.lr *= 0.1
            logger.info(f"Round {self.round}: LR decayed to {self.lr}")

        self.round += 1
        return {"learning_rate": self.lr}


    def aggregate(self, 
        previous_global: List[np.ndarray], 
        client_updates: Dict[str, Tuple[List[np.ndarray], dict]]
    ) -> List[np.ndarray]:
        """
        Main aggregator switch, selecting between:
        - FedAvg
        - Krum
        - Multi-Krum
        - TrMean
        - DNC
        etc.
        """

        if self.aggregator_type.lower() == "fedavg":
            return self._aggregate_fedavg(previous_global, client_updates)
        elif self.aggregator_type.lower() == "trmean":
            return self._aggregate_trmean(previous_global, client_updates)
        elif self.aggregator_type.lower() == "multi-krum":
            return self._aggregate_multi_krum(previous_global, client_updates)
        elif self.aggregator_type.lower() == "dnc":
            return self._aggregate_dnc(previous_global, client_updates)
        else:
            logger.warning(f"[{self.aggregator_type}] aggregator not recognized. Defaulting to FedAvg.")
            return self._aggregate_fedavg(previous_global, client_updates)

    def _aggregate_trmean(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        """ TrMean aggregation. """
        logger.info(f"Round {self.round} - TrMean aggregation")
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
                    # if after min() too big, skip trimming
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

    def _aggregate_fedavg(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        """ FedAvg aggregator """
        logger.info(f"Round {self.round} - FedAvg aggregation")
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
    
    def _aggregate_multi_krum(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        # KRUM aggregation        
        logger.info(f"Round {self.round} - Multi-Krum aggregation")
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
        # else, run multi k-krum with fedavg on remaining clients
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
    

    def _aggregate_dnc(self, previous_global, client_updates):
        """ DNC aggregation """
        logger.info(f"Round {self.round} - DNC aggregation")
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

        if self.round % 30 == 0 or self.round % 29 == 0:
            for round in self.used_clients_per_round:
                logger.info(f"Used clients Round {round}: {self.used_clients_per_round[round]}")
            
        return x_agg
        