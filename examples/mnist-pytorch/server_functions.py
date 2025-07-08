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

        # DNC params
        self.p  = 0.08     # coordinate subsample fraction
        self.f  = 1     # tolerated Byzantines
        self.R  = 1    # SVD repetitions

        # EE_TrMean params
        self.trimming_fraction = 0.1
        self.score_gap_alpha = 0.1
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.9
        self.client_scores: Dict[str, float] = {}
        self.client_rounds: Dict[str, int] = {}
        self.alpha_min   = 0.30     # starting value
        self.alpha_max   = 0.90     # value to reach
        self.alpha_ramp  = "session"   # "session" or "global"
        self.total_round_target = 300   # only used if ramp == "global"
        

    #  helper: parse client-id into idx / defense / delay / late list
    def _quick_parse(self, cid: str):
        # split client names by "_"
        parts = cid.split("_")
        # check if the client id is in the expected format
        if len(parts) < 4 or parts[1] != "client":
            raise ValueError
        # extract the index, defense type, delay
        idx       = int(parts[2])
        defense    = parts[3]
        delay      = 0
        late_list  = []
        # extract delay and late list if present
        for p in parts[4:]:
            # delay is a single integer
            if p.startswith("delay"):
                delay = int(p[5:])
            # late_list is a range of indices
            elif p.startswith("late"):
                late_list = [int(x) for x in p[4:].split("-") if x]
        # return the parsed values
        return idx, defense, delay, late_list


    def client_selection(self, client_ids: List[str]) -> List[str]:
        # parse the first client id to determine the aggregator type, delay and late clients
        for cid in client_ids:
            try:
                _, defense, delay, late_list = self._quick_parse(cid)
                self.aggregator_type = defense
                self.late_delay      = max(0, delay - 1)
                self.late_set        = set(late_list)
                break
            except Exception:
                continue
        # log the aggregator type and late clients
        logger.info(f"[CFG] aggregator={self.aggregator_type} "
                    f"late_delay={self.late_delay} late_set={sorted(self.late_set)}")
        # keep a round in session counter to determine the current round in the session
        round_in_session = self.round % self.session_length
        # filter out clients that are in the late set
        allowed = []
        for cid in client_ids:
            # get the client id
            try:
                idx, *_ = self._quick_parse(cid)
            except Exception:                # id not in our pattern
                allowed.append(cid)
                continue
            # if the client is in the late set and the round in session is less than the late delay, skip it
            if idx in self.late_set and round_in_session < self.late_delay:
                logger.info(f"[{self.aggregator_type}] exclude {cid}")
            # else append the client id to the allowed list
            else:
                allowed.append(cid)
        
        # log the selected clients for this round
        logger.info(f"Round {self.round} - selected clients: {allowed}")

        return allowed


    def client_settings(self, global_model: List[np.ndarray]) -> dict:
        round_in_session = self.round % self.session_length

        # start of a new “session”  (but not round 0), need to reset some parameters
        if round_in_session == 0 and self.round != 0:
            self.session_counter += 1
            self.lr = 0.1
            self.trimming_fraction = 0.1
            self.score_gap_alpha = 0.2
            self.epsilon = 1.0
            self.epsilon_end = 0.05
            self.epsilon_decay = 0.95
            self.client_scores: Dict[str, float] = {}
            self.client_rounds: Dict[str, int] = {}
            self.top_decay = 0.98
            self.alpha_min   = 0.10     # starting value
            self.alpha_max   = 0.90     # value to reach
            self.alpha_ramp  = "session"   # "session" or "global"
            self.total_round_target = 300  
            logger.info(f"Round {self.round}: new session → LR reset to {self.lr}")

        # every 10 rounds (except round 0)  → decay LR
        elif self.round > 0 and self.round % self.lr_decay_period == 0:
            self.lr *= 0.1
            logger.info(f"Round {self.round}: LR decayed to {self.lr}")

        # increment the round counter
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
        - EE-TrMean
        """
        # gate for the aggregator type
        if self.aggregator_type.lower() == "fedavg":
            return self._aggregate_fedavg(previous_global, client_updates)
        elif self.aggregator_type.lower() == "trmean":
            return self._aggregate_trmean(previous_global, client_updates)
        elif self.aggregator_type.lower() == "multi-krum":
            return self._aggregate_multi_krum(previous_global, client_updates)
        elif self.aggregator_type.lower() == "dnc":
            return self._aggregate_dnc(previous_global, client_updates)
        elif self.aggregator_type.lower() == "ee-trmean":
            return self._aggregate_ee_trmean(previous_global, client_updates)
        else:
            logger.warning(f"[{self.aggregator_type}] aggregator not recognized. Defaulting to FedAvg.")
            return self._aggregate_fedavg(previous_global, client_updates)



    def _aggregate_trmean(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        """ TrMean aggregation. """
        logger.info(f"Round {self.round} - TrMean aggregation")
        # trimming factor and number of clients 
        trimming_fraction = self.trimming_fraction
        num_clients = len(client_updates)
        # for tracking the participating clients
        client_ids = list(client_updates.keys())
        # create list to store parameters for each layer with same shape as previous_global
        layerwise_params = [[] for _ in range(len(previous_global))]
        # for tracking number of data samples per client
        sample_counts = []


        # if less than 3 fallback to FedAvg
        if num_clients < 3:
            logger.info("Not enough clients for TrMean (need >=3). Falling back to FedAvg.")
            self.used_clients_per_round[self.round] = client_ids
            return self._aggregate_fedavg(previous_global, client_updates)

        # for tracking num of examples per client
        for cid in client_ids:
            client_parameters, metadata = client_updates[cid]
            # weight is the number of examples
            w = metadata.get("num_examples", 1)
            sample_counts.append(float(w))

            # build a layerwise param list, with shape: layerwise_params[0] = [ client1.layer0 , client2.layer0 , … , clientN.layer0 ]
            for layer_idx, layer_param in enumerate(client_parameters):
                layerwise_params[layer_idx].append(layer_param)

        # for tracking the used clients
        final_used_clients = set()

        # coordinate-wise TrMean for each layer
        new_global = []

        # iterate over each layer in the previous global model
        for layer_idx in range(len(previous_global)):
            
            # stack the parameters for this layer across all clients, so each clients parameters become a row
            stacked = np.stack(layerwise_params[layer_idx], axis=0)
            # save the original shape of the layer for reshaping later
            original_shape = stacked.shape[1:]
            flattened = stacked.reshape(num_clients, -1) # shape: (num_clients, NxM), each clients parameters are flattened into a single row
            
            # create an array to store the aggregated parameters for this layer
            aggregated_flat = np.zeros(flattened.shape[1], dtype=flattened.dtype)

            # iterate over each coordinate (column) in the flattened parameters
            for col in range(flattened.shape[1]):
                # create a list of tuples (parameter value, client id, sample count) for averaging
                layer_triplett = []
                # iterate of each row in the flattened parameters, ie. each coordinate for each client
                for i in range(num_clients):
                    params = flattened[i, col]
                    # append the parameter value, client id and sample count to the list
                    layer_triplett.append((params, client_ids[i], sample_counts[i]))
                
                # sort the layer_triplett by parameter value
                layer_triplett.sort(key=lambda x: x[0]) 
                # trim the layer_triplett based on the trimming fraction
                temp_number_of_trimmed_clients = int(trimming_fraction * num_clients)
                # calculate the maximum number of clients that can be trimmed
                max_trimmed = (num_clients - 1) // 2
                # ensure at least one client is left after trimming (if trimming_fraction is too high)
                number_of_trimmed_clients = max(1, min(temp_number_of_trimmed_clients, max_trimmed))

                # safe guard against too many clients being trimmed
                if number_of_trimmed_clients * 2 >= num_clients:
                    # if after min() too big, skip trimming
                    trimmed_triplets = layer_triplett
                # trim the layer_triplett by removing the top and bottom `number_of_trimmed_clients` from triplett
                else:
                    trimmed_triplets = layer_triplett[number_of_trimmed_clients:-number_of_trimmed_clients]

                # calculate the aggregated value for this coordinate
                total_weight = sum(tr[2] for tr in trimmed_triplets)  # sum of sample_counts
                # if for chance no examples are reported, use mean of trimmed values
                if total_weight == 0:
                    aggregated_value = np.mean([tr[0] for tr in trimmed_triplets]) if trimmed_triplets else 0.0
                # otherwise calculate the weighted mean
                else:
                    weighted_sum = sum(tr[0] * tr[2] for tr in trimmed_triplets)
                    aggregated_value = weighted_sum / total_weight

                # append the aggregated value to the aggregated_flat array
                aggregated_flat[col] = aggregated_value

                # Mark those clients as "used" for this coordinate, for calculating final used clients
                for val, c_id, w in trimmed_triplets:
                    final_used_clients.add(c_id)

            # reshape to original layer shape
            new_layer = aggregated_flat.reshape(original_shape)
            # append the new layer to the new_global list
            new_global.append(new_layer)

        # log which clients contributed to aggregation
        logger.info(f"Round {self.round} - TrMean used clients (union across coordinates): {sorted(final_used_clients)}")
        if self.round % 30 == 0 or self.round % 29 == 0:
            for round in self.used_clients_per_round:
                logger.info(f"Used clients Round {round}: {self.used_clients_per_round[round]}")

        # aggregated model is a list of numpy arrays, one for each layer
        return new_global

    def _aggregate_fedavg(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
        """ FedAvg aggregator """
        logger.info(f"Round {self.round} - FedAvg aggregation")
        # weights from metadata
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        # total weight for averaging
        total_weight = 0.0
        # iterate over the client updates
        for client_id, (params, metadata) in client_updates.items():
            # fetch the number of examples from metadata, default to 1 if not present
            w = metadata.get("num_examples", 1)
            # append num examples to total weight
            total_weight += w
            # add the parameters to the weighted sum
            for i, p in enumerate(params):
                weighted_sum[i] += p * w

        if total_weight == 0:
            # fallback to previous global if no clients reported any examples
            return previous_global
        
        logger.info(f"Round {self.round} - FedAvg used clients: {list(client_updates.keys())}")
        # calculate the average by dividing the weighted sum by the total weight
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
            # distances from this client to all other clients
            distances = []
            # iterate over all other clients
            for other_client_id in client_ids:
                # skip if the client is the same as the other client
                if client_id == other_client_id:
                    continue
                # extract params from each client
                params1 = client_updates[client_id][0]
                params2 = client_updates[other_client_id][0]

                # skip if the params are not of the same length, for whatever reason
                if len(params1) != len(params2):
                    continue  # continue if they have diff. num. layers
                
                # calculate the euclidean distance between the parameters of the two clients
                layer_distances = []
                # iterate over the layers of the parameters
                for layer1, layer2 in zip(params1, params2):
                    # skip if the layers are not of the same shape
                    if layer1.shape != layer2.shape:
                        continue 
                    # calculate the euclidean distance between the two layers
                    layer_distances.append(np.linalg.norm(layer1 - layer2))
                # sum the distances for all layers
                total_distance = sum(layer_distances)  # sum all layers
                # append the total distance to the distances list
                distances.append(total_distance)
            # sort the clients by the sum of distances to other clients
            distances.sort()
            # sum the tolerated clients distances, removing the f largest distances and subtracting 1 
            distances_to_sort = max(1, num_clients - f - 1)
            # calculate the sum of the distances for the remaining clients
            sum_distances = sum(distances[:distances_to_sort])
            # append distance sum to the dictionary with client id as key
            distance_sums[client_id] = sum_distances

        # for tracking the used clients in this round
        used_clients = set()
        # if num clients are below 2, return the old model
        if num_clients == 0:
            return previous_global
        # if num clients is 1, return the first available model
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
            # fetch the number of examples for each client
            k_smallest_num_examples = {client_id: client_updates[client_id][1]["num_examples"] for client_id in k_smallest_clients}
            # fedAvg for the remaining clients 
            # initialize the weighted sum and total weight
            weighted_sum = [np.zeros_like(param) for param in previous_global]
            total_weight = 0
            # for loop over the k-smallest euclidean distance sum updates
            for client_id in k_smallest_clients:
                # get the number of examples for this client, default to 1 if not present
                num_examples = k_smallest_num_examples.get(client_id, 1)
                # add the number of examples to the total weight
                total_weight += num_examples
                # fetch params from the client
                client_parameters = client_updates[client_id][0]  
                # note the client as used in this round
                used_clients.add(client_id) 
                # add the parameters to the weighted sum
                for i, param in enumerate(client_parameters):
                    weighted_sum[i] += param * num_examples
            # add to the used clients per round
            self.used_clients_per_round[self.round] = list(used_clients)
            logger.info("Models aggregated")
            # fedavg on the k-smallest clients
            x_agg = [param / total_weight for param in weighted_sum]

        # for logging
        if self.round % 30 == 0 or self.round % 29 == 0:
            for round in self.used_clients_per_round:
                logger.info(f"Used clients Round {round}: {self.used_clients_per_round[round]}")
        
        return x_agg

    # helper function to flatten the weights for DNC
    def _flatten(self, weights):
        # flatten each client parameter list into a single vector for DNC
        return np.concatenate([w.ravel() for w in weights])

    def _aggregate_dnc(self, previous_global, client_updates):
        """ DNC aggregator """
        N = len(client_updates)
        # if no clients or too few clients, fallback to FedAvg
        if N == 0 or N <= self.f:
            logger.warning("DnC: fallback to FedAvg.")
            return self._aggregate_fedavg(previous_global, client_updates)

        # get the client ids and flatten the updates
        ids          = list(client_updates.keys())
        flat_updates = [self._flatten(client_updates[cid][0]) for cid in ids]
        # stack the flattened updates into a 2D array, with each row being a client update
        Z_full       = np.stack(flat_updates, axis=0)
        # get the number of coordinates (features) in the updates
        d_full       = Z_full.shape[1]
        # calculate the number of coordinates to sample, based on the fraction p
        d_sub        = max(1, int(self.p * d_full))

        # initialize a boolean array to track surviving clients
        survivor = np.ones(N, dtype=bool)
        # iterate R times to perform the DNC algorithm
        for _ in range(self.R):
            # randomly sample d_sub coordinates from the full set of coordinates, 
            # d_full is the total number of coordinates, d_sub is the number of coordinates to sample, replace=False ensures no duplicates
            coords = np.random.choice(d_full, d_sub, replace=False)
            # fetch the sampled coordinates from all clientss
            Z      = Z_full[:, coords]
            # center the data by subtracting the mean of each coordinate
            Z      = Z - Z.mean(axis=0, keepdims=True)
            # perform SVD on the centered data, full_matrices=False returns only the singular values and vectors
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            # take the first right singular vector (the one with the largest singular value)
            u       = Vt[0]
            # calculate the scores for each client by taking the absolute value of the dot product of Z and u, 
            # essentially the rows of Z (client updates) projected onto the first right singular vector
            scores  = np.abs(Z @ u)
            # keep the N - f clients with the lowest scores (f is the number of tolerated Byzantine clients)
            keep    = np.argsort(scores)[:N - self.f]
            # create a mask to keep only the clients that are not pruned
            mask    = np.zeros(N, dtype=bool); mask[keep] = True
            # update the survivor array with the mask
            survivor &= mask
            # if all clients are pruned, fallback to FedAvg, for too high f
            if survivor.sum() == 0:
                logger.warning("DnC pruned all clients, FedAvg fallback.")
                return self.fedavg(previous_global, client_updates)

        # filter the client ids based on the survivor mask
        trusted = [cid for cid, m in zip(ids, survivor) if m]
        # log the number of trusted clients
        self.used_clients_per_round[self.round] = trusted
        logger.info(f"DnC kept {len(trusted)}/{N} clients.")

        # FedAvg on survivors
        # first calculate the total weight, which is the sum of the number of examples reported by each surviving client
        total = sum(client_updates[cid][1].get("num_examples",1) for cid in trusted)
        # create a list of zero arrays with the same shape as the previous global model
        out   = [np.zeros_like(v) for v in previous_global]
        # iterate over the trusted clients and accumulate their updates, weighted by the number of examples
        for cid in trusted:
            # fetch the number of examples from metadata, default to 1 if not present
            w = client_updates[cid][1].get("num_examples",1)
            # accumulate the updates for each layer
            for i, v in enumerate(client_updates[cid][0]):
                # add the weighted update to the output
                out[i] += w * v

        # log the used clients for this round, a
        if self.round % self.session_length == 0:
            for round in self.used_clients_per_round:
                logger.info(f"Used clients Round {round}: {self.used_clients_per_round[round]}")

        # return the averaged updates, divided by the total weight
        return [v / total for v in out]
    
    # helper function to decay epsilon for EE-TrMean
    def _decay_epsilon(self):
        # as long as epsilon is above the end value, decay it
        if self.epsilon > self.epsilon_end:
            # decay epsilon by multiplying it with the decay factor
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _avg_score(self, cid: str) -> float:
        # average score = total survived coords / rounds participated
        rounds = max(1, self.client_rounds.get(cid, 0))
        return self.client_scores.get(cid, 0.0) / rounds
    
    def _current_alpha(self) -> float:
        # calculate the current alpha value based on the alpha ramp type
        if self.alpha_ramp == "session":
           # progress within the current session (0 … 1)
           progress = (self.round % self.session_length) / (self.session_length - 1)
        else:  # global ramp
           # progress within the whole session (0 … 1)
           progress = min(1.0, self.round / float(self.total_round_target))
        # Sqare root easing for achieving a steeper initial ramp
        eased = progress ** 0.5

        # calculate the alpha value based on the eased progress
        return self.alpha_min + (self.alpha_max - self.alpha_min) * eased
    
    # helper function to trim the triplets based on the trimming fraction
    def _trim(self, trips):
        """trips is sorted list of (value, cid, weight)."""
        # number of trips, ie number of clients
        m = len(trips)
        # always trim at least one client, but not more than half of the clients
        t = max(1, min(int(self.trimming_fraction * m), (m - 1) // 2))
        # if the number of clients is less than 2 * t, return the whole list
        if 2 * t >= m:
            return trips
        # else return the trimmed list, which is the list without the first t and last t elementss
        return trips[t:-t]
    
    
    def _aggregate_ee_trmean(
        self,
        previous_global: List[np.ndarray],
        client_updates: Dict[str, Tuple[List[np.ndarray], dict]],
    ) -> List[np.ndarray]:

        # fetch the number of clients and check if we have enough clients
        client_ids = list(client_updates.keys())
        n = len(client_ids)
        # if there are less than 3 clients, fallback to FedAvg
        if n < 3:
            logger.info("<3 clients – FedAvg fallback.")
            return self._aggregate_fedavg(previous_global, client_updates)

        # check if we have any new clients
        new_clients = [cid for cid in client_ids if self.client_rounds.get(cid, 0) == 0]
        old_clients = [cid for cid in client_ids if self.client_rounds.get(cid, 0) > 0]

        # exploration vs. exploitation decision, random exploration based on this rounds epsilon
        explore_this_round = (random.random() < self.epsilon)
        top = []
        logger.info(f"Epsilon this round: {self.epsilon}")

        if explore_this_round:
            # exploration => all old clients
            top = old_clients
            logger.info("Exploration => all old clients chosen.")
        else:
            # exploit, by picking top-scoring among old
            avgs = {cid: self._avg_score(cid) for cid in old_clients}

            # log the average scores of old clients
            for cid in old_clients:
                old_score = self.client_scores.get(cid, 0.0)
                logger.info(f"{cid} with avg_score={avgs[cid]:.2f}, raw_score={old_score:.1f}")

            # if we have old clients with scores, pick the top ones
            if avgs:
                # get the maximum average score among old clients
                leader    = max(avgs.values())
                # calculate the current alpha value
                alpha_now = self._current_alpha()
                logger.info(f"Current alpha: {alpha_now} ")
                # calculate the threshold for being included in the top clients
                threshold = alpha_now * leader
                # top clients are those with an average score above the threshold
                top = [cid for cid in old_clients if avgs[cid] >= threshold]
                # if less half of the clients, include the top half, reliying on 50 % should be honest
                if len(top) < n // 2:
                   half = n // 2
                    # sort old clients by score (desc) and extend `top` up to half of the clientss
                   extra = sorted(old_clients, key=avgs.get, reverse=True)[:half]
                   top   = list(dict.fromkeys(top + extra))   
            logger.info(f"Exploitation => chosen old clients: {top}.")

        chosen_old = top
        logger.info( f"EE-TrMean exploitation => chosen old clients: {chosen_old}, plus {len(new_clients)} new clients forced." )
        
        # exploit set is always new clients + old clients, new clients are always explored
        chosen_clients = sorted(list(set(new_clients).union(chosen_old)))
        logger.info(f"EE-TrMean aggregator round {self.round}: chosen clients = {chosen_clients}")
        
        # add the used clients to the used clients per round, for average scoring calc.
        for cid in chosen_clients:
            self.client_rounds[cid] = self.client_rounds.get(cid, 0) + 1
        
	    # weights from metadata, default to 1 if not present, iterate over client_updates
        weights = {
            cid: float(meta.get("num_examples", 1))
            for cid, (_, meta) in client_updates.items()
        }

        # Build stacks for chosen clients only
        chosen_stacks = [
            np.stack([client_updates[cid][0][l] for cid in chosen_clients], axis=0)
            for l in range(len(previous_global))
        ]

        # initialize a set to track used clients and a list for the new global model
        used_clients: set[str] = set()
        new_global: List[np.ndarray] = []

        # coordinate-wise trimmed mean on the chosen stacks, ie chosen clients
        for layer_idx, stacked in enumerate(chosen_stacks):
            # get the shape of the stacked array, and the number of chosen clients
            shape = stacked.shape[1:]
            num_chosen = stacked.shape[0]
            # reshape the stacked array to a 2D array, where each row is a client and each column is a coordinate
            flat = stacked.reshape(num_chosen, -1)
            # create an output array to store the aggregated values for this layer
            out_flat = np.empty(flat.shape[1], dtype=flat.dtype)

            # iterate over each coordinate (column) in the flattened array
            for col in range(flat.shape[1]):
                # build array of (value, cid, weight)
                triple_list = []
                # iterate over the chosen clients and their corresponding values in the flattened array, so row for row of each client
                for i, cid in enumerate(chosen_clients):
                    # get the param value for this coordinate from the flattened array
                    val = flat[i, col]
                    # get the weight, ie number of examples from metadata
                    w_  = weights[cid]
                    # create a tuple of (value, client id, weight) and append it to the triple list
                    triple_list.append((val, cid, w_))

                # sort & trim, the chosen clients by their values
                triple_list.sort(key=lambda x: x[0])
                survivors = self._trim(triple_list)

                # Weighted average on the survivors
                # get the total weight of the survivors, which is the sum of the num examples
                total_w = sum(t[2] for t in survivors)
                # edge case: if no number of examples, use mean of trimmed values
                if total_w == 0:
                    out_val = np.mean([t[0] for t in survivors]) if survivors else 0.0
                # otherwise calculate the weighted average
                else:
                    # calculate the weighted average of the values in the survivors
                    out_val = sum(t[0] * t[2] for t in survivors) / total_w
                # append the output value to the output flat array
                out_flat[col] = out_val

                # coordinate-level scoring
                for _, scid, _ in survivors:
                    used_clients.add(scid)
                    self.client_scores[scid] = self.client_scores.get(scid, 0.0) + 1.0

            # reshape the output flat array to the original layer shape and append it to the new global model
            new_global.append(out_flat.reshape(shape))

        # record usage
        self.used_clients_per_round[self.round] = sorted(used_clients)

        # decay epsilon
        self._decay_epsilon()

        # If near session end, log usage
        if self.round % self.session_length == 0 or self.round % self.session_length == 0:
            last = sorted(self.used_clients_per_round.keys())[-self.session_length:]
            for r in last:
                logger.info(f"Used clients Round {r}: {self.used_clients_per_round[r]}")   

        return new_global
    
