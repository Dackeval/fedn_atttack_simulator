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
        # Trimming Mean aggregation
        trimming_fraction = 0.1
        num_clients = len(client_updates)
        # if there are less than 3 clients, run FedAvg
        if num_clients < 3:  
            logger.info("Not enough clients for TrMean. Need > 2.")
            # for FedAvg aggregation part of TrMean
            weighted_sum = [np.zeros_like(param) for param in previous_global]
            total_weight = 0
            # run fedavg on the trimmed list x_agg
            for client_id, (client_parameters, metadata) in client_updates.items():
                num_examples = metadata.get("num_examples", 1)
                total_weight += num_examples
                for i, param in enumerate(client_parameters):
                    weighted_sum[i] += param * num_examples
            logger.info("Models aggregated")
            return [param / total_weight for param in weighted_sum]
        # else run TrMean
        else:
            # Trimming Mean aggregation
            x_agg = []
            for dimension_i in range(len(previous_global)):
                param_list = []
                for client_id, (client_parameters, metadata) in client_updates.items():
                    param_dim_i = client_parameters[dimension_i] 
                    num_examples = metadata["num_examples"]  
                    param_list.append((param_dim_i, num_examples)) 
                # sort using the first element of the tuple
                sorted_param_list = sorted(param_list, key=lambda x: x[0])  # Sort by parameter value

                trim_size = int( 2 * trimming_fraction * num_clients ) # 2 for upper and lower bounds
                trim_size = max(trim_size, 1) 

                # remove lower bound and upper bound, spceifies start and end of the slice
                trimmed_param_list = sorted_param_list[trim_size:-trim_size]
                
                # run FedAvg on the trimmed list and append to x_agg
                tr_param_array = np.array([param for param, _ in trimmed_param_list])
                tr_num_samples = np.array([num_samples for _, num_samples in trimmed_param_list])
                
                # weighted sum of the parameters
                fed_avg_dim_i = np.sum(tr_param_array * tr_num_samples[:, np.newaxis], axis=0) / np.sum(tr_num_samples)
                x_agg.append(fed_avg_dim_i)
            
            return x_agg


                