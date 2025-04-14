# from fedn.common.log_config import logger
# from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

# class ServerFunctions(ServerFunctionsBase):
#     def __init__(self) -> None:
#         self.round = 0  # Keep track of training rounds
#         self.lr = 0.1  # Initial learning rate
#         # logger.warning("DEBUG TEST FEDAVG- init")
#         # logger.info("DEBUG TEST FEDAV- - init")
#         # logger.debug("DEBUG TEST FEDAVG- - init")
#         # logger.error("DEBUG TEST FEDAV- - init")
#         # logger.critical("DEBUG TEST FEDAVG- - init")

#     def client_selection(self, client_ids: List[str]) -> List[str]:
#         # Select up to 10 random clients
#         logger.warning("DEBUG TEST FEDAVG- client_selection")
#         logger.info("DEBUG TEST FEDAV- client_selection")
#         logger.debug("DEBUG TEST FEDAVG- client_selection")
#         logger.error("DEBUG TEST FEDAV- client_selection")
#         logger.critical("DEBUG TEST FEDAVG- client_selection")
#         return random.sample(client_ids, min(len(client_ids), 10))

#     def client_settings(self, global_model: List[np.ndarray]) -> dict:
#         # Adjust the learning rate every 10 rounds
#         if self.round % 10 == 0:
#             self.lr *= 0.1
#         self.round += 1
#         # logger.warning("DEBUG TEST FEDAVG- client_settings")
#         # logger.info("DEBUG TEST FEDAV- client_settings")
#         # logger.debug("DEBUG TEST FEDAVG- client_settings")
#         # logger.error("DEBUG TEST FEDAV- client_settings")
#         # logger.critical("DEBUG TEST FEDAVG- client_settings")

#         return {"learning_rate": self.lr}

#     def aggregate(self, previous_global: List[np.ndarray], client_updates: Dict[str, Tuple[List[np.ndarray], dict]]) -> List[np.ndarray]:
#         # Implement a weighted FedAvg aggregation
#         logger.warning("DEBUG TEST FEDAVG")
#         logger.info("DEBUG TEST FEDAVG")
#         logger.debug("DEBUG TEST FEDAVG")
#         logger.error("DEBUG TEST FEDAVG")
#         logger.critical("DEBUG TEST FEDAVG")

#         weighted_sum = [np.zeros_like(param) for param in previous_global]
#         total_weight = 0
#         for client_id, (client_parameters, metadata) in client_updates.items():
#             num_examples = metadata.get("num_examples", 1)
#             total_weight += num_examples
#             for i, param in enumerate(client_parameters):
#                 weighted_sum[i] += param * num_examples
#         logger.info("Models aggregated")
#         return [param / total_weight for param in weighted_sum]
    
from fedn.common.log_config import logger
from fedn.network.combiner.hooks.allowed_import import Dict, List, ServerFunctionsBase, Tuple, np, random

class ServerFunctions(ServerFunctionsBase):
    def __init__(self) -> None:
        self.round = 0  # Keep track of training rounds
        self.lr = 0.1  # Initial learning rate
        self.used_clients_per_round = {}


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
        return [param / total_weight for param in weighted_sum]
    


