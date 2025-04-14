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
        # Implement a weighted FedAvg aggregation
        weighted_sum = [np.zeros_like(param) for param in previous_global]
        total_weight = 0
        for client_id, (client_parameters, metadata) in client_updates.items():
            num_examples = metadata.get("num_examples", 1)
            total_weight += num_examples
            for i, param in enumerate(client_parameters):
                weighted_sum[i] += param * num_examples
        logger.info("Models aggregated")
        return [param / total_weight for param in weighted_sum]
    


# -------------------------------------------------------------------
# 2) Now the IRIS-based "compile_model" and parameter extraction
# -------------------------------------------------------------------
from sklearn.linear_model import SGDClassifier
from sklearn import datasets

def compile_model():
    """
    Compile an SGDClassifier for IRIS. We'll do a short partial fit
    so it has shape (3,4) for coef_ if 3 classes.
    """
    iris = datasets.load_iris()
    X, y = iris.data, iris.target  # X shape (150,4), y in {0,1,2}

    model = SGDClassifier(
        warm_start=True,
        loss='log_loss',
        max_iter=20,
        learning_rate='invscaling',
        eta0=0.001,
        random_state=42
    )
    # We'll train once to define the shape of coef_ and intercept_
    # (In a real FL scenario, each client might partial_fit.)
    model.fit(X, y)

    return model

def get_sklearn_parameters(model):
    """ Return list-of-arrays: [coef, intercept]. """
    # e.g. model.coef_.shape => (3,4) for IRIS, intercept_ => (3,)
    # We'll store them as two arrays. This is the format
    # your aggregator expects: a list of arrays, i.e. [coef, intercept].
    return [model.coef_, model.intercept_]

def set_sklearn_parameters(model, parameters):
    """ Set model's coef_ and intercept_ from aggregator output. """
    # parameters = [new_coef, new_intercept]
    model.coef_ = parameters[0]
    model.intercept_ = parameters[1]


# -------------------------------------------------------------------
# 3) Test harness for IRIS aggregator
# -------------------------------------------------------------------
def simulate_trmean_aggregator_iris():
    # 1) "previous_global" from a compiled IRIS model
    seed_model = compile_model()
    previous_global = get_sklearn_parameters(seed_model)
    # previous_global is now [coef (shape= (3,4)), intercept (shape= (3,))]

    # 2) Create 4 dummy clients with slightly perturbed parameters
    client_updates = {}
    num_clients = 4
    for client_idx in range(num_clients):
        # Copy the seed model's parameters
        perturbed = []
        for layer in previous_global:
            noise = 0.01 * np.random.randn(*layer.shape)
            perturbed.append(layer + noise)

        metadata = {"num_examples": random.randint(5, 100)}
        client_updates[f"client_{client_idx}"] = (perturbed, metadata)

    # 3) Instantiate aggregator
    aggregator = ServerFunctions()

    # 4) Call aggregator
    new_global = aggregator.aggregate(previous_global, client_updates)

    # 5) Print results
    print("=== IRIS Aggregation Complete ===")
    # new_global is [new_coef, new_intercept]
    for i, layer in enumerate(new_global):
        print(f"Layer {i} shape: {layer.shape}")
        # Check for NaNs
        if np.isnan(layer).any():
            print(f"  -> Layer {i} has NaNs!")
        else:
            print(f"  -> Layer {i} OK (mean={layer.mean():.4f}, std={layer.std():.4f})")

    # 6) If you want to see how it looks in an actual sklearn model:
    updated_model = compile_model()  # re-compile a fresh model
    set_sklearn_parameters(updated_model, new_global)
    # Now updated_model.coef_, updated_model.intercept_ have the aggregator result
    print("New aggregator model shapes:", updated_model.coef_.shape, updated_model.intercept_.shape)


if __name__ == "__main__":
    simulate_trmean_aggregator_iris()