import os
import logging

logger = logging.getLogger("fedn")
logging.basicConfig(level=logging.INFO)


def load_env_params():
    # Read environment variables from Kubernetes pod
    client_index_str = os.environ.get("CLIENT_ID", "none")
    logger.info(f"client_index_str: {client_index_str}")
    env_malicious_flag = os.environ.get("MALICIOUS", "false").strip().lower()
    env_malicious = (env_malicious_flag == "true")
    logger.info(f"env_malicious_flag={env_malicious_flag}, env_malicious={env_malicious}, client_index={client_index_str}" )
    attack = os.environ.get("ATTACK_TYPE", "none")
    inflation_factor = int(os.environ.get("INFLATION_FACTOR", "1"))
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    epochs = int(os.environ.get("EPOCHS", "5"))
    lr = float(os.environ.get("LR", "0.01"))
    malicious = env_malicious

    return client_index_str, malicious, attack, inflation_factor, batch_size, epochs, lr