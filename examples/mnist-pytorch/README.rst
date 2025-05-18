Fedn MNIST Simulator

A lightweight reference implementation for running end‑to‑end federated learning experiments on the MNIST dataset using FEDn.

1. Quick Start (in 60 seconds 🏃‍♀️)

# Clone and enter the repo
git clone https://github.com/<your-org>/<repo>.git
cd <repo>

# Create an isolated Python env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Split the MNIST data set into 10 IID balanced partitions
python tools/split_mnist.py \
  --n_splits 10 \
  --iid iid \
  --balanced balanced

# Start the simulator
(Before running the simulator, make sure you have a MinIO/S3 bucket set up, the environment variables below configured and package and seed model uploaded to FEDn Studio (see section 5).)

python test_simulation.py

By default the first run stores the partitions on disk only.Add --push to the command above to automatically upload them to your MinIO/S3 bucket (see section 4).

2. Prerequisites

Tool

Tested version

Install

Python

3.12.7

https://www.python.org/downloads/

PyTorch

2.2.2

pip install torch==2.2.2

FEDn CLI

latest

pip install fedn

Helm

v3.17.1

https://helm.sh/docs/intro/install/

kubectl

v1.31.3

https://kubernetes.io/docs/tasks/tools/

⚠️  The simulator should also work with newer versions, but the above are verified.

3. Dataset & Partitions

The simulator expects each client’s MNIST partition to be located either

Locally inside the container at/app/data/mnist/<IID>_<BALANCED>/clients/<CLIENT_ID>/mnist.pt, or

Remotely in your object store under the same key.

On start‑up data.py tries the local path first and—if the file is missing—downloads it from MinIO/S3 using the variables listed below.

Required environment variables

Variable

Example

Description

DATA_ENDPOINT

minio:9000

MinIO/S3 endpoint (host:port)

DATA_ACCESS_KEY

minioadmin

Object‑store access key

DATA_SECRET_KEY

minioadmin

Object‑store secret key

DATA_BUCKET_NAME

fedn

Bucket containing the partitions

CLIENT_ID

1

1‑based index of the simulated client

IID

iid

iid or noniid

BALANCED

balanced

balanced or unbalanced

Tip: These variables are automatically injected into every mnist‑sim client pod by the Helm chart. Edit values.yaml if you need different defaults.

Mounting a local volume (skip the download)

If you prefer not to rely on an object store you can mount a host directory at /app/data:

# values.yaml
volumeMounts:
  - name: mnist-data
    mountPath: /app/data
volumes:
  - name: mnist-data
    hostPath:
      path: /absolute/path/on/host/mnist
      type: Directory

After changing the values file, apply the update:

helm upgrade fedn charts/fedn -f values.yaml

Resulting object‑store layout

fedn (bucket)
└─ mnist
   └─ <iid>_<balanced>
      └─ clients
         ├─ 1
         │  └─ mnist.pt
         ├─ 2
         │  └─ mnist.pt
         └─ …

4. Configuration

All runtime parameters live in config.yaml under the top‑level key simulation:. Below is the current schema together with default values and valid options.

simulation:
  # FEDn connection
  combiner_ip: "https://<combiner-host>"   # gRPC endpoint of the combiner, available on FEDn Studio
  client_token: ""                         # Client Token, available on FEDn Studio 
  auth_token:   ""                         # Admin Token, available on FEDn Studio 

  # ⚔️  Adversarial training setup
  attack_type:    label_flip_basic          # label_flip_basic | grad_boost_basic | little_is_enough |
                                             # artificial_backdoor_05p_center | artificial_backdoor_05p |
                                             # backdoor_35int
  inflation_factor: 2                       # Only used when attack_type == grad_boost_basic
  defense_type:   Multi-KRUM                # DNC | KRUM | Multi-KRUM | TrMean | FedAvg | EE_DNC | EE_Multi-KRUM

  # Training hyper‑parameters
  batch_size:    32
  epochs:        1
  learning_rate: 0.01

  # Clients & data store
  benign_clients:    1                      # Number of honest clients
  malicious_clients: 1                      # Number of Byzantine clients

  data_endpoint:   s3.eu-north-1.amazonaws.com # MinIO/S3 endpoint, where the remote data is stored
  data_access_key: <ACCESS_KEY>
  data_secret_key: <SECRET_KEY>
  data_bucket_name: simulator-mnist-data-bucket

  iid:       iid                            # iid | noniid
  balanced:  balanced                       # balanced | unbalanced

  pushfetch_or_fetch: fetch                 # push | fetch 

  # Late‑joining clients
  late_client_ind:  [1]                     # Client indices (1‑based)
  late_client_delay: 5                      # Delay in FL rounds

  # Session length
  rounds: 30

Key concepts

Key

Purpose

attack_type

Selects the adversarial strategy executed by the malicious clients.

defense_type

Aggregation rule used by the combiner to mitigate attacks.

pushfetch_or_fetch

push – the simulator first splits & uploads the MNIST data before training. fetch – assumes partitions are already present in the bucket and simply downloads them.

late_client_*

Simulate straggler behaviour by having selected clients join after late_client_delay rounds.

Tip: You can override any value via environment variables, e.g. SIMULATION_ROUNDS=50 python test_simulation.py.

5. Running a Simulation

Package the client code and seed model

fedn package create --path client
fedn run build --path client

Upload to FEDn Studio

fedn studio login -u <user> -P <pwd> -H <studio_host>
fedn project set-context -id <project_id> -H <studio_host>
fedn model set-active -f model.npz -H <studio_host>

Launch the simulator

python test_simulation.py

You will be prompted for:

Whether to reuse existing data partitions or create & upload new ones.

A session name (e.g. mnist-iid-balanced-10c-2025-05-18).

FEDn spins up the aggregator, combiner, and client containers automatically.Progress is streamed to your terminal and is also visible in Studio.

6. Monitoring & Logs

FEDn Studio – web UI for model lineage and metrics

kubectl get pods – watch the Kubernetes pods.

7. Tear‑down

# Remove the simulated client pods/deployment (created by the simulator)
helm uninstall mnist-sim

8. Contributing

PRs are welcome! Please run pre-commit run --all-files before pushing.

