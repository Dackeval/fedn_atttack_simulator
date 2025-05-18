# FEDn Attack Simulator

*A attack and defense simulator for federated learning experiments on the MNIST dataset with [FEDn](https://github.com/scaleoutsystems/fedn). Works locally on Docker + Kubernetes or against any MinIO/S3 store and K8s cluster.*

---

## 1. Quick Start (in 60 seconds 🏃‍♀️)

```bash
# 1 · Clone and enter the repo
git clone [https://github.com/<your‑org>/<repo>.git](https://github.com/Dackeval/fedn_atttack_simulator.git)
cd fedn_atttack_simulator

# 2 · Create an isolated Python env
python -m sim .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 · Install Python dependencies
pip install -r requirements.txt

# 4 · Split MNIST into 10 IID balanced partitions
python tools/split_mnist.py \
  --n_splits 10 \
  --iid iid \
  --balanced balanced

# 6 · Before starting the simulator (see §5 for the full workflow)
python test_simulation.py
```

> **Before the first run**, ensure you have:
>
> * A MinIO/S3 bucket (or mount a local volume – see §3)
> * The client **package** and **seed model** uploaded to FEDn Studio
> * `config.yaml` pointing to the correct endpoints / tokens

---

## 2. Prerequisites

| Tool     | Tested version | Install                                                                            |
| -------- | -------------: | ---------------------------------------------------------------------------------- |
| Python   |     **3.12.7** | [https://www.python.org/downloads/](https://www.python.org/downloads/)             |
| PyTorch  |      **2.2.2** | `pip install torch==2.2.2`                                                         |
| FEDn CLI |     **latest** | `pip install fedn`                                                                 |
| Helm     |    **v3.17.1** | [https://helm.sh/docs/intro/install/](https://helm.sh/docs/intro/install/)         |
| kubectl  |    **v1.31.3** | [https://kubernetes.io/docs/tasks/tools/](https://kubernetes.io/docs/tasks/tools/) |

The simulator *should* work with newer versions, but the above are verified.

---

## 3. Dataset & Partitions

Each client expects its partition at:

```
/app/data/mnist/<IID>_<BALANCED>/clients/<CLIENT_ID>/mnist.pt
```

If the file is missing, `data.py` pulls it from your object store using the environment variables below.

### 3.1 Required environment variables

| Variable           | Example      | Purpose                        |
| ------------------ | ------------ | ------------------------------ |
| `DATA_ENDPOINT`    | `minio:9000` | MinIO/S3 endpoint (host\:port) |
| `DATA_ACCESS_KEY`  | `minioadmin` | Access key                     |
| `DATA_SECRET_KEY`  | `minioadmin` | Secret key                     |
| `DATA_BUCKET_NAME` | `fedn`       | Bucket name                    |
| `CLIENT_ID`        | `1`          | 1‑based client index           |
| `IID`              | `iid`        | `iid` or `noniid`              |
| `BALANCED`         | `balanced`   | `balanced` or `unbalanced`     |

These vars are injected into every **mnist‑sim** pod by the Helm chart (`examples/mnist‑pytorch/chart/values.yaml`).

### 3.2 Object‑store layout

```
fedn (bucket)
└─ mnist
   └─ <iid>_<balanced>
      └─ clients
         ├─ 1
         │  └─ mnist.pt
         ├─ 2
         │  └─ mnist.pt
         └─ …
```

---

## 4. Configuration (`config.yaml`)

```yaml
simulation:
  # FEDn connection
  combiner_ip: "https://<combiner-host>"   # gRPC endpoint (see Studio)
  client_token: ""                         # Client Token (Studio)
  auth_token:   ""                         # Admin Token (Studio)

  # Adversarial setup
  attack_type:      label_flip_basic          # label_flip_basic | grad_boost_basic | little_is_enough |
                                                # artificial_backdoor_05p_center | artificial_backdoor_05p |
                                                # backdoor_35int
  inflation_factor: 2                         # Used only with grad_boost_basic
  defense_type:     Multi-KRUM                # DNC | KRUM | Multi-KRUM | TrMean | FedAvg | EE_DNC | EE_Multi-KRUM

  # Training params
  batch_size:    32
  epochs:        1
  learning_rate: 0.01

  # Clients & data store
  benign_clients:    1
  malicious_clients: 1
  data_endpoint:   s3.eu-north-1.amazonaws.com
  data_access_key: <ACCESS_KEY>
  data_secret_key: <SECRET_KEY>
  data_bucket_name: simulator-mnist-data-bucket

  iid:       iid          # iid | noniid
  balanced:  balanced     # balanced | unbalanced

  pushfetch_or_fetch: fetch   # pushfetch | fetch

  # Late‑joining clients
  late_client_ind:  [1]
  late_client_delay: 5

  # Session length
  rounds: 30
```

### 4.1 Key concepts

| Key                  | Purpose                                                        |
| -------------------- | -------------------------------------------------------------- |
| `attack_type`        | Strategy executed by malicious clients                         |
| `defense_type`       | Aggregation rule to mitigate attacks                           |
| `pushfetch_or_fetch` | `push` – split ➜ upload ➜ download · `fetch` – only download |
| `late_client_*`      | Simulate stragglers joining late                               |

---

## 5. Running a Simulation

1. **Package the client and seed model**

   ```bash
   fedn package create --path client
   fedn run build --path client
   ```

2. **Upload them to FEDn Studio**

   ```bash
   export FEDN_AUTH_TOKEN=<access-token> # available on Studio under clients
   fedn studio login -u <user> -P <password> -H <studio_host>
   fedn project set-context -id <project_id> -H <studio_host>
   fedn model set-active -f model.npz -H <studio_host>
   ```

3. **Launch the simulator**

   ```bash
   python test_simulation.py
   ```

   The script prompts you to reuse/create data partitions and asks for a **session name** (e.g. `mnist-iid-balanced-10c-2025-05-18`).

The simulator spins up the `mnist‑sim` client pods. 

---

## 6. Monitoring & Logs

* **FEDn Studio** – model lineage, metrics, TensorBoard
* `kubectl get pods` –A to see Kubernetes client pods
---

## 7. Tear‑down

```bash
# Remove the simulated client deployment
helm uninstall mnist-sim

```

---

## 8. Contributing

PRs are welcome! 
