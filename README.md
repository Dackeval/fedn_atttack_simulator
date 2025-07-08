# FEDn Attack Simulator

*A attack and defense simulator for federated learning experiments on the MNIST dataset with [FEDn](https://github.com/scaleoutsystems/fedn). Works locally on Docker + Kubernetes or against any MinIO/S3 store and K8s cluster.

Currently the simulator only runs on the Mnist-Pytorch Example. *

---

## 1. Quick Start (in 60 seconds 🏃‍♀️)

```bash
# 1 · Clone and enter the repo
git clone https://github.com/Dackeval/fedn_atttack_simulator.git
cd fedn_atttack_simulator

# 2 · Create an isolated Python env
python -m sim .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3 · Install Python dependencies
pip install -r examples
/mnist-pytorch/requirements.txt

# 4 · Before starting the simulator (see §5 for the full workflow)
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
  combiner_ip: "https://<FEDn project URL>"   # Project URL (see Studio)
  client_token: ""                         # Client Token (Studio)
  auth_token:   ""                         # Admin Token (Studio)* auth_token is Admin Token on FEDn 

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
   fedn studio login -u <user> -P <pwd> -H <studio_host>
   fedn project set-context -id <project_id> -H <studio_host>
   fedn model set-active -f model.npz -H <studio_host>
   ```

3. **Launch the simulator**

   ```bash
   python test_simulation.py
   ```

   The script prompts you to reuse/create data partitions and asks for a **session name** (e.g. `mnist-iid-balanced-10c-2025-05-18`).

The simulator spins up `mnist‑sim` client pods. 

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
## 8. Results (quick overview)

### 8.1 Label-Flipping · IID · balanced
<img width="500" alt="1" src="https://github.com/user-attachments/assets/ef900380-bb40-4432-9be5-13c192848dc2" />
<img width="500" alt="2" src="https://github.com/user-attachments/assets/f2c40f32-8873-4090-b572-6aa862ea230f" />

### 8.2 Label-Flipping · IID · imbalanced
<img width="500" alt="3" src="https://github.com/user-attachments/assets/e9aead81-bf5f-48f5-9449-7247cf70cf2a" />
<img width="500" alt="4" src="https://github.com/user-attachments/assets/ee57dc35-0388-480f-992b-475196693d5f" />

### 8.3 Label-Flipping · non-IID · partially imbalanced
<img src="https://github.com/user-attachments/assets/56b9686e-2f74-4757-b81d-982bb6803694" alt="Accuracy – malicious clients" width="500" />
<img src="https://github.com/user-attachments/assets/052e89f6-7137-4d55-9c50-44d3b67b80ef" alt="Accuracy – benign clients"  width="500" />

### 8.4 Label-Flipping · non-IID · imbalanced
<img width="500" alt="7" src="https://github.com/user-attachments/assets/fd2ed454-85fe-42b8-a1f5-a3fe5f60d658" />
<img width="500" alt="8" src="https://github.com/user-attachments/assets/66b82ae2-8261-4bfe-8c24-9bc47326dbd5" />

### 8.5 Little-is-Enough · IID · imbalanced
<img width="500" alt="9" src="https://github.com/user-attachments/assets/6519c2bb-602f-49ed-8c95-3682d2f3ec03" />
<img width="500" alt="10" src="https://github.com/user-attachments/assets/414c5a7a-f3a9-4c3c-8c5d-9c9f69b71572" />

### 8.6 Little-is-Enough · non-IID · partially imbalanced
<img width="500" alt="11" src="https://github.com/user-attachments/assets/eb994ee6-2eb8-4ca1-91e4-9edde93f1b67" />
<img width="500" alt="12" src="https://github.com/user-attachments/assets/c1f6fc02-9e61-4c0b-a996-833bb82cf2fb" />

### 8.3 Experimental grid  
We ran **180+ simulations** crossing

* **Attacks:** Label-Flipping · Little-Is-Enough  
* **Data regimes:** IID Balanced / IID Imbalanced / non-IID Balanced / non-IID Imbalanced
* **Late join:** Benign *or* Malicious clients injected from round 5
* **5 AGRs:** FedAvg, TrMean, Multi-KRUM, DNC, EE-TrMean

### 8.3 Key take-aways

| Aggregator | Key take-aways |
|------------|-------|
| **Trimmed-Mean (TrMean)** | Mitigates some poisoning but never fully excludes an malicious update. |
| **Multi-KRUM** | Achieves relatively high accuracy. However model performance can drop sharply when the data heterogeneity increases. |
| **Divide-and-Conquer (DNC)** | Achieves relatively high accuracy and prunes malicious updates, though limited by *f = 1* in this study. |
| **EE-TrMean** | Novel adaptive AGR using the TrMean rule with an epsilon greedy algorithm and a alfa ramp. Achieved some of the highest accuracies, malicious exclusion and inclusion of late benign clients. However at times excluded an honest-but-different client. |

---

📄 **Full 12-table grid, plots & methodology** → 


## 9. Contributing

PRs are welcome! 

## 10. Contact

Email: Sigge.dackevall@gmail.com
