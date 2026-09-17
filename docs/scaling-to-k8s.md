# Scaling to production (Kubernetes + Terraform)

This repo runs the self-healing *logic* in one pure-Python process so it clones
and runs in seconds. Here is how each component maps to a real production stack -
the design that the in-process version stands in for.

```
                         +------------------------+
   incoming data ---->   |  serving (FastAPI)     |  <-- model pulled from registry
                         |  on k8s Deployment     |
                         +-----------+------------+
                                     | metrics (latency, predictions)
                                     v
   +-------------+   drift +   +-----------+     +---------------------+
   |  Evidently  |-- accuracy->| Prometheus|---->|  Grafana dashboard  |
   |  drift job  |             +-----+-----+     +---------------------+
   +-------------+                   |
                                     | alert / schedule
                                     v
                          +----------------------+
                          |  controller          |  (this repo's ControllerBrain)
                          |  k8s CronJob         |
                          +----------+-----------+
                            retrain / evaluate / promote
                                     |
                          +----------v-----------+
                          |  model registry      |  MLflow + object storage (S3)
                          |  versions + rollback |
                          +----------------------+
```

| In this repo (pure Python) | In production |
|---|---|
| `ModelRegistry` (in memory) | MLflow Model Registry + S3/GCS artifact store |
| `assess_health` drift + accuracy | Evidently drift job + Prometheus metrics |
| `ControllerBrain` policy | a Kubernetes CronJob (or Argo Workflow) running the same policy |
| `retrain` | a training Job on the cluster (GPU node pool if needed) |
| `promote` guardrail | a canary / blue-green rolling update, gated on the eval metric |
| `rollback` | `kubectl rollout undo` / registry stage demotion |
| serving (`model.accuracy`) | FastAPI behind a k8s Service + HPA |
| `decisions.jsonl` ledger | structured logs to Loki / an audit table |
| environment | provisioned by Terraform (cluster, node pools, registry, IAM) |

The controller policy, the guardrail (never promote a worse model), the
drift-vs-current-baseline measurement, and the audit ledger are identical in both
worlds - only the substrate changes.
