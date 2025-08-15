
# Aurora Cost Advisor (Lambda)

Decides whether an **Aurora cluster** should run **Serverless v2** or **Provisioned** from a **compute + I/O** cost perspective, across multiple lookback windows. Works for clusters using **Standard** storage (`aurora`) and **I/O-Optimized** (`aurora-iopt1`).
**Storage capacity costs are deliberately excluded** (they’re the same between modes).

## Input & Output

* **Input (event):** `{ "db_cluster_id": "<aurora-cluster-id>" }`
* **Output (body):** JSON with:

  * Per-window workload stats and costs (Serverless vs Provisioned)
  * Recommendation per window
  * Advice for current provisioned size (upgrade/downgrade/stay)
  * Post-processed **cheapest overall** (and top picks)

> **Region/pricing/knobs** are static at the top of the code so you can edit them in one place. No Pricing API calls.

---

## What it analyzes

**Time windows**

* 2 weeks @ 60s period
* 1 month @ 120s period
* 2 months @ 300s period

**CloudWatch metrics**

* Cluster: `DBLoad`, `DBLoadCPU`, `ServerlessDatabaseCapacity`
* Writer: `CPUUtilization`, `DatabaseConnections`
* All members: `ReadIOPS`, `WriteIOPS`

**Why these**

* `DBLoad` (Average Active Sessions) captures concurrency pressure.
* `CPUUtilization` captures compute saturation (used for provisioned ACU estimation).
* `ServerlessDatabaseCapacity` is the **actual ACU usage** when on Serverless v2.
* IOPS metrics let us estimate **I/O request count** for Standard storage billing.

---

## Cost model (compute + I/O)

* **Serverless v2**

  * **ACU-hours × ACU price**, with an **I/O-Optimized multiplier** if `aurora-iopt1`.
  * **Standard storage** also adds **I/O request charges**.
* **Provisioned**

  * For each instance class:

    * **Hourly price × hours × instance count** (members in cluster)
    * Apply **I/O-Optimized multiplier** for `aurora-iopt1`
    * **Standard storage** adds **I/O request charges**

> **Excluded:** storage capacity (GB) and any other common costs between modes.

---

## Sizing & estimation logic

### If the cluster is **Serverless v2**

1. **Real ACU**: Integrate `ServerlessDatabaseCapacity` over time (clamped to Min/Max ACU) → **ACU-hours**.
2. **Serverless cost**: ACU-hours × price (+ I/O for Standard; no I/O for iopt1).
3. **“What-if Provisioned” sizing** (for comparison):

   * Use **ACU p95** as steady-state demand.
   * Map `1 ACU ≈ ~1 vCPU` and divide by headroom (e.g., 70%) to determine **required vCPUs**.
   * Pick **provisioned instance classes** with ≥ required vCPUs; compute their costs and select the cheapest that fits.
4. **Recommendation**: Compare **best provisioned** vs **serverless** and choose cheaper.

> **Why we don’t need `CPUUtilization` for serverless:**
> `ServerlessDatabaseCapacity` already represents the capacity Aurora actually delivered (blending CPU/memory/concurrency). It’s a stronger, direct signal than CPU%. We can map **ACU → vCPU** for a clean “what-if provisioned” comparison without relying on CPU% heuristics.

### If the cluster is **Provisioned**

1. **ACU estimation** (per-sample):

   * From **DBLoad × factor** (sensitivity factors: 0.75, 1.0, 1.25)
   * From **CPUUtilization × writer vCPUs / headroom**
   * **Blend** the two series (`max` by default, or weighted average), **snap** to 0.5 ACU steps, and **clamp** to bounds.
2. **Serverless cost**: Use the estimated **ACU-hours** as if the workload were serverless.
3. **Provisioned options**:

   * Instance **fits** if `DBLoad p95 ≤ vCPUs × headroom` (concurrency rule of thumb).
   * Compute cost for all classes, pick **cheapest that fits**.
   * Also compute **current instance** cost.
4. **Recommendation**: Compare **best provisioned** vs **serverless** and choose cheaper.
5. **Resize advice** (if currently provisioned):

   * **Current vs Serverless** (is serverless already cheaper?)
   * **Resize within Provisioned** (upgrade/downgrade/stay) + savings
   * **Best Provisioned vs Serverless** (even after resizing, which is cheaper?)

---

## Storage types handled

* **Standard (`aurora`)**: compute price as-is; **adds I/O request charges**.
* **I/O-Optimized (`aurora-iopt1`)**: **no I/O charges**, but **compute prices are multiplied** (serverless ACU and provisioned per-class) to reflect iopt1 premium.
* Run in `"auto"` mode (use detected storage) or `"both"` to see head-to-head results.

---

## Post-processing (rollup)

After running all (window × factor × storage) scenarios:

* Pick **cheapest overall** configuration.
* Pick **cheapest per storage type**.
* List **top 3 cheapest** entries.
* Return the full details + a human-readable summary.

---

## Assumptions & tuning

* **Static prices**: ACU price, per-instance hourly prices, per-class iopt1 multipliers, I/O price/M request (edit at the top of the file).
* **Headroom**: default 70% (both DBLoad→vCPU fit rule and CPU→ACU conversion).
* **ACU step**: 0.5 ACU; **Min/Max ACU** defaults used unless read from cluster (serverless).
* **Windows/periods**: tune as needed for your data density and cost stability.
* **I/O charges** only applied on **Standard** storage.

---

## Limitations

* Uses **heuristics** for provisioned ACU estimation; results depend on factors/headroom and the time windows analyzed.
* **Prices are sample values**; update them to your region/account.
* Excludes storage **capacity** costs and other **common/shared** costs.

---

## TL;DR decision rule

Here’s a **GitHub README-friendly** version of that explanation, formatted in Markdown.

---

## 📊 Percentile Metrics & ACU Factor Conversion

This section documents how the Aurora cost analysis Lambda processes metrics and converts load values into ACU (Aurora Capacity Unit) estimates.

---

### **1. Percentile Calculations**

We calculate **p50**, **p95**, and **p99** values for three CloudWatch metrics:

```python
dbload_p50 = percentile(dbload, 50.0) if dbload else 0.0
dbload_p95 = percentile(dbload, 95.0) if dbload else 0.0
dbload_p99 = percentile(dbload, 99.0) if dbload else 0.0

cpu_p50    = percentile(cpu, 50.0) if cpu else 0.0
cpu_p95    = percentile(cpu, 95.0) if cpu else 0.0
cpu_p99    = percentile(cpu, 99.0) if cpu else 0.0

connections_p95 = percentile(conn, 95.0) if conn else 0.0
```

#### **Meaning:**

* **p50 (median)** → The middle value; 50% of the time load is below this.
* **p95** → Higher than 95% of samples; used for near-peak sizing.
* **p99** → Higher than 99% of samples; captures rare spikes.

If no data is available, defaults to `0.0`.

---

### **2. `AAS_TO_ACU_FACTORS` Conversion**

Provisioned clusters don’t have direct ACU metrics (ACU is a Serverless-only measure).
We estimate ACUs from the `DBLoad` (Average Active Sessions) metric.

```python
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]
```

#### **Meaning:**

* **0.75** → Lightweight workload (1 DBLoad < 1 ACU)
* **1.00** → Neutral workload (1 DBLoad ≈ 1 ACU)
* **1.25** → Heavy workload (1 DBLoad > 1 ACU)

---

### **3. How They Work Together**

1. Take **p95 DBLoad** (near-peak load).
2. Multiply it by each factor to get **three ACU estimates**.
3. For each ACU estimate:

   * Calculate cost for **Aurora Serverless v2**
   * Calculate cost for **Aurora Provisioned** (all instance classes)
4. Return recommendations for each factor scenario.

---

**Example:**

If `dbload_p95 = 4.0`:

| Factor | ACU Estimate |
| ------ | ------------ |
| 0.75   | 3.0 ACUs     |
| 1.00   | 4.0 ACUs     |
| 1.25   | 5.0 ACUs     |

* If all factors recommend the same option → decision is **robust**.
* If results vary by factor → decision is **sensitive** to workload assumptions.

---

### **Why This Matters**

Including both percentile metrics and ACU estimates:

* Makes recommendations **transparent**
* Helps validate assumptions
* Shows **sensitivity** of results to workload weight

---



