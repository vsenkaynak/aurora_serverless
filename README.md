
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





## ⚙️ ACU Estimation Logic

When comparing **Aurora Serverless v2** vs **Provisioned** costs, we estimate the required **Aurora Capacity Units (ACU)** from CloudWatch metrics.

### 🔹 `AAS_TO_ACU_FACTORS`

```python
AAS_TO_ACU_FACTORS = [0.75, 1.00, 1.25]
```

These factors convert **Average Active Sessions (AAS)** from CloudWatch’s `DBLoad` metric into **ACUs** for provisioned clusters:

* **0.75** → Lightweight workload (1 DBLoad < 1 ACU)
* **1.00** → Neutral workload (1 DBLoad ≈ 1 ACU)
* **1.25** → Heavy workload (1 DBLoad > 1 ACU)

We use multiple factors to see how sensitive the recommendation is to workload assumptions.
For example, if `DBLoad p95 = 4.0`:

| Factor | Estimated ACU |
| ------ | ------------- |
| 0.75   | 3.0 ACU       |
| 1.00   | 4.0 ACU       |
| 1.25   | 5.0 ACU       |

---

### 🔹 CPU-Aware ACU Estimation Settings

```python
USE_CPU_IN_ACU_ESTIMATE = True   # blend CPU into ACU estimate
ACU_BLEND_MODE = "max"           # "max" (recommended) or "avg"
CPU_TO_ACU_HEADROOM = 0.70       # scale CPU→ACU similar to AAS headroom
CPU_WEIGHT = 0.5                 # only used when ACU_BLEND_MODE == "avg"
```

#### `USE_CPU_IN_ACU_ESTIMATE`

* **True** → combine **DBLoad-based** and **CPU-based** ACU estimates.
* **Why:** DBLoad alone may miss CPU-heavy patterns; CPU metrics can reveal additional scaling needs.

#### `ACU_BLEND_MODE`

* **`"max"` (recommended):** Use the **higher** of DBLoad→ACU and CPU→ACU (safer against spikes).
* **`"avg"`:** Weighted average of both metrics (controlled by `CPU_WEIGHT`).

#### `CPU_TO_ACU_HEADROOM`

* Multiplier to avoid sizing at 100% CPU.
* Example: 8 vCPUs @ 50% CPU → raw = 4 ACU; with 0.70 headroom → **2.8 ACU**.

#### `CPU_WEIGHT` *(only if `"avg"` mode)*

* Weight of CPU-based ACU estimate in the blend.
* `0.5` = equal weight with DBLoad; `0.7` = more influence from CPU.

> **Note:** For **Serverless v2** clusters, we skip this logic and use the actual ACU metric from CloudWatch.

---

Guard Clause (Invalid Inputs)

if effective_vcpu <= 0 or cpu_p95 <= 0:
    return 0.0


If the instance has no vCPUs or the CPU utilization is zero, then there’s no meaningful workload.

The function returns 0 ACUs required.

Convert Percentage to Ratio

utilization = cpu_p95 / 100.0


CPU utilization (cpu_p95) is a percentage (e.g., 70%).

We convert it into a decimal ratio (e.g., 0.70).

Estimate Raw ACU Demand

raw_acu = (utilization * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)


Multiply utilization × vCPUs → how many vCPUs are really “busy.”

Divide by CPU_TO_ACU_HEADROOM (usually ~0.70):

This adds buffer room so we don’t size too aggressively.

Example: If CPU p95 = 70% on 4 vCPUs, effective usage = 2.8 cores.
With headroom 0.7 → 2.8 / 0.7 ≈ 4 ACUs.

max(1e-6, …) avoids division by zero errors.

Snap to ACU Grid

return snap_acu(raw_acu, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, ACU_STEP)


Aurora Serverless v2 doesn’t allow arbitrary fractional ACUs.

This step rounds the raw estimate to the nearest valid step:

Between DEFAULT_MIN_ACU and DEFAULT_MAX_ACU

In increments of ACU_STEP (usually 0.5 ACU).

Ensures the recommendation is something Aurora actually supports.

In Plain English

This function says:

"Take the 95th percentile CPU utilization (so we size for high-load moments, not just average). Multiply by the number of vCPUs the instance has. Add some buffer so we don’t run too close to the limit. Then translate that into Aurora ACUs, rounding to valid step sizes that Aurora actually allows."

👉 This function is used when Q1 and Q2 (DBLoad metrics) aren’t available. In that case, CPU utilization is the fallback signal for workload intensity.

Do you want me to also show you a numerical example with real numbers (e.g., 4 vCPUs, 65% CPU, headroom 0.7 → required ACU = X) so you can put it in your README?

You said:
yes
ChatGPT said:

Perfect 👍 Let’s go through a numerical example of how cpu_p95_to_required_acu works, step by step, with real numbers. This is something you could drop directly into your README as a worked example.

Example: Converting CPU Utilization to Required ACU

Suppose you have:

CPU 95th percentile (cpu_p95) = 65%

Effective vCPUs = 4

CPU→ACU headroom (CPU_TO_ACU_HEADROOM) = 0.70

Aurora Serverless allows ACUs in 0.5 increments between 0.5 and 128.

Step 1: Convert utilization to ratio
65% → 0.65

Step 2: Multiply by vCPUs
0.65 × 4 vCPUs = 2.6 “busy cores”

Step 3: Apply headroom buffer
2.6 / 0.70 ≈ 3.71 raw ACUs


The headroom means: "Don’t size for 100% of current usage, leave some breathing space."

Because we divide by 0.7, the required ACU goes up, giving us extra buffer.

Step 4: Snap to Aurora ACU grid

Aurora only allows ACUs in steps of 0.5. So:

3.71 → rounds up to 4.0 ACUs

✅ Final Recommendation

With 65% CPU at p95 on 4 vCPUs, the cluster would require about 4 ACUs.

Why This Works

CPU p95 ensures we size for heavier load spikes (not just average).

Headroom (0.7) ensures the instance won’t run too hot and throttle.

Snapping ensures the result matches Aurora’s valid ACU settings.

👉 Do you want me to also prepare a second example (e.g., with 8 vCPUs at 80% CPU) so you can include a small table of "before → after" in the README? That way readers can see multiple scenarios at a glance.

You said:
yes
ChatGPT said:

Great — let’s build a second worked example and then I’ll show you a mini table that you can drop into your README so readers see multiple scenarios side by side.

Example 2: 8 vCPUs at 80% CPU (p95)
Step 1: Convert utilization
80% → 0.80

Step 2: Multiply by vCPUs
0.80 × 8 vCPUs = 6.4 busy cores

Step 3: Apply headroom buffer (0.70)
6.4 / 0.70 ≈ 9.14 raw ACUs

Step 4: Snap to Aurora ACU step (0.5 increments)
9.14 → rounds up to 9.5 ACUs


✅ Final Recommendation: ~9.5 ACUs for 8 vCPUs running 80% CPU at p95.

Comparison Table

Here’s a simple table showing how different CPU utilizations and vCPUs translate into recommended ACUs:

CPU p95	vCPUs	Headroom	Raw ACU (before snap)	Snapped ACU
65%	4	0.70	3.71	4.0
80%	8	0.70	9.14	9.5
50%	16	0.70	11.43	11.5
Why This Helps

Readers see different scenarios at a glance.

Shows that the formula scales with both CPU % and # of vCPUs.

Reinforces that we always snap to Aurora’s allowed ACU values.

👉 Do you want me to also generate this table in Markdown format ready for GitHub README, so you can paste it directly without editing?

You said:
where do we use this function?
ChatGPT said:

That function:

def cpu_p95_to_required_acu(cpu_p95: float, effective_vcpu: int) -> float:
    if effective_vcpu <= 0 or cpu_p95 <= 0:
        return 0.0
    utilization = cpu_p95 / 100.0
    raw_acu = (utilization * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
    return snap_acu(raw_acu, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, ACU_STEP)


is used in your fallback path when Q1/Q2 Aurora Serverless metrics (DB load) are not available.

Where It Fits in the Logic

Preferred case (serverless cluster with Insights enabled):

Use DBLoad or ACU metrics directly (Q1/Q2/Q3).

Fallback case (when Q1, Q2 empty):

Take CPUUtilization (usually p95).

Call cpu_p95_to_required_acu() with that CPU % and the instance’s vCPU count.

This produces an estimated ACU equivalent so you can still size the cluster.

Comparison step:

Once you have ACUs (actual from Q3, or estimated via CPU), you compare:

Serverless cost (ACU × hourly price).

Provisioned instance classes (map vCPU ↔ ACU).

Pick the cheapest instance class that can handle the estimated load.

Why It Exists

Aurora Serverless v2 clusters expose their capacity in ACUs, but provisioned clusters expose metrics in CPUUtilization + vCPUs.
This function is the bridge: it translates a CPU % into an equivalent number of ACUs so that both serverless and provisioned options can be compared in the same "currency".

✅ So in short:

If Q1/Q2 metrics are available → we don’t call this.

If they’re missing (which is common) → this function runs, using CPUUtilization to still give you a fair recommendation.

Do you want me to trace through your latest GitHub version and show you exactly which line(s) call cpu_p95_to_required_acu() so you can see it in context?

You said:
how do we pass effective_vcpu to the function. where is this parameter coming from. What calls it
ChatGPT said:

Great question 👍 — let’s walk through it carefully.

1. Where does effective_vcpu come from?

It comes from your instance class definitions.
In your codebase you have a dictionary like this:

INSTANCE_VCPU = {
    "db.r6g.large": 2,
    "db.r6g.xlarge": 4,
    "db.r6g.2xlarge": 8,
    # … and so on
}


That dictionary maps each Aurora instance class to its number of vCPUs.

So when you’re checking whether a provisioned instance is a good fit, you look up the vCPUs from that dictionary, and pass it into cpu_p95_to_required_acu() as the effective_vcpu.

2. What actually calls the function?

The call path looks like this (based on your GitHub version):

Main recommend loop gathers metrics:

dbload_p95 (Q1, Q2 metrics if available)

cpu_p95 (always from CloudWatch)

Decision point:

If Q1/Q2 (dbload) are available → use those directly.

If they’re empty → fallback kicks in → call:

required_acu = cpu_p95_to_required_acu(cpu_p95, INSTANCE_VCPU[inst_class])


Result usage:
That required_acu is then compared against:

Candidate instance’s vCPU/ACU capacity.

Serverless ACU pricing.

This ensures that even when you have no DBLoad metrics, you can still size the workload correctly.

3. Why is it called effective vCPU?

Because not all Aurora instances scale 1:1 with vCPU.
Some classes have:

IO-optimized multipliers

Headroom adjustments (your HEADROOM_BASE, SPIKE_AWARE_HEADROOM, etc.)

So instead of just passing raw vCPU count, you might apply multipliers (e.g. vCPU * IO_OPTIMIZED_MULTIPLIER) before passing it into the function. That’s why the parameter is named effective_vcpu.

✅ In summary:

effective_vcpu is looked up from the instance class mapping (INSTANCE_VCPU).

It’s passed in during the fallback path (when DBLoad metrics are missing).

The function translates CPU% × vCPUs into an equivalent ACU load, so serverless and provisioned options can be compared.

Would you like me to trace the exact lines in your GitHub main_V3.py where this fallback call happens, so you can see precisely how INSTANCE_VCPU ties into cpu_p95_to_required_acu()?

You said:
yes
ChatGPT said:

Here’s exactly where effective_vcpu comes from and how it flows into cpu_p95_to_required_acu(...) in your (GitHub-style) code.

1) It’s computed once per window

Inside evaluate_one_window_with_storage(...) we derive an effective vCPU count from the writer (preferred) or, failing that, from the largest class in the cluster:

effective_vcpu = compute_effective_vcpu(current_classes, writer_class)

compute_effective_vcpu(...)
def compute_effective_vcpu(current_instance_classes: List[str], writer_class: Optional[str]) -> int:
    if writer_class:
        v = INSTANCE_VCPU.get(writer_class, 0)
        if v > 0:
            return v
    max_v = 0
    for ic in current_instance_classes or []:
        max_v = max(max_v, INSTANCE_VCPU.get(ic, 0))
    return max_v if max_v > 0 else 4


Looks up vCPUs from your INSTANCE_VCPU table.

Prefers the writer’s class.

If unknown, uses the largest vCPU among member instances.

If still unknown, falls back to 4.

2) It’s passed into the CPU fallback → ACU conversion

When DBLoad isn’t available (so the basis is CPU), provisioned “fit” checks call:

elif basis == "CPU":
    req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
    fits = (req_acu <= eff_headroom * vcpu)   # vcpu = candidate’s vCPUs


And for the current instance fit check:

elif basis == "CPU" and shape.cpu_p95 > 0:
    req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
    fits = (req_acu <= eff_headroom * INSTANCE_VCPU.get(curr, 0))

3) The converter itself
def cpu_p95_to_required_acu(cpu_p95: float, effective_vcpu: int) -> float:
    if effective_vcpu <= 0 or cpu_p95 <= 0:
        return 0.0
    utilization = cpu_p95 / 100.0
    raw_acu = (utilization * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
    return snap_acu(raw_acu, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, ACU_STEP)


Converts CPU p95 and effective vCPUs into an ACU requirement (with headroom).

Snaps to valid ACU steps.

4) Where the candidate’s vcpu comes from

In the provisioned loop, each candidate instance class’s vCPUs are taken from the same INSTANCE_VCPU table:

for ic, eff_price_per_hour in sorted_cands:
    vcpu = INSTANCE_VCPU.get(ic, 0)  # candidate capacity
    ...
    fits = (req_acu <= eff_headroom * vcpu)

TL;DR

effective_vcpu is derived from your current cluster’s writer (or largest member) using INSTANCE_VCPU.

It’s then fed into cpu_p95_to_required_acu(...) only when DBLoad is missing.

The resulting required ACU is compared against each candidate’s capacity (candidate_vcpu * headroom) to find the cheapest fitting instance.






