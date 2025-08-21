Here’s your text reformatted into a proper **GitHub README.md** structure, with clear headings, code blocks, and lists so it reads well on GitHub:

---

````markdown
# Aurora Serverless v2 Cost Analyzer Lambda

## What Problem This Lambda Solves

You give it a single input:

```json
{"cluster_id": "<your-aurora-db-cluster-identifier>"}
````

The Lambda then:

* Looks back over **3 windows** (≈ 14 days, 30 days, 60 days).
* Pulls **CloudWatch metrics** for that cluster.
* Ensures enough CPU data exists (≥ 50% of expected samples).
* Estimates **serverless v2 compute usage (ACUs)** and equivalent monthly cost.
* Finds the **cheapest provisioned instance class** (with headroom) and computes its monthly cost.
* Reports results for **Standard** storage and **I/O-Optimized** storage.
* Returns **structured JSON only** with numbers you can compare.

---

## Key Ideas (Simple Terms)

* Sizing is based on **p95** (busy-but-usual).
* If **p99 vs p95 is high (spiky workload)** → lower utilization target (more buffer).
* **I/O-Optimized storage**: higher compute price but no I/O charges.
* **Standard storage**: cheaper compute, but you pay per-I/O.
* Windows with **poor CPU coverage** are skipped with diagnostics.

---

## Step-by-Step Flow

### 1. Configuration & Prices

* Time windows:

  * `"14d_60s"` → 14 days @ 60-second period
  * `"30d_120s"` → 30 days @ 120-second period
  * `"60d_300s"` → 60 days @ 300-second period
* Serverless ACU-hour price (+ multiplier if `aurora-iopt1`).
* Per-I/O price for Standard storage (I/O free on `aurora-iopt1`).
* Candidate instance classes (no `.medium`, no old t2/t3):

  * vCPU count
  * hourly price
  * I/O-Optimized multipliers
* All static values (editable in code).

---

### 2. Input & Early Discovery

* Lambda validates event = `{"cluster_id": "..."}`.
* Calls RDS to describe cluster & members.
* Figures out:

  * Engine mode: **provisioned vs serverless-v2**
  * Storage type: **Standard vs I/O-Optimized**
  * Writer instance & class.

---

### 3. Windows & Storage Modes

* For storage: `"standard"`, `"iopt1"`, or both.
* For each window (14d, 30d, 60d):

  * Freeze start/end times (with -5m skew for CloudWatch lag).
  * Run logic with **3 sensitivity factors**: `[0.75, 1.00, 1.25]`.
* Each (storage, window, factor) → one recommendation record.

---

### 4. CloudWatch Metrics Fetched

* **Cluster metrics:**

  * `ServerlessDatabaseCapacity` (ACUs, only on serverless-v2)
  * `DBLoad`, `DBLoadCPU` (if Performance Insights enabled)
* **Writer metrics:**

  * `CPUUtilization`, `DatabaseConnections`
* **All instances:**

  * `ReadIOPS`, `WriteIOPS`, `CPUUtilization`
* Paginates until all points retrieved.

---

### 5. CPU Coverage Check

* Expected points = days × (86400 / period\_seconds).
* Skip window if `< 50% coverage`.
* Return diagnostics:

  * `coverage_ratio`, `expected_points`, `actual_points`
  * `earliest_sample_time`, `db_create_time`, `db_age_days`
  * `skip_reason`

---

### 6. Percentiles & Spikes

* Compute **p50, p95, p99** for DBLoad & CPU.
* Sizing by **p95**.
* Spike detection: if `p99 ≥ 1.3 × p95` → lower headroom (e.g., 0.70 → 0.60).
* Calm workloads can slightly raise headroom (max \~0.80).

---

### 7. Estimating ACUs

Priority:

1. If **Serverless v2 ACU metrics** exist → use directly.
2. Else if **DBLoad available** → convert DBLoad × factor → ACUs.
3. Else **CPU fallback**:

   * `ACU = (CPU% × vCPU) / headroom` (snapped to 0.5).
   * If writer CPU unavailable → use max across members.

* Convert ACU-seconds → ACU-hours.

---

### 8. Estimating I/O

* `total_requests += (ReadIOPS + WriteIOPS) × period_seconds`
* Standard: charged per request.
* I/O-Optimized: requests = free.

---

### 9. Monthly Cost Calculation

Normalize to **730 hours/month** for comparability.

* **Serverless:**

  * Compute = `ACU_hours × ACU_price` (× iopt multiplier if `aurora-iopt1`)
  * I/O = per million requests (Standard only)
  * Monthly equiv = `(compute + io) × (730 / window_hours)`
* **Provisioned:**

  * Compute = instance hourly price (× iopt multiplier if `aurora-iopt1`)
  * I/O = same as above
  * Monthly equiv = `(compute + io) × (730 / window_hours)`

---

### 10. Picking Best Provisioned

* Candidate classes sorted by cost.
* First class that “fits” = recommendation.

**Fit check:**

* If DBLoad p95:
  `DBLoad_p95 ≤ headroom × vCPU(candidate)`
* Else CPU fallback:
  `CPU_p95 → required_ACU ≤ headroom × vCPU(candidate)`

Also evaluates **current instance**:

* Whether it fits
* Monthly equivalent cost

---

### 11. Output Shape

Each `(storage, window, factor)` produces JSON:

```json
{
  "serverless_cost": { "monthly_equivalent": 273.75 },
  "provisioned_best": { "instance_class": "db.r6g.8xlarge", "monthly_equivalent": 310 },
  "provisioned_current": { "instance_class": "db.r6g.4xlarge", "monthly_equivalent": 400 },
  "headroom_used": 0.60,
  "coverage": {
    "expected_points": 17280,
    "actual_points": 6000,
    "coverage_ratio": 0.35,
    "skipped": true,
    "skip_reason": "Insufficient CPU coverage"
  }
}
```

Additionally:

* `cheapest_overall`
* `cheapest_by_storage`
* `top3_overall`

---

## Worked Examples

**CPU → ACU fallback**

* Writer vCPUs = 4
* CPU p95 = 65%
* Headroom = 0.70
* ACU ≈ (0.65 × 4) / 0.70 = 3.71 → rounded to 4.0

If spiky (CPU p99 = 90%) → headroom = 0.60 → stricter fit → larger instance.

---

**Provisioned Fit Example**

* DBLoad p95 = 10
* Headroom = 0.60
* Candidate `db.r6g.xlarge` (4 vCPUs): `10 ≤ 2.4` → No
* Candidate `db.r6g.4xlarge` (16 vCPUs): `10 ≤ 9.6` → No
* Candidate `db.r6g.8xlarge` (32 vCPUs): `10 ≤ 19.2` → **Fits**

---

## Why CPU-Only Coverage Matters

If DB is too new, older windows lack samples.
Skipping avoids **underestimating costs**.

---

## Logging

* Toggle with `DEBUG = False`.
* Output = structured JSON only.

---

## What to Compare

For each `(storage, window, factor)`:

* `serverless_cost.monthly_equivalent` vs `provisioned_best.monthly_equivalent`.
* Compare against **current instance class**.
* Check `coverage_ratio` and prefer windows with strong coverage.


