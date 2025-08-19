Here’s your explanation rewritten in **GitHub README.md format**. It uses Markdown headings, lists, and code blocks so it looks clean and professional when rendered on GitHub. You can copy this directly into your repo’s `README.md` file.

---

# Aurora Serverless Cost Advisor

This AWS Lambda function analyzes an Aurora cluster’s historical workload metrics and recommends whether **Aurora Serverless v2** or a **Provisioned instance** would be cheaper for your usage.

It compares costs across multiple time windows, considers **Standard vs I/O-Optimized storage**, and returns a structured JSON recommendation.

---

## 🚀 Usage

Provide input JSON in the form:

```json
{"cluster_id": "your-aurora-cluster-id"}
```

❌ Any other format will be rejected.

The Lambda will return structured JSON with cost comparisons and recommendations.

---

## 📊 Metrics Used

The function queries **Amazon CloudWatch** and **Performance Insights** (if enabled):

* **DBLoad / DBLoadCPU** – preferred measure of database load (from PI).
* **ServerlessDatabaseCapacity** – actual Aurora Serverless v2 ACU usage.
* **CPUUtilization** – fallback if DBLoad not available.
* **DatabaseConnections** – workload shape indicator.
* **ReadIOPS / WriteIOPS** – used to estimate I/O request costs.

> If **Performance Insights is not enabled**, the function automatically falls back to CPU metrics.

---

## ⏱ Time Windows

The workload is analyzed over three time windows:

* **2 weeks**
* **1 month**
* **2 months**

Each ends **5 minutes before “now”** to avoid incomplete metrics.

---

## ⚙️ How It Works

1. **Identify Cluster**
   Looks up cluster details: mode (Serverless v2 vs Provisioned), members, and storage type.

2. **Select Storage Modes**
   Evaluates both **Standard** and **I/O-Optimized** storage (capacity cost is the same; only compute and per-I/O differ).

3. **Fetch Metrics**
   Collects workload data from CloudWatch/PI for each window.

4. **Summarize Workload**
   Calculates p50, p95, and p99 values for DBLoad, CPU, and ACU.

5. **Estimate ACUs**

   * Serverless: uses **actual ACU** values.
   * Provisioned: estimates required ACU from DBLoad (or CPU if DBLoad missing).

6. **Estimate I/O Requests**
   Calculates I/O costs (Standard storage only).

7. **Compute Costs**

   * **Serverless**: ACU-hours × ACU price + I/O (if Standard).
   * **Provisioned**: instance hourly price × hours × instance count + I/O (if Standard).

   Results normalized to **monthly equivalent (730 hours)**.

8. **Pick the Best Provisioned Option**
   Chooses the cheapest instance class that fits your workload with safety margin.

---

## 📦 Output

For each time window and storage mode:

* Workload stats (DBLoad, CPU, ACU percentiles)
* **Serverless monthly equivalent cost**
* **Best provisioned instance** (class + monthly cost)
* **Current provisioned instance** result (if applicable)
* Summary with:

  * Overall cheapest option
  * Cheapest per storage mode
  * Top 3 cheapest options overall

Returned as structured JSON, e.g.:

```json
{
  "2w": {
    "serverless_monthly": 420.0,
    "provisioned_best": {
      "instance_class": "db.r6g.xlarge",
      "monthly_cost": 365.0
    }
  },
  "summary": {
    "overall_cheapest": "provisioned",
    "best_instance": "db.r6g.xlarge",
    "savings": 55.0
  }
}
```

---

## 🔧 Configurable Constants

Edit these at the top of the file:

* `ACU_PRICE_PER_HOUR` – hourly cost for ACU
* `INSTANCE_PRICES` – dictionary of instance classes and hourly costs
* `IO_PRICE_PER_MILLION` – I/O cost for Standard storage
* Multipliers for **I/O-Optimized compute**
* **Safety headroom** factor for sizing (default: 70%)
* Time windows and sampling periods

---

## ❓ Why Not CPU for Serverless?

* **Aurora Serverless v2 publishes ACU usage directly** – this is the most accurate way to size cost.
* **CPU% is only a proxy**, so it’s used only when DBLoad or ACU isn’t available.

This ensures recommendations are accurate for serverless workloads.

---

## ✅ Guardrails

* Strict input validation (`{"cluster_id": "X"}` only).
* Automatic metric fallbacks if some metrics are missing.
* Windows end 5 minutes before “now” to avoid partial CloudWatch data.

---

## 🌐 The Big Picture

This Lambda is your **Aurora cost advisor**.
It looks at your real workload, simulates both **Serverless v2** and **Provisioned** paths, and highlights the **cheapest safe option**—across multiple time windows so you can make confident decisions.

Got it 👍 — here’s how you can integrate the **CPU fallback explanation** directly into your GitHub README so it flows naturally with your existing sections.



---

## ⚡ CPU Fallback Explained

Aurora recommendations are ideally based on **DBLoad** (from Performance Insights) or **Serverless ACU metrics**. These map directly to Aurora capacity and give the most accurate sizing guidance.

However, in many clusters **Performance Insights is not enabled**, and **provisioned clusters don’t report ACU usage**. In those cases, the function falls back to using **CPUUtilization** from CloudWatch.

### Why CPU Fallback Exists

* **Always available**: Every RDS/Aurora cluster publishes `CPUUtilization`.
* **Reasonable proxy**: If CPU p95 is consistently high, the instance is under pressure.
* **Prevents gaps**: Ensures the tool can still provide recommendations when DBLoad is missing.

### Limitations

* CPU% does not always reflect Aurora’s internal scaling needs.
* Workloads that are **I/O-heavy** or **connection-heavy** may require more capacity even if CPU% looks low.
* The mapping requires assumptions (e.g., “assume 4 vCPUs”), which can under- or over-estimate actual capacity.

### When It’s “Good Enough”

* When you need a **ballpark estimate** of provisioned vs serverless costs.
* For **CPU-bound workloads** where CPU usage directly correlates with load.
* When Performance Insights is disabled and you cannot enable it.

### When It May Mislead

* For **I/O-heavy workloads** (lots of queries, low CPU).
* **Spiky workloads** that look fine on average but have bursts.
* When comparing **very different instance classes** with different vCPU counts.

### Takeaway

✅ CPU fallback ensures the function *always works*, but it’s a **best-effort estimate**.
For production decisions, enabling **Performance Insights** is strongly recommended so DBLoad can be used instead.


