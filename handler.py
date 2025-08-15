# lambda_function.py
"""
Aurora Serverless v2 vs Provisioned (compute-only) – Lambda
- Event MUST include only: {"db_cluster_id": "<your-cluster-id>"}
- Everything else is static in this file (edit the CONFIG section).
- Compares compute-only costs (no storage/backups/common costs).
- Three windows: 14d@60s, 30d@120s, 60d@300s
- Sensitivity factors for DBLoad(AAS) → ACU mapping.
- Uses static pricing (no Pricing API).

IAM needed: rds:DescribeDBClusters, rds:DescribeDBInstances, cloudwatch:GetMetricData
"""

import json
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import boto3

# =========================
# ====== CONFIG (edit) =====
# =========================

REGION = "us-east-1"  # CloudWatch/RDS region to query

# Metric windows to analyze: (label, lookback_days, period_seconds)
WINDOWS: List[Tuple[str, int, int]] = [
    ("14d_60s", 14, 60),
    ("30d_120s", 30, 120),
    ("60d_300s", 60, 300),
]

# Sensitivity factors for mapping DBLoad(AAS) → ACUs (if ACU metric is absent)
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]

# Compute-only prices
SERVERLESS_ACU_PER_HOUR = 0.12  # USD per ACU-hour (Serverless v2)

# On-demand hourly prices for provisioned instances (compute-only)
INSTANCE_HOURLY_PRICES: Dict[str, float] = {
    "db.r7g.large": 0.252,
    "db.r7g.xlarge": 0.504,
    "db.r7g.2xlarge": 1.008,
    "db.r7g.4xlarge": 2.016,
    "db.r6g.large": 0.246,
    "db.r6g.xlarge": 0.492,
    "db.r6g.2xlarge": 0.984,
    "db.r6g.4xlarge": 1.968,
}

# If you want to model I/O-Optimized uplift in compute price, set > 1.0
IO_OPTIMIZED_MULTIPLIER = 1.0

# Candidate instance classes to consider (order matters; smaller → larger)
CANDIDATE_CLASSES: List[str] = [
    "db.r7g.large", "db.r7g.xlarge", "db.r7g.2xlarge", "db.r7g.4xlarge",
    "db.r6g.large", "db.r6g.xlarge", "db.r6g.2xlarge", "db.r6g.4xlarge",
]

# Approx vCPUs used for sizing via DBLoad(AAS) p95 headroom rule
INSTANCE_VCPU: Dict[str, int] = {
    "db.r7g.large": 2,  "db.r7g.xlarge": 4,  "db.r7g.2xlarge": 8,  "db.r7g.4xlarge": 16,
    "db.r6g.large": 2,  "db.r6g.xlarge": 4,  "db.r6g.2xlarge": 8,  "db.r6g.4xlarge": 16,
}

# Sizing rule: keep p95 AAS under this fraction of vCPUs
AAS_HEADROOM = 0.70

# ACU bounds when estimating from AAS (if ACU metric missing)
MIN_ACU = 1.0
MAX_ACU = 64.0

# =========================
# ====== IMPLEMENTATION ===
# =========================

@dataclass
class MetricSeries:
    timestamps: List[dt.datetime]
    values: List[float]

@dataclass
class WorkloadShape:
    dbload_p50: float
    dbload_p95: float
    dbload_p99: float
    cpu_p50: float
    cpu_p95: float
    cpu_p99: float
    connections_p95: float

@dataclass
class CostBreakdown:
    monthly_total: float
    detail: str

@dataclass
class ProvisionedOption:
    instance_class: str
    fits: bool
    monthly_cost: float
    breakdown: CostBreakdown

@dataclass
class WindowDecision:
    label: str
    factor: float
    lookback_days: int
    period_seconds: int
    db_identifier: str
    engine: str
    mode_detected: str
    instances_found: List[str]
    current_instance_classes: List[str]
    shape: WorkloadShape
    serverless_cost: Optional[CostBreakdown]
    provisioned_best: Optional[ProvisionedOption]
    provisioned_current: Optional[ProvisionedOption]
    summary: str

def percentile(vals: List[float], p: float) -> float:
    if not vals: return 0.0
    s = sorted(vals)
    if p <= 0: return s[0]
    if p >= 100: return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c: return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def monthly_hours(start: dt.datetime, end: dt.datetime) -> float:
    return (end - start).total_seconds() / 3600.0

def get_clients(region: str):
    return boto3.client("rds", region_name=region), boto3.client("cloudwatch", region_name=region)

def describe_cluster(rds, db_identifier: str):
    resp = rds.describe_db_clusters(DBClusterIdentifier=db_identifier)
    cluster = resp["DBClusters"][0]
    insts = []
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            insts.append(rds.describe_db_instances(DBInstanceIdentifier=m["DBInstanceIdentifier"])["DBInstances"][0])
    return cluster, insts

def detect_mode(cluster: dict) -> str:
    if cluster.get("EngineMode") == "serverless": return "serverless-v1"
    if cluster.get("ServerlessV2ScalingConfiguration"): return "serverless-v2"
    return "provisioned"

def fetch_metrics(cw, cluster_id: str, writer_instance_id: Optional[str],
                  start: dt.datetime, end: dt.datetime, period: int) -> Dict[str, MetricSeries]:
    def mquery(qid, metric, dims, stat):
        return {"Id": qid, "MetricStat": {"Metric": {"Namespace":"AWS/RDS","MetricName":metric,"Dimensions":dims},
                                          "Period": period, "Stat": stat}, "ReturnData": True}
    dims_cluster = [{"Name":"DBClusterIdentifier","Value":cluster_id}]
    queries = [
        mquery("q1","DBLoad",dims_cluster,"Average"),
        mquery("q2","DBLoadCPU",dims_cluster,"Average"),
        mquery("q3","ServerlessDatabaseCapacity",dims_cluster,"Average"),
    ]
    if writer_instance_id:
        dims_inst = [{"Name":"DBInstanceIdentifier","Value":writer_instance_id}]
        queries += [
            mquery("q4","CPUUtilization",dims_inst,"Average"),
            mquery("q5","DatabaseConnections",dims_inst,"Average"),
        ]

    out: Dict[str, MetricSeries] = {}
    next_token = None
    while True:
        args = {"StartTime":start, "EndTime":end, "MetricDataQueries":queries,
                "ScanBy":"TimestampAscending", "MaxDatapoints":50000}
        if next_token: args["NextToken"] = next_token
        resp = cw.get_metric_data(**args)
        names = ["dbload","dbloadcpu","sv2_acu","cpu","connections"]
        for i, r in enumerate(resp["MetricDataResults"]):
            name = names[i]
            ts, vals = r.get("Timestamps", []) or [], r.get("Values", []) or []
            if name in out:
                out[name].timestamps.extend(ts); out[name].values.extend(vals)
            else:
                out[name] = MetricSeries(ts, vals)
        next_token = resp.get("NextToken")
        if not next_token: break

    # sort by timestamp
    for k, s in out.items():
        if s.timestamps:
            pair = sorted(zip(s.timestamps, s.values), key=lambda x: x[0])
            out[k] = MetricSeries([p[0] for p in pair], [p[1] for p in pair])
    return out

def summarize_shape(metrics: Dict[str, MetricSeries]) -> WorkloadShape:
    dbload = metrics.get("dbload").values if metrics.get("dbload") else []
    cpu = metrics.get("cpu").values if metrics.get("cpu") else []
    conn = metrics.get("connections").values if metrics.get("connections") else []
    return WorkloadShape(
        dbload_p50=percentile(dbload, 50.0) if dbload else 0.0,
        dbload_p95=percentile(dbload, 95.0) if dbload else 0.0,
        dbload_p99=percentile(dbload, 99.0) if dbload else 0.0,
        cpu_p50=percentile(cpu, 50.0) if cpu else 0.0,
        cpu_p95=percentile(cpu, 95.0) if cpu else 0.0,
        cpu_p99=percentile(cpu, 99.0) if cpu else 0.0,
        connections_p95=percentile(conn, 95.0) if conn else 0.0,
    )

def estimate_acu_seconds(metrics: Dict[str, MetricSeries],
                         factor: float, min_acu: float, max_acu: float) -> Tuple[float, str]:
    sv2, dbload = metrics.get("sv2_acu"), metrics.get("dbload")
    def infer_period(ts: List[dt.datetime]) -> int:
        return max(1, int((ts[1]-ts[0]).total_seconds())) if ts and len(ts) >= 2 else 60
    if sv2 and sv2.values:
        period = infer_period(sv2.timestamps)
        acu_seconds = sum((v or 0.0) * period for v in sv2.values)
        return float(acu_seconds), "Used ServerlessDatabaseCapacity"
    if dbload and dbload.values:
        period = infer_period(dbload.timestamps)
        est_acu = [max(min_acu, min(max_acu, (v or 0.0) * factor)) for v in dbload.values]
        acu_seconds = sum(a * period for a in est_acu)
        return float(acu_seconds), f"Estimated ACU from DBLoad (factor={factor:.2f})"
    return 0.0, "No DBLoad/ACU metric found; ACU=0"

def calc_serverless_cost(acu_seconds: float) -> CostBreakdown:
    acu_hours = acu_seconds / 3600.0
    total = acu_hours * SERVERLESS_ACU_PER_HOUR
    return CostBreakdown(total, f"{acu_hours:.1f} ACU-hrs × ${SERVERLESS_ACU_PER_HOUR:.4f}/ACU-hr")

def calc_provisioned_cost(instance_hour_price: float, hours: float, instance_count: int) -> CostBreakdown:
    total = instance_hour_price * IO_OPTIMIZED_MULTIPLIER * hours * instance_count
    note = f"{instance_count}× inst-hrs × ${instance_hour_price:.4f}/hr"
    if IO_OPTIMIZED_MULTIPLIER != 1.0:
        note += f" × io_opt_mult {IO_OPTIMIZED_MULTIPLIER:.3f}"
    return CostBreakdown(total, note)

def choose_instance_by_aas(dbload_p95: float) -> List[Tuple[str, bool]]:
    out = []
    for ic in CANDIDATE_CLASSES:
        v = INSTANCE_VCPU.get(ic, 0)
        out.append((ic, dbload_p95 <= AAS_HEADROOM * v))
    return out

def evaluate_one_window(*, label: str, factor: float,
                        db_identifier: str, region: str,
                        lookback_days: int, period_seconds: int) -> WindowDecision:
    rds, cw = get_clients(region)
    cluster, insts = describe_cluster(rds, db_identifier)
    engine, mode = cluster.get("Engine","aurora"), detect_mode(cluster)

    writer_id, instance_ids, current_classes = None, [], []
    for m in cluster.get("DBClusterMembers", []) or []:
        instance_ids.append(m["DBInstanceIdentifier"])
        if m.get("IsClusterWriter"): writer_id = m["DBInstanceIdentifier"]
    for di in insts:
        if di.get("DBInstanceClass"): current_classes.append(di["DBInstanceClass"])

    end, start = now_utc(), now_utc() - dt.timedelta(days=lookback_days)
    metrics = fetch_metrics(cw, cluster_id=db_identifier, writer_instance_id=writer_id,
                            start=start, end=end, period=period_seconds)
    shape = summarize_shape(metrics)
    hours = monthly_hours(start, end)
    inst_count = max(1, len(instance_ids))

    acu_seconds, acu_notes = estimate_acu_seconds(metrics, factor, MIN_ACU, MAX_ACU)
    sv2_cost = calc_serverless_cost(acu_seconds)

    options: List[ProvisionedOption] = []
    for ic, fits in choose_instance_by_aas(shape.dbload_p95):
        price = INSTANCE_HOURLY_PRICES.get(ic)
        if price is None: continue
        pcost = calc_provisioned_cost(price, hours, inst_count)
        options.append(ProvisionedOption(ic, fits, pcost.monthly_total, pcost))

    fits_only = [o for o in options if o.fits]
    best_prov = min(fits_only, key=lambda o: o.monthly_cost) if fits_only else None

    current_prov = None
    if current_classes:
        curr = current_classes[0]
        if curr in INSTANCE_HOURLY_PRICES:
            pcost = calc_provisioned_cost(INSTANCE_HOURLY_PRICES[curr], hours, inst_count)
            fits = (shape.dbload_p95 <= AAS_HEADROOM * INSTANCE_VCPU.get(curr, 0))
            current_prov = ProvisionedOption(curr, fits, pcost.monthly_total, pcost)

    lines = []
    lines.append(f"[{label}|factor={factor:.2f}] Cluster={db_identifier} | Engine={engine} | Mode={mode}")
    lines.append(f"Lookback={lookback_days}d @ {period_seconds}s; AAS p95={shape.dbload_p95:.2f}, CPU p95={shape.cpu_p95:.1f}%, Conn p95={shape.connections_p95:.0f}")
    if best_prov:
        lines.append(f"Best Provisioned: {best_prov.instance_class} → ${best_prov.monthly_cost:,.2f}/mo ({best_prov.breakdown.detail})")
    else:
        lines.append("Best Provisioned: (none fits or missing price entries)")
    lines.append(f"Serverless v2: ${sv2_cost.monthly_total:,.2f}/mo ({acu_notes}; {sv2_cost.detail})")
    if best_prov:
        if best_prov.monthly_cost < sv2_cost.monthly_total:
            lines.append(f"→ Recommendation: PROVISIONED on {best_prov.instance_class} (cheaper by ${sv2_cost.monthly_total - best_prov.monthly_cost:,.2f}/mo).")
        else:
            lines.append(f"→ Recommendation: SERVERLESS v2 (cheaper by ${best_prov.monthly_cost - sv2_cost.monthly_total:,.2f}/mo).")
    else:
        lines.append("→ Recommendation: SERVERLESS v2 (no valid provisioned fit).")

    if mode == "provisioned" and current_prov and best_prov:
        cur_v = INSTANCE_VCPU.get(current_prov.instance_class, 0)
        new_v = INSTANCE_VCPU.get(best_prov.instance_class, 0)
        if new_v < cur_v: lines.append(f"Resize advice: DOWNSIZE {current_prov.instance_class} → {best_prov.instance_class}.")
        elif new_v > cur_v: lines.append(f"Resize advice: UPGRADE {current_prov.instance_class} → {best_prov.instance_class}.")
        else: lines.append("Resize advice: Current size already optimal among candidates.")

    return WindowDecision(
        label=label, factor=factor,
        lookback_days=lookback_days, period_seconds=period_seconds,
        db_identifier=db_identifier, engine=engine, mode_detected=mode,
        instances_found=instance_ids, current_instance_classes=current_classes,
        shape=shape, serverless_cost=sv2_cost, provisioned_best=best_prov,
        provisioned_current=current_prov, summary="\n".join(lines)
    )

def lambda_handler(event, context):
    try:
        db_cluster_id = event.get("db_cluster_id")
        if not db_cluster_id:
            return {"statusCode": 400, "body": json.dumps({"error": "db_cluster_id is required in the event."})}

        results: Dict[str, Dict[str, dict]] = {}
        summaries: List[str] = []
        for label, days, period in WINDOWS:
            results[label] = {}
            for factor in AAS_TO_ACU_FACTORS:
                wd = evaluate_one_window(
                    label=label, factor=factor,
                    db_identifier=db_cluster_id, region=REGION,
                    lookback_days=days, period_seconds=period
                )
                key = f"{factor:.2f}"
                results[label][key] = asdict(wd)
                summaries.append("=" * 88)
                summaries.append(f"Window: {label} | Factor: {key}")
                summaries.append(wd.summary)

        body = {
            "db_cluster_id": db_cluster_id,
            "region": REGION,
            "windows": WINDOWS,
            "factors": AAS_TO_ACU_FACTORS,
            "serverless_acu_per_hour": SERVERLESS_ACU_PER_HOUR,
            "instance_hourly_prices": INSTANCE_HOURLY_PRICES,
            "io_optimized_multiplier": IO_OPTIMIZED_MULTIPLIER,
            "results": results,
            "summary": "\n".join(summaries),
        }
        return {"statusCode": 200, "body": json.dumps(body, default=str)}
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
