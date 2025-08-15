# lambda_function.py
"""
Aurora Serverless v2 vs Provisioned cost comparison (Python 3.12)
- REAL ACU for Serverless v2
- Provisioned: ACU estimated from DBLoad × factor and CPUUtilization (blend)
- Standard vs I/O-Optimized storage comparison (aurora vs aurora-iopt1)
- I/O charges only for Standard storage
- Multi-window (14d/30d/60d), AAS→ACU sensitivity, wide instance catalog
- Post-processor: cheapest overall, by storage, and top-3
- Verbose logging with print(asdict(...)) for easy CloudWatch tracing
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

REGION = "us-east-1"

# Compare storage modes:
#   "auto"     -> use detected StorageType
#   "standard" -> force Standard (aurora)
#   "iopt1"    -> force I/O-Optimized (aurora-iopt1)
#   "both"     -> evaluate twice: Standard + I/O-Optimized
STORAGE_COMPARISON_MODE = "both"

WINDOWS: List[Tuple[str, int, int]] = [
    ("14d_60s", 14, 60),
    ("30d_120s", 30, 120),
    ("60d_300s", 60, 300),
]

# AAS→ACU sensitivity factors used only when we estimate ACU
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]

# Serverless v2 (Standard storage) compute price
SERVERLESS_ACU_PER_HOUR = 0.12  # USD per ACU-hour

# If I/O-Optimized, multiply ACU price by this factor
SERVERLESS_IOPT_MULTIPLIER = 1.17

# I/O request price for Standard storage (no I/O charges for aurora-iopt1)
IO_PRICE_PER_MILLION = 0.20  # USD per 1,000,000 requests
CHARGE_IO_FOR_STANDARD = True

# ---- ACU snapping/limits (for estimated ACU on provisioned clusters) ----
DEFAULT_MIN_ACU = 0.5
DEFAULT_MAX_ACU = 64.0
ACU_STEP = 0.5  # valid ACU granularity

# ---- CPU-aware ACU estimation (provisioned only) ----
USE_CPU_IN_ACU_ESTIMATE = True   # blend CPU into ACU estimate
ACU_BLEND_MODE = "max"           # "max" (recommended) or "avg"
CPU_TO_ACU_HEADROOM = 0.70       # scale CPU→ACU similar to AAS headroom
CPU_WEIGHT = 0.5                 # only used when ACU_BLEND_MODE == "avg"

# -------- Instance catalog (classes + vCPUs + prices) --------
CANDIDATE_CLASSES: List[str] = [
    # Burstable
    "db.t3.medium", "db.t3.large", "db.t3.xlarge", "db.t3.2xlarge",
    "db.t4g.medium", "db.t4g.large", "db.t4g.xlarge", "db.t4g.2xlarge",
    # Memory-optimized (Intel r4/r5)
    "db.r4.large", "db.r4.xlarge", "db.r4.2xlarge", "db.r4.4xlarge", "db.r4.8xlarge", "db.r4.16xlarge",
    "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge", "db.r5.4xlarge", "db.r5.12xlarge", "db.r5.24xlarge",
    # Graviton (r6g/r7g/r8g)
    "db.r6g.large", "db.r6g.xlarge", "db.r6g.2xlarge", "db.r6g.4xlarge", "db.r6g.8xlarge", "db.r6g.12xlarge", "db.r6g.16xlarge",
    "db.r7g.large", "db.r7g.xlarge", "db.r7g.2xlarge", "db.r7g.4xlarge", "db.r7g.8xlarge",
    "db.r8g.large", "db.r8g.xlarge", "db.r8g.2xlarge", "db.r8g.4xlarge", "db.r8g.8xlarge",
    # Extended memory-optimized
    "db.x1e.xlarge", "db.x1e.2xlarge", "db.x1e.4xlarge", "db.x1e.8xlarge", "db.x1e.16xlarge", "db.x1e.32xlarge",
    "db.x2g.large", "db.x2g.xlarge", "db.x2g.2xlarge", "db.x2g.4xlarge", "db.x2g.8xlarge", "db.x2g.12xlarge", "db.x2g.16xlarge",
    # High-frequency compute
    "db.z1d.large", "db.z1d.xlarge", "db.z1d.2xlarge", "db.z1d.3xlarge", "db.z1d.6xlarge", "db.z1d.12xlarge",
]

INSTANCE_VCPU: Dict[str, int] = {
    "db.t3.medium": 2, "db.t3.large": 2, "db.t3.xlarge": 4, "db.t3.2xlarge": 8,
    "db.t4g.medium": 2, "db.t4g.large": 2, "db.t4g.xlarge": 4, "db.t4g.2xlarge": 8,
    "db.r4.large": 2, "db.r4.xlarge": 4, "db.r4.2xlarge": 8, "db.r4.4xlarge": 16, "db.r4.8xlarge": 32, "db.r4.16xlarge": 64,
    "db.r5.large": 2, "db.r5.xlarge": 4, "db.r5.2xlarge": 8, "db.r5.4xlarge": 16, "db.r5.12xlarge": 48, "db.r5.24xlarge": 96,
    "db.r6g.large": 2, "db.r6g.xlarge": 4, "db.r6g.2xlarge": 8, "db.r6g.4xlarge": 16, "db.r6g.8xlarge": 32, "db.r6g.12xlarge": 48, "db.r6g.16xlarge": 64,
    "db.r7g.large": 2, "db.r7g.xlarge": 4, "db.r7g.2xlarge": 8, "db.r7g.4xlarge": 16, "db.r7g.8xlarge": 32,
    "db.r8g.large": 2, "db.r8g.xlarge": 4, "db.r8g.2xlarge": 8, "db.r8g.4xlarge": 16, "db.r8g.8xlarge": 32,
    "db.x1e.xlarge": 4, "db.x1e.2xlarge": 8, "db.x1e.4xlarge": 16, "db.x1e.8xlarge": 32, "db.x1e.16xlarge": 64, "db.x1e.32xlarge": 128,
    "db.x2g.large": 2, "db.x2g.xlarge": 4, "db.x2g.2xlarge": 8, "db.x2g.4xlarge": 16, "db.x2g.8xlarge": 32, "db.x2g.12xlarge": 48, "db.x2g.16xlarge": 64,
    "db.z1d.large": 2, "db.z1d.xlarge": 4, "db.z1d.2xlarge": 8, "db.z1d.3xlarge": 12, "db.z1d.6xlarge": 24, "db.z1d.12xlarge": 48,
}

# Hourly prices (compute only). Fill what you want considered.
INSTANCE_HOURLY_PRICES: Dict[str, float] = {
    "db.t3.medium": 0.067, "db.t3.large": 0.134, "db.t3.xlarge": 0.268, "db.t3.2xlarge": 0.536,
    "db.t4g.medium": 0.058, "db.t4g.large": 0.116, "db.t4g.xlarge": 0.232, "db.t4g.2xlarge": 0.464,
    "db.r4.large": 0.29, "db.r4.xlarge": 0.58, "db.r4.2xlarge": 1.16, "db.r4.4xlarge": 2.32, "db.r4.8xlarge": 4.64, "db.r4.16xlarge": 9.28,
    "db.r5.large": 0.25, "db.r5.xlarge": 0.50, "db.r5.2xlarge": 1.00, "db.r5.4xlarge": 2.00, "db.r5.12xlarge": 6.00, "db.r5.24xlarge": 12.00,
    "db.r6g.large": 0.246, "db.r6g.xlarge": 0.492, "db.r6g.2xlarge": 0.984, "db.r6g.4xlarge": 1.968, "db.r6g.8xlarge": 3.936, "db.r6g.12xlarge": 5.904, "db.r6g.16xlarge": 7.872,
    "db.r7g.large": 0.252, "db.r7g.xlarge": 0.504, "db.r7g.2xlarge": 1.008, "db.r7g.4xlarge": 2.016, "db.r7g.8xlarge": 4.032,
    # Add r8g/x1e/x2g/z1d as needed...
}

# If I/O-Optimized (aurora-iopt1), multiply provisioned hourly prices by:
IO_OPTIMIZED_MULTIPLIERS_BY_CLASS: Dict[str, float] = {
    "db.r6g.large": 1.170, "db.r6g.xlarge": 1.170, "db.r6g.2xlarge": 1.170, "db.r6g.4xlarge": 1.170,
    "db.r6g.8xlarge": 1.170, "db.r6g.12xlarge": 1.170, "db.r6g.16xlarge": 1.170,
    "db.r7g.large": 1.167, "db.r7g.xlarge": 1.167, "db.r7g.2xlarge": 1.167, "db.r7g.4xlarge": 1.167, "db.r7g.8xlarge": 1.167,
    "db.r8g.large": 1.167, "db.r8g.xlarge": 1.167, "db.r8g.2xlarge": 1.167, "db.r8g.4xlarge": 1.167, "db.r8g.8xlarge": 1.167,
    "db.r4.large": 1.15, "db.r4.xlarge": 1.15, "db.r4.2xlarge": 1.15, "db.r4.4xlarge": 1.15, "db.r4.8xlarge": 1.15, "db.r4.16xlarge": 1.15,
    "db.r5.large": 1.15, "db.r5.xlarge": 1.15, "db.r5.2xlarge": 1.15, "db.r5.4xlarge": 1.15, "db.r5.12xlarge": 1.15, "db.r5.24xlarge": 1.15,
    "db.t3.medium": 1.15, "db.t3.large": 1.15, "db.t3.xlarge": 1.15, "db.t3.2xlarge": 1.15,
    "db.t4g.medium": 1.15, "db.t4g.large": 1.15, "db.t4g.xlarge": 1.15, "db.t4g.2xlarge": 1.15,
    "db.x1e.xlarge": 1.12, "db.x1e.2xlarge": 1.12, "db.x1e.4xlarge": 1.12, "db.x1e.8xlarge": 1.12, "db.x1e.16xlarge": 1.12, "db.x1e.32xlarge": 1.12,
    "db.x2g.large": 1.12, "db.x2g.xlarge": 1.12, "db.x2g.2xlarge": 1.12, "db.x2g.4xlarge": 1.12, "db.x2g.8xlarge": 1.12, "db.x2g.12xlarge": 1.12, "db.x2g.16xlarge": 1.12,
    "db.z1d.large": 1.15, "db.z1d.xlarge": 1.15, "db.z1d.2xlarge": 1.15, "db.z1d.3xlarge": 1.15, "db.z1d.6xlarge": 1.15, "db.z1d.12xlarge": 1.15,
    "default": 1.17,
}

AAS_HEADROOM = 0.70

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
    storage_type_used: str
    io_optimized: bool
    instances_found: List[str]
    current_instance_classes: List[str]
    shape: WorkloadShape
    serverless_cost: Optional[CostBreakdown]
    provisioned_best: Optional[ProvisionedOption]
    provisioned_current: Optional[ProvisionedOption]
    io_request_summary: str
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
    print(f"[get_clients] region={region}")
    return boto3.client("rds", region_name=region), boto3.client("cloudwatch", region_name=region)

def describe_cluster(rds, db_identifier: str):
    print(f"[describe_cluster] DBClusterIdentifier={db_identifier}")
    resp = rds.describe_db_clusters(DBClusterIdentifier=db_identifier)
    cluster = resp["DBClusters"][0]
    insts = []
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            iid = m["DBInstanceIdentifier"]
            print(f"[describe_cluster] Describe DB instance id={iid}")
            insts.append(rds.describe_db_instances(DBInstanceIdentifier=iid)["DBInstances"][0])
    print(f"[describe_cluster] Instances found: {len(insts)}")
    return cluster, insts

def detect_mode(cluster: dict) -> str:
    if cluster.get("EngineMode") == "serverless": return "serverless-v1"
    if cluster.get("ServerlessV2ScalingConfiguration"): return "serverless-v2"
    return "provisioned"

def detect_storage_type(cluster: dict, insts: List[dict]) -> str:
    st = (cluster.get("StorageType") or "").lower()
    if st:
        print(f"[detect_storage_type] Cluster StorageType={st}")
        return st
    for di in insts:
        st_i = (di.get("StorageType") or "").lower()
        if st_i:
            print(f"[detect_storage_type] Instance StorageType={st_i} (used)")
            return st_i
    print("[detect_storage_type] StorageType unknown")
    return ""

def get_cluster_acu_limits(cluster: dict) -> Tuple[float, float]:
    cfg = cluster.get("ServerlessV2ScalingConfiguration") or {}
    minc = float(cfg.get("MinCapacity") or DEFAULT_MIN_ACU)
    maxc = float(cfg.get("MaxCapacity") or DEFAULT_MAX_ACU)
    print(f"[get_cluster_acu_limits] MinCapacity={minc}, MaxCapacity={maxc}")
    return minc, maxc

def snap_acu(val: float, min_acu: float, max_acu: float, step: float = ACU_STEP) -> float:
    if val <= 0: return min_acu
    snapped = max(min_acu, min(max_acu, round(val / step) * step))
    return snapped

def fetch_metrics(cw, cluster_id: str, instance_ids: List[str], writer_instance_id: Optional[str],
                  start: dt.datetime, end: dt.datetime, period: int) -> Dict[str, MetricSeries]:
    print(f"[fetch_metrics] cluster={cluster_id}, period={period}s, {start} → {end}")
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
        dims_writer = [{"Name":"DBInstanceIdentifier","Value":writer_instance_id}]
        queries += [
            mquery("q4","CPUUtilization",dims_writer,"Average"),
            mquery("q5","DatabaseConnections",dims_writer,"Average"),
        ]
    qid = 6
    for iid in instance_ids:
        dims_inst = [{"Name":"DBInstanceIdentifier","Value":iid}]
        queries += [
            mquery(f"ri_{qid}","ReadIOPS",dims_inst,"Average"),
            mquery(f"wi_{qid+1}","WriteIOPS",dims_inst,"Average"),
        ]
        qid += 2

    out: Dict[str, MetricSeries] = {}
    next_token = None
    total_points = 0
    while True:
        args = {"StartTime":start, "EndTime":end, "MetricDataQueries":queries,
                "ScanBy":"TimestampAscending", "MaxDatapoints":50000}
        if next_token: args["NextToken"] = next_token
        resp = cw.get_metric_data(**args)
        for r in resp["MetricDataResults"]:
            name = r["Id"]
            ts, vals = r.get("Timestamps", []) or [], r.get("Values", []) or []
            total_points += len(vals)
            if name in out:
                out[name].timestamps.extend(ts); out[name].values.extend(vals)
            else:
                out[name] = MetricSeries(ts, vals)
        next_token = resp.get("NextToken")
        if not next_token: break

    for k, s in out.items():
        if s.timestamps:
            pair = sorted(zip(s.timestamps, s.values), key=lambda x: x[0])
            out[k] = MetricSeries([p[0] for p in pair], [p[1] for p in pair])
    print(f"[fetch_metrics] datapoints={total_points}")
    return out

def summarize_shape(metrics: Dict[str, MetricSeries]) -> WorkloadShape:
    dbload = metrics.get("q1").values if metrics.get("q1") else []
    cpu = metrics.get("q4").values if metrics.get("q4") else []
    conn = metrics.get("q5").values if metrics.get("q5") else []
    shape = WorkloadShape(
        dbload_p50=percentile(dbload, 50.0) if dbload else 0.0,
        dbload_p95=percentile(dbload, 95.0) if dbload else 0.0,
        dbload_p99=percentile(dbload, 99.0) if dbload else 0.0,
        cpu_p50=percentile(cpu, 50.0) if cpu else 0.0,
        cpu_p95=percentile(cpu, 95.0) if cpu else 0.0,
        cpu_p99=percentile(cpu, 99.0) if cpu else 0.0,
        connections_p95=percentile(conn, 95.0) if conn else 0.0,
    )
    print(f"[summarize_shape] AAS p95={shape.dbload_p95:.2f}, CPU p95={shape.cpu_p95:.1f}%, Conn p95={shape.connections_p95:.0f}")
    return shape

def infer_period_from_series(ts: List[dt.datetime]) -> int:
    return max(1, int((ts[1]-ts[0]).total_seconds())) if ts and len(ts) >= 2 else 60

def estimate_acu_seconds(cluster_mode: str, cluster: dict, metrics: Dict[str, MetricSeries],
                         factor: float, min_acu_default: float, max_acu_default: float,
                         writer_vcpu: int) -> Tuple[float, str]:
    """
    - For serverless-v2: integrate real ServerlessDatabaseCapacity; clamp to cluster Min/Max.
    - For provisioned: estimate ACU per-sample from DBLoad×factor and CPUUtilization, blend, snap to ACU_STEP, clamp to defaults.
    """
    # Serverless v2: real ACU
    if cluster_mode == "serverless-v2":
        sv2 = metrics.get("q3")
        minc, maxc = get_cluster_acu_limits(cluster)
        if sv2 and sv2.values:
            period = infer_period_from_series(sv2.timestamps)
            clamped = [max(minc, min(maxc, v or 0.0)) for v in sv2.values]
            acu_seconds = sum(v * period for v in clamped)
            print(f"[estimate_acu_seconds] Serverless mode: using ACU metric; period={period}s, clamped to [{minc},{maxc}], ACU-sec={acu_seconds:.0f}")
            return float(acu_seconds), f"Used ServerlessDatabaseCapacity (clamped to {minc}-{maxc} ACU)"
        print("[estimate_acu_seconds] Serverless mode but no ACU samples; falling back to DBLoad/CPU estimate")

    # Provisioned (or fallback)
    dbload = metrics.get("q1")
    cpu = metrics.get("q4")  # writer CPUUtilization
    # pick a reasonable period from whichever series exists
    base_series = dbload or cpu or MetricSeries([], [])
    period = infer_period_from_series(base_series.timestamps)

    # A) ACU from DBLoad × factor
    acu_from_dbload: List[float] = []
    if dbload and dbload.values:
        for v in dbload.values:
            raw = (v or 0.0) * factor
            snapped = snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP)
            acu_from_dbload.append(snapped)

    # B) ACU from CPUUtilization × writer vCPU / headroom
    acu_from_cpu: List[float] = []
    if USE_CPU_IN_ACU_ESTIMATE and cpu and cpu.values and writer_vcpu and writer_vcpu > 0:
        for cpu_pct in cpu.values:
            u = (cpu_pct or 0.0) / 100.0
            raw = (u * writer_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
            snapped = snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP)
            acu_from_cpu.append(snapped)
    elif USE_CPU_IN_ACU_ESTIMATE and writer_vcpu == 0:
        print("[estimate_acu_seconds] CPU blending enabled but writer_vcpu unknown; skipping CPU path")

    # Blend the two series
    est_series: List[float] = []
    if acu_from_dbload and acu_from_cpu:
        n = min(len(acu_from_dbload), len(acu_from_cpu))
        if ACU_BLEND_MODE == "avg":
            for i in range(n):
                mixed = CPU_WEIGHT * acu_from_cpu[i] + (1 - CPU_WEIGHT) * acu_from_dbload[i]
                est_series.append(snap_acu(mixed, min_acu_default, max_acu_default, ACU_STEP))
            detail = f"Estimated ACU from DBLoad×{factor:.2f} and CPU (avg w={CPU_WEIGHT}); snapped {ACU_STEP} ACU"
        else:  # "max"
            for i in range(n):
                est_series.append(max(acu_from_dbload[i], acu_from_cpu[i]))
            detail = f"Estimated ACU from DBLoad×{factor:.2f} and CPU (max); snapped {ACU_STEP} ACU"
    elif acu_from_dbload:
        est_series = acu_from_dbload
        detail = f"Estimated ACU from DBLoad×{factor:.2f}; snapped {ACU_STEP} ACU"
    elif acu_from_cpu:
        est_series = acu_from_cpu
        detail = f"Estimated ACU from CPU; snapped {ACU_STEP} ACU"
    else:
        print("[estimate_acu_seconds] No DBLoad/CPU data; ACU=0")
        return 0.0, "No DBLoad/ACU metric found; ACU=0"

    acu_seconds = sum(a * period for a in est_series)
    print(f"[estimate_acu_seconds] {detail}; bounds=[{min_acu_default},{max_acu_default}], ACU-sec={acu_seconds:.0f}")
    return float(acu_seconds), detail

def estimate_total_io_requests(metrics: Dict[str, MetricSeries]) -> Tuple[float, str]:
    total_requests = 0.0
    series_count = 0
    for key, series in metrics.items():
        if key.startswith("ri_") or key.startswith("wi_"):
            period = infer_period_from_series(series.timestamps)
            total_requests += sum((v or 0.0) * period for v in series.values)
            series_count += 1
    detail = f"Integrated ReadIOPS/WriteIOPS across {series_count} instance series"
    print(f"[estimate_total_io_requests] total_requests={total_requests:.0f} ({detail})")
    return total_requests, detail

def calc_serverless_cost(acu_seconds: float, storage_type: str, io_requests: float) -> CostBreakdown:
    acu_hours = acu_seconds / 3600.0
    acu_price = SERVERLESS_ACU_PER_HOUR * (SERVERLESS_IOPT_MULTIPLIER if storage_type == "aurora-iopt1" else 1.0)
    compute_total = acu_hours * acu_price
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    cb = CostBreakdown(compute_total + io_total,
                       f"{acu_hours:.1f} ACU-hrs × ${acu_price:.4f}/ACU-hr + "
                       + ("no I/O charge (iopt1)" if storage_type == "aurora-iopt1"
                          else f"I/O charged ({io_requests/1_000_000:.3f}M @ ${IO_PRICE_PER_MILLION:.2f}/M)"))
    print("[calc_serverless_cost] asdict:", asdict(cb))
    return cb

def get_io_multiplier(storage_type: str, instance_class: str) -> float:
    if storage_type == "aurora-iopt1":
        mult = IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get(instance_class, IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get("default", 1.17))
        print(f"[get_io_multiplier] I/O-Optimized; {instance_class} → x{mult}")
        return mult
    print(f"[get_io_multiplier] Standard storage; {instance_class} → x1.0")
    return 1.0

def calc_provisioned_cost(compute_price_per_hour: float, hours: float, instance_count: int,
                          storage_type: str, io_requests: float) -> CostBreakdown:
    compute_total = compute_price_per_hour * hours * instance_count
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    cb = CostBreakdown(compute_total + io_total,
                       f"{instance_count}× inst-hrs × ${compute_price_per_hour:.4f}/hr + "
                       + ("no I/O charge (iopt1)" if storage_type == "aurora-iopt1"
                          else f"I/O charged ({io_requests/1_000_000:.3f}M @ ${IO_PRICE_PER_MILLION:.2f}/M)"))
    print("[calc_provisioned_cost] asdict:", asdict(cb))
    return cb

def choose_instance_by_aas(dbload_p95: float) -> List[Tuple[str, bool]]:
    print(f"[choose_instance_by_aas] AAS p95={dbload_p95:.2f}, headroom={AAS_HEADROOM:.2f}")
    out = []
    for ic in CANDIDATE_CLASSES:
        v = INSTANCE_VCPU.get(ic, 0)
        fits = dbload_p95 <= AAS_HEADROOM * v
        print(f"  - {ic}: vCPU={v}, threshold={AAS_HEADROOM*v:.2f}, fits={fits}")
        out.append((ic, fits))
    return out

def evaluate_one_window_with_storage(*, label: str, factor: float,
                                     db_identifier: str, region: str,
                                     lookback_days: int, period_seconds: int,
                                     storage_type_used: str,
                                     detected_storage_type: str,
                                     cluster: dict) -> WindowDecision:
    print(f"\n=== [evaluate_one_window] {label} factor={factor:.2f} storage={storage_type_used} (detected={detected_storage_type or 'unknown'}) ===")
    rds, cw = get_clients(region)

    # Re-describe instances to know classes / writer
    insts: List[dict] = []
    writer_id = None
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            iid = m["DBInstanceIdentifier"]
            di = rds.describe_db_instances(DBInstanceIdentifier=iid)["DBInstances"][0]
            insts.append(di)
            if m.get("IsClusterWriter"): writer_id = iid

    engine, mode = cluster.get("Engine","aurora"), detect_mode(cluster)

    instance_ids = [m["DBInstanceIdentifier"] for m in cluster.get("DBClusterMembers", []) or []]
    current_classes = [di.get("DBInstanceClass") for di in insts if di.get("DBInstanceClass")]
    print(f"[evaluate_one_window] writer={writer_id}, members={instance_ids}, current_classes={current_classes}, mode={mode}")

    # Determine writer vCPU (for CPU→ACU)
    writer_vcpu = 0
    if writer_id:
        for di in insts:
            if di.get("DBInstanceIdentifier") == writer_id:
                writer_class = di.get("DBInstanceClass")
                writer_vcpu = INSTANCE_VCPU.get(writer_class or "", 0)
                print(f"[evaluate_one_window] writer_class={writer_class}, writer_vcpu={writer_vcpu}")
                break

    end, start = now_utc(), now_utc() - dt.timedelta(days=lookback_days)
    metrics = fetch_metrics(cw, cluster_id=db_identifier, instance_ids=instance_ids, writer_instance_id=writer_id,
                            start=start, end=end, period=period_seconds)
    shape = summarize_shape(metrics)
    hours = monthly_hours(start, end)
    inst_count = max(1, len(instance_ids))
    print(f"[evaluate_one_window] hours={hours:.1f}, inst_count={inst_count}")

    # ACU usage: REAL if serverless-v2; else estimate from DBLoad & CPU
    acu_seconds, acu_notes = estimate_acu_seconds(mode, cluster, metrics, factor,
                                                  DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, writer_vcpu)
    io_requests, io_detail = estimate_total_io_requests(metrics)

    # ----- SERVERLESS (with storage type) -----
    sv2_cost = calc_serverless_cost(acu_seconds, storage_type_used, io_requests)

    # ----- PROVISIONED options (with storage type) -----
    options: List[ProvisionedOption] = []
    for ic, fits in choose_instance_by_aas(shape.dbload_p95):
        base_price = INSTANCE_HOURLY_PRICES.get(ic)
        if base_price is None:
            print(f"[evaluate_one_window] Skipping {ic} (no price configured)")
            continue
        mult = get_io_multiplier(storage_type_used, ic)
        compute_price = base_price * mult
        pcost = calc_provisioned_cost(compute_price, hours, inst_count, storage_type_used, io_requests)
        opt = ProvisionedOption(ic, fits, pcost.monthly_total, pcost)
        print("[evaluate_one_window] provisioned_option asdict:", asdict(opt))
        options.append(opt)

    fits_only = [o for o in options if o.fits]
    best_prov = min(fits_only, key=lambda o: o.monthly_cost) if fits_only else None
    if best_prov:
        print("[evaluate_one_window] best_provisioned asdict:", asdict(best_prov))
    else:
        print("[evaluate_one_window] No fitting provisioned option among priced classes")

    # Current class (if provisioned)
    current_prov = None
    if current_classes:
        curr = current_classes[0]
        if curr in INSTANCE_HOURLY_PRICES:
            mult = get_io_multiplier(storage_type_used, curr)
            compute_price = INSTANCE_HOURLY_PRICES[curr] * mult
            pcost = calc_provisioned_cost(compute_price, hours, inst_count, storage_type_used, io_requests)
            fits = (shape.dbload_p95 <= AAS_HEADROOM * INSTANCE_VCPU.get(curr, 0))
            current_prov = ProvisionedOption(curr, fits, pcost.monthly_total, pcost)
            print("[evaluate_one_window] current_provisioned asdict:", asdict(current_prov))
        else:
            print(f"[evaluate_one_window] current class {curr} has no configured price")

    # ----- Summary text -----
    lines = []
    lines.append(f"[{label}|factor={factor:.2f}|storage={storage_type_used}] Cluster={db_identifier} | Engine={engine} | Mode={mode} | DetectedStorage={detected_storage_type or 'unknown'}")
    lines.append(f"Lookback={lookback_days}d @ {period_seconds}s; AAS p95={shape.dbload_p95:.2f}, CPU p95={shape.cpu_p95:.1f}%, Conn p95={shape.connections_p95:.0f}")
    lines.append(f"I/O: {io_detail}; billed @ ${IO_PRICE_PER_MILLION:.2f}/M when Standard")
    if best_prov:
        lines.append(f"Best Provisioned: {best_prov.instance_class} → ${best_prov.monthly_cost:,.2f}/mo ({best_prov.breakdown.detail})")
    else:
        lines.append("Best Provisioned: (none fits or missing price entries)")
    lines.append(f"Serverless v2: ${sv2_cost.monthly_total:,.2f}/mo ({acu_notes}; {sv2_cost.detail})")
    if best_prov:
        if best_prov.monthly_cost < sv2_cost.monthly_total:
            diff = sv2_cost.monthly_total - best_prov.monthly_cost
            lines.append(f"→ Recommendation: PROVISIONED on {best_prov.instance_class} (cheaper by ${diff:,.2f}/mo).")
        else:
            diff = best_prov.monthly_cost - sv2_cost.monthly_total
            lines.append(f"→ Recommendation: SERVERLESS v2 (cheaper by ${diff:,.2f}/mo).")
    else:
        lines.append("→ Recommendation: SERVERLESS v2 (no valid provisioned fit).")

    if mode == "provisioned" and current_prov and best_prov:
        cur_v = INSTANCE_VCPU.get(current_prov.instance_class, 0)
        new_v = INSTANCE_VCPU.get(best_prov.instance_class, 0)
        if new_v < cur_v: lines.append(f"Resize advice: DOWNSIZE {current_prov.instance_class} → {best_prov.instance_class}.")
        elif new_v > cur_v: lines.append(f"Resize advice: UPGRADE {current_prov.instance_class} → {best_prov.instance_class}.")
        else: lines.append("Resize advice: Current size already optimal among candidates.")

    wd = WindowDecision(
        label=label, factor=factor,
        lookback_days=lookback_days, period_seconds=period_seconds,
        db_identifier=db_identifier, engine=engine, mode_detected=mode,
        storage_type_used=storage_type_used, io_optimized=(storage_type_used == "aurora-iopt1"),
        instances_found=instance_ids, current_instance_classes=current_classes,
        shape=shape, serverless_cost=sv2_cost, provisioned_best=best_prov,
        provisioned_current=current_prov, io_request_summary=f"{io_detail}; total={io_requests:.0f}",
        summary="\n".join(lines)
    )
    print("[evaluate_one_window] window_decision asdict:", asdict(wd))
    return wd

# ---------- Post-processor helpers ----------

def _collect_candidates(flat_results: Dict) -> List[Dict]:
    out: List[Dict] = []
    for storage_key, win_map in flat_results.items():
        for window_label, factor_map in win_map.items():
            for factor_str, wd in factor_map.items():
                sv = wd.get("serverless_cost") or {}
                out.append({
                    "storage": storage_key,
                    "window": window_label,
                    "factor": factor_str,
                    "type": "serverless",
                    "instance_class": None,
                    "monthly_cost": sv.get("monthly_total", float("inf")),
                    "detail": sv.get("detail", ""),
                    "headroom_fit": None,
                })
                bp = wd.get("provisioned_best")
                if bp:
                    out.append({
                        "storage": storage_key,
                        "window": window_label,
                        "factor": factor_str,
                        "type": "provisioned",
                        "instance_class": bp.get("instance_class"),
                        "monthly_cost": bp.get("monthly_cost", float("inf")),
                        "detail": (bp.get("breakdown") or {}).get("detail", ""),
                        "headroom_fit": bp.get("fits", True),
                    })
    out = [c for c in out if isinstance(c.get("monthly_cost", None), (int, float))]
    return out

def _pick_cheapest(cands: List[Dict]) -> Optional[Dict]:
    return min(cands, key=lambda c: c["monthly_cost"]) if cands else None

def _topk(cands: List[Dict], k: int) -> List[Dict]:
    return sorted(cands, key=lambda c: c["monthly_cost"])[:k]

# -------------- Lambda entry ----------------

def lambda_handler(event, context):
    try:
        print(f"[lambda_handler] event={json.dumps(event)}")
        db_cluster_id = event.get("db_cluster_id")
        if not db_cluster_id:
            msg = "db_cluster_id is required"
            print(f"[lambda_handler] ERROR: {msg}")
            return {"statusCode": 400, "body": json.dumps({"error": msg})}

        rds, _ = get_clients(REGION)
        cluster, insts = describe_cluster(rds, db_cluster_id)
        detected_storage = detect_storage_type(cluster, insts)  # may be ""

        # Decide which storage types to simulate
        if STORAGE_COMPARISON_MODE == "auto":
            modes_to_run = [detected_storage or "aurora"]
        elif STORAGE_COMPARISON_MODE == "standard":
            modes_to_run = ["aurora"]
        elif STORAGE_COMPARISON_MODE == "iopt1":
            modes_to_run = ["aurora-iopt1"]
        elif STORAGE_COMPARISON_MODE == "both":
            modes_to_run = ["aurora", "aurora-iopt1"]
        else:
            print(f"[lambda_handler] WARNING: unknown STORAGE_COMPARISON_MODE={STORAGE_COMPARISON_MODE}, defaulting to 'auto'")
            modes_to_run = [detected_storage or "aurora"]

        print(f"[lambda_handler] DetectedStorage={detected_storage or 'unknown'}, ComparisonModes={modes_to_run}")

        results: Dict[str, Dict[str, Dict[str, dict]]] = {}
        summaries: List[str] = []

        for storage_mode in modes_to_run:
            storage_key = "standard" if storage_mode == "aurora" else "iopt1"
            results[storage_key] = {}
            for label, days, period in WINDOWS:
                results[storage_key][label] = {}
                for factor in AAS_TO_ACU_FACTORS:
                    wd = evaluate_one_window_with_storage(
                        label=label, factor=factor,
                        db_identifier=db_cluster_id, region=REGION,
                        lookback_days=days, period_seconds=period,
                        storage_type_used=storage_mode, detected_storage_type=detected_storage,
                        cluster=cluster
                    )
                    key = f"{factor:.2f}"
                    results[storage_key][label][key] = asdict(wd)
                    summaries.append("=" * 90)
                    summaries.append(f"Storage: {storage_key.upper()} | Window: {label} | Factor: {key}")
                    summaries.append(wd.summary)

        # -------- Post-processor: pick cheapest options --------
        print("[post] Building candidate list for cheapest selection...")
        candidates = _collect_candidates(results)
        print(f"[post] Total candidates={len(candidates)}")

        cheapest_overall = _pick_cheapest(candidates)
        cheapest_by_storage = {}
        for sk in ["standard", "iopt1"]:
            cands_sk = [c for c in candidates if c["storage"] == sk]
            cheapest_by_storage[sk] = _pick_cheapest(cands_sk)

        top3_overall = _topk(candidates, 3)

        print("[post] Cheapest overall:", json.dumps(cheapest_overall, default=str))
        print("[post] Cheapest by storage:", json.dumps(cheapest_by_storage, default=str))
        print("[post] Top3 overall:", json.dumps(top3_overall, default=str))

        body = {
            "db_cluster_id": db_cluster_id,
            "region": REGION,
            "windows": WINDOWS,
            "factors": AAS_TO_ACU_FACTORS,
            "storage_comparison_mode": STORAGE_COMPARISON_MODE,
            "detected_storage_type": detected_storage or "unknown",
            "serverless_acu_per_hour": SERVERLESS_ACU_PER_HOUR,
            "serverless_iopt_multiplier": SERVERLESS_IOPT_MULTIPLIER,
            "io_price_per_million": IO_PRICE_PER_MILLION,
            "instance_hourly_prices": INSTANCE_HOURLY_PRICES,
            "io_optimized_multipliers_by_class": IO_OPTIMIZED_MULTIPLIERS_BY_CLASS,
            "charge_io_for_standard": CHARGE_IO_FOR_STANDARD,
            "results": results,
            "postprocessor": {
                "cheapest_overall": cheapest_overall,
                "cheapest_by_storage": cheapest_by_storage,
                "top3_overall": top3_overall,
            },
            "summary": "\n".join(summaries),
        }
        print("[lambda_handler] Done (with CPU-aware ACU + post-processed recommendations)")
        return {"statusCode": 200, "body": json.dumps(body, default=str)}
    except Exception as e:
        print(f"[lambda_handler] EXCEPTION: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
