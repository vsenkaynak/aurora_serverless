# lambda_function.py
"""
Aurora Serverless v2 vs Provisioned cost comparison (Python 3.12)

Changes in this version:
- Global DEBUG flag with log() wrapper (toggle all prints).
- Q1/Q2 (DBLoad/DBLoadCPU) queried only if Performance Insights is enabled on instances.
- Q3 (ServerlessDatabaseCapacity) always queried at cluster level (DBClusterIdentifier).
- CPU-only fallback when DBLoad/ACU are missing; unified chooser uses DBLoad p95 or CPU p95.
- Stable windows with -5min CW skew; structured JSON output with monthly_equivalent only.
- NEW: exclude all *.medium classes from candidates, sort by effective hourly cost, stop at first fit.
"""

import json
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import boto3

# ============== GLOBAL DEBUG TOGGLE ==============
DEBUG = True  # set False to silence all logs
def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

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

# AAS→ACU sensitivity factors (used when estimating ACU for provisioned)
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]

# Serverless v2 compute price (Standard storage)
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

# Normalize to a month for reporting
MONTH_HOURS = 730.0

# -------- Instance catalog (classes + vCPUs + prices) --------
CANDIDATE_CLASSES = [
    # R4 (legacy)
    "db.r4.large", "db.r4.xlarge", "db.r4.2xlarge", "db.r4.4xlarge", "db.r4.8xlarge", "db.r4.16xlarge",

    # R5
    "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge", "db.r5.4xlarge", "db.r5.12xlarge", "db.r5.24xlarge",

    # R6g (Graviton2)
    "db.r6g.large", "db.r6g.xlarge", "db.r6g.2xlarge", "db.r6g.4xlarge", "db.r6g.8xlarge",
    "db.r6g.12xlarge", "db.r6g.16xlarge",

    # R6i (Intel Ice Lake)
    "db.r6i.large", "db.r6i.xlarge", "db.r6i.2xlarge", "db.r6i.4xlarge", "db.r6i.8xlarge",
    "db.r6i.12xlarge", "db.r6i.16xlarge", "db.r6i.24xlarge", "db.r6i.32xlarge",

    # R7g (Graviton3)
    "db.r7g.large", "db.r7g.xlarge", "db.r7g.2xlarge", "db.r7g.4xlarge", "db.r7g.8xlarge",

    # X2g (Extended memory-optimized, Graviton2)
    "db.x2g.large", "db.x2g.xlarge", "db.x2g.2xlarge", "db.x2g.4xlarge", "db.x2g.8xlarge",
    "db.x2g.12xlarge", "db.x2g.16xlarge",

    # Z1d (High-frequency compute)
    "db.z1d.large", "db.z1d.xlarge", "db.z1d.2xlarge", "db.z1d.3xlarge", "db.z1d.6xlarge", "db.z1d.12xlarge",
]


INSTANCE_VCPU = {
    # R4
    "db.r4.large": 2, "db.r4.xlarge": 4, "db.r4.2xlarge": 8, "db.r4.4xlarge": 16,
    "db.r4.8xlarge": 32, "db.r4.16xlarge": 64,

    # R5
    "db.r5.large": 2, "db.r5.xlarge": 4, "db.r5.2xlarge": 8, "db.r5.4xlarge": 16,
    "db.r5.12xlarge": 48, "db.r5.24xlarge": 96,

    # R6g
    "db.r6g.large": 2, "db.r6g.xlarge": 4, "db.r6g.2xlarge": 8, "db.r6g.4xlarge": 16,
    "db.r6g.8xlarge": 32, "db.r6g.12xlarge": 48, "db.r6g.16xlarge": 64,

    # R6i
    "db.r6i.large": 2, "db.r6i.xlarge": 4, "db.r6i.2xlarge": 8, "db.r6i.4xlarge": 16,
    "db.r6i.8xlarge": 32, "db.r6i.12xlarge": 48, "db.r6i.16xlarge": 64,
    "db.r6i.24xlarge": 96, "db.r6i.32xlarge": 128,

    # R7g
    "db.r7g.large": 2, "db.r7g.xlarge": 4, "db.r7g.2xlarge": 8, "db.r7g.4xlarge": 16, "db.r7g.8xlarge": 32,

    # X2g (extended memory)
    "db.x2g.large": 2, "db.x2g.xlarge": 4, "db.x2g.2xlarge": 8, "db.x2g.4xlarge": 16,
    "db.x2g.8xlarge": 32, "db.x2g.12xlarge": 48, "db.x2g.16xlarge": 64,

    # Z1d (high-frequency compute)
    "db.z1d.large": 2, "db.z1d.xlarge": 4, "db.z1d.2xlarge": 8, "db.z1d.3xlarge": 12,
    "db.z1d.6xlarge": 24, "db.z1d.12xlarge": 48,
}

INSTANCE_HOURLY_PRICES = {
    # R4 (legacy — not supported with I/O-Optimized storage; keep only if you run old engine versions)
    "db.r4.large": 0.290, "db.r4.xlarge": 0.580, "db.r4.2xlarge": 1.160, "db.r4.4xlarge": 2.320,
    "db.r4.8xlarge": 4.640, "db.r4.16xlarge": 9.280,

    # R5
    "db.r5.large": 0.250, "db.r5.xlarge": 0.500, "db.r5.2xlarge": 1.000, "db.r5.4xlarge": 2.000,
    "db.r5.12xlarge": 6.000, "db.r5.24xlarge": 12.000,

    # R6g (Graviton2)
    "db.r6g.large": 0.246, "db.r6g.xlarge": 0.492, "db.r6g.2xlarge": 0.984, "db.r6g.4xlarge": 1.968,
    "db.r6g.8xlarge": 3.936, "db.r6g.12xlarge": 5.904, "db.r6g.16xlarge": 7.872,

    # R6i (Intel Ice Lake)
    "db.r6i.large": 0.270, "db.r6i.xlarge": 0.540, "db.r6i.2xlarge": 1.080, "db.r6i.4xlarge": 2.160,
    "db.r6i.8xlarge": 4.320, "db.r6i.12xlarge": 6.480, "db.r6i.16xlarge": 8.640,
    "db.r6i.24xlarge": 12.960, "db.r6i.32xlarge": 17.280,

    # R7g (Graviton3)
    "db.r7g.large": 0.252, "db.r7g.xlarge": 0.504, "db.r7g.2xlarge": 1.008, "db.r7g.4xlarge": 2.016,
    "db.r7g.8xlarge": 4.032,

    # X2g (extended memory, Graviton2) — starter values
    "db.x2g.large": 0.370, "db.x2g.xlarge": 0.740, "db.x2g.2xlarge": 1.480, "db.x2g.4xlarge": 2.960,
    "db.x2g.8xlarge": 5.920, "db.x2g.12xlarge": 8.880, "db.x2g.16xlarge": 11.840,

    # Z1d (high-frequency compute) — starter values
    "db.z1d.large": 0.310, "db.z1d.xlarge": 0.620, "db.z1d.2xlarge": 1.240, "db.z1d.3xlarge": 1.860,
    "db.z1d.6xlarge": 3.720, "db.z1d.12xlarge": 7.440,
}

# If I/O-Optimized (aurora-iopt1), multiply provisioned hourly prices by:
IO_OPTIMIZED_MULTIPLIERS_BY_CLASS = {
    # Family-level multipliers; override per-class if you have exacts
    "db.r4.large": 1.150, "db.r4.xlarge": 1.150, "db.r4.2xlarge": 1.150, "db.r4.4xlarge": 1.150,
    "db.r4.8xlarge": 1.150, "db.r4.16xlarge": 1.150,

    "db.r5.large": 1.150, "db.r5.xlarge": 1.150, "db.r5.2xlarge": 1.150, "db.r5.4xlarge": 1.150,
    "db.r5.12xlarge": 1.150, "db.r5.24xlarge": 1.150,

    "db.r6g.large": 1.170, "db.r6g.xlarge": 1.170, "db.r6g.2xlarge": 1.170, "db.r6g.4xlarge": 1.170,
    "db.r6g.8xlarge": 1.170, "db.r6g.12xlarge": 1.170, "db.r6g.16xlarge": 1.170,

    "db.r6i.large": 1.160, "db.r6i.xlarge": 1.160, "db.r6i.2xlarge": 1.160, "db.r6i.4xlarge": 1.160,
    "db.r6i.8xlarge": 1.160, "db.r6i.12xlarge": 1.160, "db.r6i.16xlarge": 1.160,
    "db.r6i.24xlarge": 1.160, "db.r6i.32xlarge": 1.160,

    "db.r7g.large": 1.167, "db.r7g.xlarge": 1.167, "db.r7g.2xlarge": 1.167, "db.r7g.4xlarge": 1.167,
    "db.r7g.8xlarge": 1.167,

    "db.x2g.large": 1.120, "db.x2g.xlarge": 1.120, "db.x2g.2xlarge": 1.120, "db.x2g.4xlarge": 1.120,
    "db.x2g.8xlarge": 1.120, "db.x2g.12xlarge": 1.120, "db.x2g.16xlarge": 1.120,

    "db.z1d.large": 1.150, "db.z1d.xlarge": 1.150, "db.z1d.2xlarge": 1.150, "db.z1d.3xlarge": 1.150,
    "db.z1d.6xlarge": 1.150, "db.z1d.12xlarge": 1.150,

    "default": 1.170,  # fallback
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
    monthly_equivalent: float  # normalized to 730h month

@dataclass
class ProvisionedOption:
    instance_class: str
    fits: bool
    monthly_equivalent: float  # normalized to 730h month

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
    start_time: str
    end_time: str

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

def hours_between(start: dt.datetime, end: dt.datetime) -> float:
    return (end - start).total_seconds() / 3600.0

def get_clients(region: str):
    log(f"[get_clients] region={region}")
    return boto3.client("rds", region_name=region), boto3.client("cloudwatch", region_name=region)

def describe_cluster(rds, db_identifier: str):
    log(f"[describe_cluster] DBClusterIdentifier={db_identifier}")
    resp = rds.describe_db_clusters(DBClusterIdentifier=db_identifier)
    cluster = resp["DBClusters"][0]
    insts = []
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            iid = m["DBInstanceIdentifier"]
            log(f"[describe_cluster] Describe DB instance id={iid}")
            insts.append(rds.describe_db_instances(DBInstanceIdentifier=iid)["DBInstances"][0])
    log(f"[describe_cluster] Instances found: {len(insts)}")
    return cluster, insts

def detect_mode(cluster: dict) -> str:
    if cluster.get("EngineMode") == "serverless": return "serverless-v1"
    if cluster.get("ServerlessV2ScalingConfiguration"): return "serverless-v2"
    return "provisioned"

def detect_storage_type(cluster: dict, insts: List[dict]) -> str:
    st = (cluster.get("StorageType") or "").lower()
    if st:
        log(f"[detect_storage_type] Cluster StorageType={st}")
        return st
    for di in insts:
        st_i = (di.get("StorageType") or "").lower()
        if st_i:
            log(f"[detect_storage_type] Instance StorageType={st_i} (used)")
            return st_i
    log("[detect_storage_type] StorageType unknown")
    return ""

def get_cluster_acu_limits(cluster: dict) -> Tuple[float, float]:
    cfg = cluster.get("ServerlessV2ScalingConfiguration") or {}
    minc = float(cfg.get("MinCapacity") or DEFAULT_MIN_ACU)
    maxc = float(cfg.get("MaxCapacity") or DEFAULT_MAX_ACU)
    log(f"[get_cluster_acu_limits] MinCapacity={minc}, MaxCapacity={maxc}")
    return minc, maxc

def snap_acu(val: float, min_acu: float, max_acu: float, step: float = ACU_STEP) -> float:
    if val <= 0: return min_acu
    return max(min_acu, min(max_acu, round(val / step) * step))

# ---------- Preflight probe (ListMetrics) ----------

def preflight_list_metrics(cw, cluster_id: str):
    def present(metric_name):
        paginator = cw.get_paginator("list_metrics")
        for page in paginator.paginate(
            Namespace="AWS/RDS",
            MetricName=metric_name,
            Dimensions=[{"Name":"DBClusterIdentifier"}]
        ):
            for m in page.get("Metrics", []):
                for d in m.get("Dimensions", []):
                    if d.get("Name") == "DBClusterIdentifier" and d.get("Value") == cluster_id:
                        return True
        return False

    res = {
        "DBLoad": present("DBLoad"),
        "DBLoadCPU": present("DBLoadCPU"),
        "ServerlessDatabaseCapacity": present("ServerlessDatabaseCapacity"),
    }
    log(f"[preflight_list_metrics] {cluster_id} → {res}")
    return res

def performance_insights_enabled(insts: List[dict]) -> bool:
    # PI is per-instance; treat as enabled if any member has PI enabled
    for di in insts:
        if di.get("PerformanceInsightsEnabled"):
            return True
    return False

# ---------- Fetch metrics (conditionally includes Q1/Q2; always Q3 at cluster) ----------

def fetch_metrics(cw, cluster_id: str, instance_ids: List[str], writer_instance_id: Optional[str],
                  start: dt.datetime, end: dt.datetime, period: int,
                  enable_q1_q2: bool) -> Dict[str, MetricSeries]:
    log(f"[fetch_metrics] cluster={cluster_id}, period={period}s, {start} → {end}, PI_metrics={enable_q1_q2}")

    def mquery(qid, metric, dims, stat):
        return {"Id": qid, "MetricStat": {"Metric": {"Namespace":"AWS/RDS","MetricName":metric,"Dimensions":dims},
                                          "Period": period, "Stat": stat}, "ReturnData": True}

    dims_cluster = [{"Name":"DBClusterIdentifier","Value":cluster_id}]
    queries = []
    if enable_q1_q2:
        queries += [
            mquery("q1","DBLoad",dims_cluster,"Average"),
            mquery("q2","DBLoadCPU",dims_cluster,"Average"),
        ]
    # Q3 always queried at cluster level; will be empty on non-SV2
    queries.append(mquery("q3","ServerlessDatabaseCapacity",dims_cluster,"Average"))

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
            mquery(f"ci_{qid+2}","CPUUtilization",dims_inst,"Average"),  # per-instance CPU for fallback
        ]
        qid += 3

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
    log(f"[fetch_metrics] datapoints={total_points}")
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
    log(f"[summarize_shape] AAS p95={shape.dbload_p95:.2f}, CPU p95={shape.cpu_p95:.1f}%, Conn p95={shape.connections_p95:.0f}")
    return shape

def infer_period_from_series(ts: List[dt.datetime]) -> int:
    return max(1, int((ts[1]-ts[0]).total_seconds())) if ts and len(ts) >= 2 else 60

# ---------- ACU estimation (with CPU-only fallback) ----------

def estimate_acu_seconds(cluster_mode: str, cluster: dict, metrics: Dict[str, MetricSeries],
                         factor: float, min_acu_default: float, max_acu_default: float,
                         writer_vcpu: int) -> Tuple[float, str]:
    # 1) Serverless v2 → use actual ACU
    if cluster_mode == "serverless-v2":
        sv2 = metrics.get("q3")
        minc, maxc = get_cluster_acu_limits(cluster)
        if sv2 and sv2.values:
            period = infer_period_from_series(sv2.timestamps)
            clamped = [max(minc, min(maxc, v or 0.0)) for v in sv2.values]
            acu_seconds = sum(v * period for v in clamped)
            log(f"[estimate_acu_seconds] Using ServerlessDatabaseCapacity; ACU-sec={acu_seconds:.0f}")
            return float(acu_seconds), f"Used ServerlessDatabaseCapacity (clamped {minc}-{maxc} ACU)"
        log("[estimate_acu_seconds] Serverless v2 but no ACU samples, falling back")

    # 2) Try DBLoad and writer CPU (normal path)
    dbload = metrics.get("q1")
    cpu_writer = metrics.get("q4")  # writer CPU
    base_series = dbload or cpu_writer or MetricSeries([], [])
    period = infer_period_from_series(base_series.timestamps)

    acu_from_dbload: List[float] = []
    if dbload and dbload.values:
        for v in dbload.values:
            raw = (v or 0.0) * factor
            acu_from_dbload.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))

    acu_from_cpu_writer: List[float] = []
    if USE_CPU_IN_ACU_ESTIMATE and cpu_writer and cpu_writer.values and writer_vcpu and writer_vcpu > 0:
        for cpu_pct in cpu_writer.values:
            u = (cpu_pct or 0.0) / 100.0
            raw = (u * writer_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
            acu_from_cpu_writer.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))

    # 2a) Blend if both present
    if acu_from_dbload and acu_from_cpu_writer:
        n = min(len(acu_from_dbload), len(acu_from_cpu_writer))
        if ACU_BLEND_MODE == "avg":
            est = [snap_acu(CPU_WEIGHT*acu_from_cpu_writer[i] + (1-CPU_WEIGHT)*acu_from_dbload[i],
                            min_acu_default, max_acu_default, ACU_STEP) for i in range(n)]
            acu_seconds = sum(a * period for a in est)
            log(f"[estimate_acu_seconds] Blend avg; ACU-sec={acu_seconds:.0f}")
            return float(acu_seconds), f"Estimated ACU from DBLoad×{factor:.2f} and CPU (avg)"
        else:
            est = [max(acu_from_dbload[i], acu_from_cpu_writer[i]) for i in range(n)]
            acu_seconds = sum(a * period for a in est)
            log(f"[estimate_acu_seconds] Blend max; ACU-sec={acu_seconds:.0f}")
            return float(acu_seconds), f"Estimated ACU from DBLoad×{factor:.2f} and CPU (max)"

    # 3) If DBLoad missing and writer CPU unavailable, fall back to max CPU across all instances
    if not acu_from_dbload and not acu_from_cpu_writer:
        per_inst_cpu = [series for k, series in metrics.items() if k.startswith("ci_") and series.values]
        if per_inst_cpu:
            min_len = min(len(s.values) for s in per_inst_cpu)
            effective_vcpu = writer_vcpu if writer_vcpu > 0 else 4
            acu_from_cpu_all: List[float] = []
            for i in range(min_len):
                cpu_pct = max((s.values[i] or 0.0) for s in per_inst_cpu)
                u = (cpu_pct or 0.0) / 100.0
                raw = (u * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
                acu_from_cpu_all.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))
            acu_seconds = sum(a * period for a in acu_from_cpu_all)
            log(f"[estimate_acu_seconds] CPU-only fallback (max across instances); ACU-sec={acu_seconds:.0f}")
            return float(acu_seconds), f"CPU-only fallback (max across instances, vCPU={effective_vcpu})"

    # 4) Single-path fallbacks
    if acu_from_dbload:
        acu_seconds = sum(a * period for a in acu_from_dbload)
        log(f"[estimate_acu_seconds] DBLoad-only; ACU-sec={acu_seconds:.0f}")
        return float(acu_seconds), f"Estimated ACU from DBLoad×{factor:.2f}"
    if acu_from_cpu_writer:
        acu_seconds = sum(a * period for a in acu_from_cpu_writer)
        log(f"[estimate_acu_seconds] CPU-only (writer); ACU-sec={acu_seconds:.0f}")
        return float(acu_seconds), "CPU-only (writer)"

    # 5) Nothing available
    log("[estimate_acu_seconds] No DBLoad/ACU/CPU data; ACU=0")
    return 0.0, "No DBLoad/ACU/CPU data; ACU=0"

# ---------- I/O requests integration ----------

def estimate_total_io_requests(metrics: Dict[str, MetricSeries]) -> float:
    total_requests = 0.0
    for key, series in metrics.items():
        if key.startswith("ri_") or key.startswith("wi_"):
            period = infer_period_from_series(series.timestamps)
            total_requests += sum((v or 0.0) * period for v in series.values)
    log(f"[estimate_total_io_requests] total_requests={total_requests:.0f}")
    return total_requests

# ---------- Month-normalized cost calculators ----------

def serverless_monthly_equivalent(acu_seconds: float, storage_type: str,
                                  io_requests: float, window_hours: float) -> float:
    acu_hours = acu_seconds / 3600.0
    acu_price = SERVERLESS_ACU_PER_HOUR * (SERVERLESS_IOPT_MULTIPLIER if storage_type == "aurora-iopt1" else 1.0)
    compute_total = acu_hours * acu_price
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    window_total = compute_total + io_total
    monthly_equiv = window_total * (MONTH_HOURS / max(1e-6, window_hours))
    log(f"[serverless_monthly_equivalent] monthly_eq={monthly_equiv:.4f}")
    return monthly_equiv

def get_io_multiplier(storage_type: str, instance_class: str) -> float:
    if storage_type == "aurora-iopt1":
        return IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get(instance_class, IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get("default", 1.17))
    return 1.0

def provisioned_monthly_equivalent(compute_price_per_hour: float, hours: float, instance_count: int,
                                   storage_type: str, io_requests: float) -> float:
    compute_total = compute_price_per_hour * hours * instance_count
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    window_total = compute_total + io_total
    monthly_equiv = window_total * (MONTH_HOURS / max(1e-6, hours))
    log(f"[provisioned_monthly_equivalent] monthly_eq={monthly_equiv:.4f}")
    return monthly_equiv

# ---------- CPU-based fallback sizing helpers ----------

def compute_effective_vcpu(current_instance_classes: List[str], writer_class: Optional[str]) -> int:
    if writer_class:
        v = INSTANCE_VCPU.get(writer_class, 0)
        if v > 0:
            return v
    max_v = 0
    for ic in current_instance_classes or []:
        max_v = max(max_v, INSTANCE_VCPU.get(ic, 0))
    return max_v if max_v > 0 else 4

def cpu_p95_to_required_acu(cpu_p95: float, effective_vcpu: int) -> float:
    if effective_vcpu <= 0 or cpu_p95 <= 0:
        return 0.0
    utilization = cpu_p95 / 100.0
    raw_acu = (utilization * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
    return snap_acu(raw_acu, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, ACU_STEP)

# ---------- Window evaluation ----------

def evaluate_one_window_with_storage(*, label: str, factor: float,
                                     db_identifier: str, region: str,
                                     lookback_days: int, period_seconds: int,
                                     storage_type_used: str,
                                     detected_storage_type: str,
                                     cluster: dict,
                                     start_time: Optional[dt.datetime],
                                     end_time: Optional[dt.datetime]) -> WindowDecision:
    log(f"\n=== [evaluate_one_window] {label} factor={factor:.2f} storage={storage_type_used} ===")
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
    writer_class = None
    if writer_id:
        for di in insts:
            if di.get("DBInstanceIdentifier") == writer_id:
                writer_class = di.get("DBInstanceClass")
                break

    # Stable times (passed in from handler)
    end = end_time or now_utc()
    start = start_time or (end - dt.timedelta(days=lookback_days))
    hours = hours_between(start, end)
    log(f"[evaluate_one_window] window={start.isoformat()} → {end.isoformat()} (hrs={hours:.2f})")

    # Preflight probe (logs presence)
    _ = preflight_list_metrics(cw, db_identifier)

    # Determine if Performance Insights metrics should be queried
    pi_enabled = performance_insights_enabled(insts)
    log(f"[evaluate_one_window] PerformanceInsightsEnabled(any)={pi_enabled}")

    # Fetch metrics (conditionally include Q1/Q2; always include Q3 at cluster)
    metrics = fetch_metrics(
        cw, cluster_id=db_identifier, instance_ids=instance_ids, writer_instance_id=writer_id,
        start=start, end=end, period=period_seconds, enable_q1_q2=pi_enabled
    )

    shape = summarize_shape(metrics)
    inst_count = max(1, len(instance_ids))

    # ACU usage
    writer_vcpu = INSTANCE_VCPU.get(writer_class or "", 0)
    acu_seconds, _ = estimate_acu_seconds(mode, cluster, metrics, factor,
                                          DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, writer_vcpu)
    io_requests = estimate_total_io_requests(metrics)

    # Serverless cost
    sv2_monthly = serverless_monthly_equivalent(acu_seconds, storage_type_used, io_requests, hours)
    sv2_cost = CostBreakdown(monthly_equivalent=sv2_monthly)

    # -------- Provisioned selection: cost-sorted, stop at first fit; exclude *.medium --------
    effective_vcpu = compute_effective_vcpu(current_classes, writer_class)

    # Build cost-sorted candidates (exclude *.medium)
    sorted_cands: List[Tuple[str, float]] = []
    for ic, base_price in INSTANCE_HOURLY_PRICES.items():
        if ic.endswith(".medium"):  # exclude .medium
            continue
        if ic not in CANDIDATE_CLASSES:
            continue
        mult = get_io_multiplier(storage_type_used, ic)
        eff_price = base_price * mult
        sorted_cands.append((ic, eff_price))
    sorted_cands.sort(key=lambda x: x[1])

    # Determine sizing basis
    basis = "AAS" if shape.dbload_p95 > 0 else ("CPU" if shape.cpu_p95 > 0 else "NONE")

    best_prov: Optional[ProvisionedOption] = None
    for ic, eff_price_per_hour in sorted_cands:
        vcpu = INSTANCE_VCPU.get(ic, 0)
        if basis == "AAS":
            fits = (shape.dbload_p95 <= AAS_HEADROOM * vcpu)
        elif basis == "CPU":
            req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
            fits = (req_acu <= AAS_HEADROOM * vcpu)
        else:
            fits = False

        monthly_eq = provisioned_monthly_equivalent(eff_price_per_hour, hours, inst_count, storage_type_used, io_requests)
        if fits:
            best_prov = ProvisionedOption(ic, True, monthly_eq)
            log(f"[provisioned_select] basis={basis}, first_fit={ic}, hourly={eff_price_per_hour:.4f}, monthly_eq={monthly_eq:.4f}")
            break  # stop at first (cheapest-by-cost) fit

    # Current class (if provisioned)
    current_prov = None
    if current_classes:
        curr = current_classes[0]
        if curr in INSTANCE_HOURLY_PRICES:
            mult = get_io_multiplier(storage_type_used, curr)
            compute_price = INSTANCE_HOURLY_PRICES[curr] * mult
            monthly_eq = provisioned_monthly_equivalent(compute_price, hours, inst_count, storage_type_used, io_requests)
            fits = False
            if basis == "AAS" and shape.dbload_p95 > 0:
                fits = (shape.dbload_p95 <= AAS_HEADROOM * INSTANCE_VCPU.get(curr, 0))
            elif basis == "CPU" and shape.cpu_p95 > 0:
                req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
                fits = (req_acu <= AAS_HEADROOM * INSTANCE_VCPU.get(curr, 0))
            current_prov = ProvisionedOption(curr, fits, monthly_eq)

    wd = WindowDecision(
        label=label, factor=factor,
        lookback_days=lookback_days, period_seconds=period_seconds,
        db_identifier=db_identifier, engine=engine, mode_detected=mode,
        storage_type_used=storage_type_used, io_optimized=(storage_type_used == "aurora-iopt1"),
        instances_found=instance_ids, current_instance_classes=current_classes,
        shape=shape, serverless_cost=sv2_cost, provisioned_best=best_prov,
        provisioned_current=current_prov,
        start_time=start.isoformat(), end_time=end.isoformat()
    )
    log("[evaluate_one_window] compact:", {
        "label": label, "factor": factor, "storage": storage_type_used,
        "sv2_monthly_equiv": sv2_cost.monthly_equivalent,
        "best_prov": (best_prov.instance_class, best_prov.monthly_equivalent) if best_prov else None,
        "current_prov": (current_prov.instance_class, current_prov.monthly_equivalent) if current_prov else None,
        "basis": basis
    })
    return wd

# ---------- Post-processor helpers (use monthly_equivalent) ----------

def _collect_candidates(flat_results: Dict) -> List[Dict]:
    out: List[Dict] = []
    for storage_key, win_map in flat_results.items():
        for window_label, factor_map in win_map.items():
            for factor_str, wd in factor_map.items():
                sv = wd.get("serverless_cost") or {}
                if "monthly_equivalent" in sv:
                    out.append({
                        "storage": storage_key,
                        "window": window_label,
                        "factor": factor_str,
                        "type": "serverless",
                        "instance_class": None,
                        "monthly_equivalent": sv["monthly_equivalent"],
                    })
                bp = wd.get("provisioned_best")
                if bp and "monthly_equivalent" in bp:
                    out.append({
                        "storage": storage_key,
                        "window": window_label,
                        "factor": factor_str,
                        "type": "provisioned",
                        "instance_class": bp.get("instance_class"),
                        "monthly_equivalent": bp["monthly_equivalent"],
                    })
    return out

def _pick_cheapest(cands: List[Dict]) -> Optional[Dict]:
    return min(cands, key=lambda c: c["monthly_equivalent"]) if cands else None

def _topk(cands: List[Dict], k: int) -> List[Dict]:
    return sorted(cands, key=lambda c: c["monthly_equivalent"])[:k]

# -------------- Lambda entry ----------------

 def lambda_handler(event, context):
     try:
        # Strict input validation: require exactly {"cluster_id": "<DBClusterIdentifier>"}
        if not isinstance(event, dict) or set(event.keys()) != {"cluster_id"} or not event["cluster_id"]:
            msg = "Invalid input. Payload must be exactly {'cluster_id': '<DBClusterIdentifier>'}."
            log(f"[lambda_handler] ERROR: {msg} event={event!r}")
            return {"statusCode": 400, "body": json.dumps({"error": msg})}

        log(f"[lambda_handler] event={json.dumps(event)}")
        db_cluster_id = event["cluster_id"]

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
            log(f"[lambda_handler] WARNING: unknown STORAGE_COMPARISON_MODE={STORAGE_COMPARISON_MODE}, defaulting to 'auto'")
            modes_to_run = [detected_storage or "aurora"]

        log(f"[lambda_handler] DetectedStorage={detected_storage or 'unknown'}, ComparisonModes={modes_to_run}")

        results: Dict[str, Dict[str, Dict[str, dict]]] = {}

        for storage_mode in modes_to_run:
            storage_key = "standard" if storage_mode == "aurora" else "iopt1"
            results[storage_key] = {}
            for label, days, period in WINDOWS:
                results[storage_key][label] = {}

                # Freeze window timestamps once per (storage_mode, window) with a -5min skew to avoid CW lag
                end_ts = now_utc() - dt.timedelta(minutes=5)
                start_ts = end_ts - dt.timedelta(days=days)
                log(f"[lambda_handler] Frozen window for {storage_key}/{label}: {start_ts.isoformat()} → {end_ts.isoformat()}")

                for factor in AAS_TO_ACU_FACTORS:
                    wd = evaluate_one_window_with_storage(
                        label=label, factor=factor,
                        db_identifier=db_cluster_id, region=REGION,
                        lookback_days=days, period_seconds=period,
                        storage_type_used=storage_mode, detected_storage_type=detected_storage,
                        cluster=cluster,
                        start_time=start_ts, end_time=end_ts
                    )
                    key = f"{factor:.2f}"
                    results[storage_key][label][key] = asdict(wd)

        # -------- Post-processor: pick cheapest options (by monthly_equivalent) --------
        candidates = _collect_candidates(results)
        cheapest_overall = _pick_cheapest(candidates)
        cheapest_by_storage = {}
        for sk in ["standard", "iopt1"]:
            cands_sk = [c for c in candidates if c["storage"] == sk]
            cheapest_by_storage[sk] = _pick_cheapest(cands_sk)
        top3_overall = _topk(candidates, 3)

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
            }
        }
        return {"statusCode": 200, "body": json.dumps(body, default=str)}
    except Exception as e:
        log(f"[lambda_handler] EXCEPTION: {e}")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
