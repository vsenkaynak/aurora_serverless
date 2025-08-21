# cpu_spike_aware.py
"""
Aurora Serverless v2 vs Provisioned cost comparison (CPU-only coverage)
Python 3.12 / AWS Lambda

Key features:
- Strict input: payload must be exactly {"cluster_id": "<DBClusterIdentifier>"}.
- Three analysis windows (≈14d, 30d, 60d) with fixed periods (60s/120s/300s).
- CPU-only coverage policy: compute expected vs actual CPU samples; skip window if < 50% coverage.
- If skipped, WindowDecision carries coverage diagnostics & skip reason.
- Metric preference for cost/fit:
    1) Serverless v2 ACU series (q3) when present (cluster metric)
    2) DBLoad (q1) if PI enabled
    3) CPU fallback (q4 writer, else max across instances)
- Spike-aware headroom (revised): lowers utilization target when p99 >> p95.
- Provisioned selection: candidates sorted by effective hourly price; stop at first fit.
- Storage modes: Standard ("aurora") vs I/O-Optimized ("aurora-iopt1"):
    * I/O-Optimized: compute price multiplier; no per-I/O charge for either serverless or provisioned.
    * Standard: no compute multiplier; I/O charge added for both serverless and provisioned.
- Costs reported as monthly_equivalent only (normalized to 730 hours).
- DEBUG toggle to silence/enable logs.
"""

import json
import math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import boto3

# ========================
# Global Debug Toggle
# ========================
DEBUG = True  # set False to silence logs
def log(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# ========================
# Configurable Values
# ========================
REGION = "us-east-1"

# Storage comparison mode:
#   "auto"     → use detected storage type if available, else assume "aurora"
#   "standard" → force Standard ("aurora")
#   "iopt1"    → force I/O-Optimized ("aurora-iopt1")
#   "both"     → evaluate both Standard and I/O-Optimized
STORAGE_COMPARISON_MODE = "both"

# Windows: (label, lookback_days, period_seconds)
WINDOWS: List[Tuple[str, int, int]] = [
    ("14d_60s", 14, 60),
    ("30d_120s", 30, 120),
    ("60d_300s", 60, 300),
]

# Convert Aurora AAS→ACU when DBLoad is available (sensitivity sweep)
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]

# Serverless v2 compute price (Standard)
SERVERLESS_ACU_PER_HOUR = 0.12  # USD per ACU-hour (Standard)
# If I/O-Optimized (aurora-iopt1), ACU price multiplier (starter value)
SERVERLESS_IOPT_MULTIPLIER = 1.17

# I/O pricing for Standard (charged for both serverless & provisioned when on Standard)
IO_PRICE_PER_MILLION = 0.20  # USD per 1,000,000 requests
CHARGE_IO_FOR_STANDARD = True

# ACU step/limits
DEFAULT_MIN_ACU = 0.5
DEFAULT_MAX_ACU = 64.0
ACU_STEP = 0.5

# CPU→ACU conversion (fallback path)
USE_CPU_IN_ACU_ESTIMATE = True
CPU_TO_ACU_HEADROOM = 0.70  # keep buffer when mapping CPU→ACU

# Spike-aware headroom (REVISED: lower utilization target when spiky)
SPIKE_AWARE_HEADROOM = True
HEADROOM_BASE = 0.70     # normal utilization target (≈70% of capacity)
HEADROOM_MIN  = 0.50     # do not go below 50%
HEADROOM_MAX  = 0.80     # sanity upper cap if raised elsewhere
SPIKE_RATIO_THRESHOLD      = 1.30   # treat as spiky when p99 >= 1.30 * p95
HEADROOM_DROP_AT_THRESHOLD = 0.05   # drop by 5 points right above threshold
HEADROOM_DROP_MAX          = 0.15   # max drop by 15 points (e.g., 0.70 → 0.55)

# Window normalization
MONTH_HOURS = 730.0

# ---------- CPU-only coverage policy ----------
COVERAGE_REQUIRED = 0.50   # require ≥ 50% CPU datapoints vs expected
SKIP_ON_LOW_COVERAGE = True

# ---------- Candidate families/classes (no .medium, no t2/t3) ----------
CANDIDATE_CLASSES: List[str] = [
    # R4 (legacy)
    "db.r4.large", "db.r4.xlarge", "db.r4.2xlarge", "db.r4.4xlarge", "db.r4.8xlarge", "db.r4.16xlarge",
    # R5
    "db.r5.large", "db.r5.xlarge", "db.r5.2xlarge", "db.r5.4xlarge", "db.r5.12xlarge", "db.r5.24xlarge",
    # R6g (Graviton2)
    "db.r6g.large", "db.r6g.xlarge", "db.r6g.2xlarge", "db.r6g.4xlarge", "db.r6g.8xlarge", "db.r6g.12xlarge", "db.r6g.16xlarge",
    # R6i (Intel Ice Lake)
    "db.r6i.large", "db.r6i.xlarge", "db.r6i.2xlarge", "db.r6i.4xlarge", "db.r6i.8xlarge", "db.r6i.12xlarge", "db.r6i.16xlarge", "db.r6i.24xlarge", "db.r6i.32xlarge",
    # R7g (Graviton3)
    "db.r7g.large", "db.r7g.xlarge", "db.r7g.2xlarge", "db.r7g.4xlarge", "db.r7g.8xlarge",
    # X2g (extended memory, Graviton2)
    "db.x2g.large", "db.x2g.xlarge", "db.x2g.2xlarge", "db.x2g.4xlarge", "db.x2g.8xlarge", "db.x2g.12xlarge", "db.x2g.16xlarge",
    # Z1d (high-frequency compute)
    "db.z1d.large", "db.z1d.xlarge", "db.z1d.2xlarge", "db.z1d.3xlarge", "db.z1d.6xlarge", "db.z1d.12xlarge",
]

# vCPU mapping
INSTANCE_VCPU: Dict[str, int] = {
    # R4
    "db.r4.large": 2, "db.r4.xlarge": 4, "db.r4.2xlarge": 8, "db.r4.4xlarge": 16, "db.r4.8xlarge": 32, "db.r4.16xlarge": 64,
    # R5
    "db.r5.large": 2, "db.r5.xlarge": 4, "db.r5.2xlarge": 8, "db.r5.4xlarge": 16, "db.r5.12xlarge": 48, "db.r5.24xlarge": 96,
    # R6g
    "db.r6g.large": 2, "db.r6g.xlarge": 4, "db.r6g.2xlarge": 8, "db.r6g.4xlarge": 16, "db.r6g.8xlarge": 32, "db.r6g.12xlarge": 48, "db.r6g.16xlarge": 64,
    # R6i
    "db.r6i.large": 2, "db.r6i.xlarge": 4, "db.r6i.2xlarge": 8, "db.r6i.4xlarge": 16, "db.r6i.8xlarge": 32, "db.r6i.12xlarge": 48, "db.r6i.16xlarge": 64, "db.r6i.24xlarge": 96, "db.r6i.32xlarge": 128,
    # R7g
    "db.r7g.large": 2, "db.r7g.xlarge": 4, "db.r7g.2xlarge": 8, "db.r7g.4xlarge": 16, "db.r7g.8xlarge": 32,
    # X2g
    "db.x2g.large": 2, "db.x2g.xlarge": 4, "db.x2g.2xlarge": 8, "db.x2g.4xlarge": 16, "db.x2g.8xlarge": 32, "db.x2g.12xlarge": 48, "db.x2g.16xlarge": 64,
    # Z1d
    "db.z1d.large": 2, "db.z1d.xlarge": 4, "db.z1d.2xlarge": 8, "db.z1d.3xlarge": 12, "db.z1d.6xlarge": 24, "db.z1d.12xlarge": 48,
}

# Starter hourly prices (USD/hour, us-east-1) — replace with exact static prices when ready
INSTANCE_HOURLY_PRICES: Dict[str, float] = {
    # R4 (legacy)
    "db.r4.large": 0.290, "db.r4.xlarge": 0.580, "db.r4.2xlarge": 1.160, "db.r4.4xlarge": 2.320, "db.r4.8xlarge": 4.640, "db.r4.16xlarge": 9.280,
    # R5
    "db.r5.large": 0.250, "db.r5.xlarge": 0.500, "db.r5.2xlarge": 1.000, "db.r5.4xlarge": 2.000, "db.r5.12xlarge": 6.000, "db.r5.24xlarge": 12.000,
    # R6g
    "db.r6g.large": 0.246, "db.r6g.xlarge": 0.492, "db.r6g.2xlarge": 0.984, "db.r6g.4xlarge": 1.968, "db.r6g.8xlarge": 3.936, "db.r6g.12xlarge": 5.904, "db.r6g.16xlarge": 7.872,
    # R6i
    "db.r6i.large": 0.270, "db.r6i.xlarge": 0.540, "db.r6i.2xlarge": 1.080, "db.r6i.4xlarge": 2.160, "db.r6i.8xlarge": 4.320, "db.r6i.12xlarge": 6.480, "db.r6i.16xlarge": 8.640, "db.r6i.24xlarge": 12.960, "db.r6i.32xlarge": 17.280,
    # R7g
    "db.r7g.large": 0.252, "db.r7g.xlarge": 0.504, "db.r7g.2xlarge": 1.008, "db.r7g.4xlarge": 2.016, "db.r7g.8xlarge": 4.032,
    # X2g
    "db.x2g.large": 0.370, "db.x2g.xlarge": 0.740, "db.x2g.2xlarge": 1.480, "db.x2g.4xlarge": 2.960, "db.x2g.8xlarge": 5.920, "db.x2g.12xlarge": 8.880, "db.x2g.16xlarge": 11.840,
    # Z1d
    "db.z1d.large": 0.310, "db.z1d.xlarge": 0.620, "db.z1d.2xlarge": 1.240, "db.z1d.3xlarge": 1.860, "db.z1d.6xlarge": 3.720, "db.z1d.12xlarge": 7.440,
}

# I/O-Optimized multipliers for provisioned compute (starter values)
IO_OPTIMIZED_MULTIPLIERS_BY_CLASS: Dict[str, float] = {
    "db.r4.large": 1.150, "db.r4.xlarge": 1.150, "db.r4.2xlarge": 1.150, "db.r4.4xlarge": 1.150, "db.r4.8xlarge": 1.150, "db.r4.16xlarge": 1.150,
    "db.r5.large": 1.150, "db.r5.xlarge": 1.150, "db.r5.2xlarge": 1.150, "db.r5.4xlarge": 1.150, "db.r5.12xlarge": 1.150, "db.r5.24xlarge": 1.150,
    "db.r6g.large": 1.170, "db.r6g.xlarge": 1.170, "db.r6g.2xlarge": 1.170, "db.r6g.4xlarge": 1.170, "db.r6g.8xlarge": 1.170, "db.r6g.12xlarge": 1.170, "db.r6g.16xlarge": 1.170,
    "db.r6i.large": 1.160, "db.r6i.xlarge": 1.160, "db.r6i.2xlarge": 1.160, "db.r6i.4xlarge": 1.160, "db.r6i.8xlarge": 1.160, "db.r6i.12xlarge": 1.160, "db.r6i.16xlarge": 1.160, "db.r6i.24xlarge": 1.160, "db.r6i.32xlarge": 1.160,
    "db.r7g.large": 1.167, "db.r7g.xlarge": 1.167, "db.r7g.2xlarge": 1.167, "db.r7g.4xlarge": 1.167, "db.r7g.8xlarge": 1.167,
    "db.x2g.large": 1.120, "db.x2g.xlarge": 1.120, "db.x2g.2xlarge": 1.120, "db.x2g.4xlarge": 1.120, "db.x2g.8xlarge": 1.120, "db.x2g.12xlarge": 1.120, "db.x2g.16xlarge": 1.120,
    "db.z1d.large": 1.150, "db.z1d.xlarge": 1.150, "db.z1d.2xlarge": 1.150, "db.z1d.3xlarge": 1.150, "db.z1d.6xlarge": 1.150, "db.z1d.12xlarge": 1.150,
    "default": 1.170,
}

# Retained for reference; dynamic headroom uses compute_effective_headroom()
AAS_HEADROOM = HEADROOM_BASE

# =========================
# Data Models
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
    monthly_equivalent: float

@dataclass
class ProvisionedOption:
    instance_class: str
    fits: bool
    monthly_equivalent: float

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
    headroom_effective: float
    start_time: str
    end_time: str

    # Coverage / age / skip diagnostics (CPU-only coverage)
    expected_points: int
    actual_points: int
    coverage_ratio: float
    coverage_required: float
    earliest_sample_time: Optional[str]
    db_create_time: Optional[str]
    db_age_days: Optional[float]
    skipped: bool
    skip_reason: Optional[str]

# =========================
# Helpers
# =========================
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
    return boto3.client("rds", region_name=region), boto3.client("cloudwatch", region_name=region)

def describe_cluster(rds, db_identifier: str):
    resp = rds.describe_db_clusters(DBClusterIdentifier=db_identifier)
    cluster = resp["DBClusters"][0]
    insts = []
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            iid = m["DBInstanceIdentifier"]
            insts.append(rds.describe_db_instances(DBInstanceIdentifier=iid)["DBInstances"][0])
    return cluster, insts

def detect_mode(cluster: dict) -> str:
    if cluster.get("EngineMode") == "serverless":
        return "serverless-v1"
    if cluster.get("ServerlessV2ScalingConfiguration"):
        return "serverless-v2"
    return "provisioned"

def detect_storage_type(cluster: dict, insts: List[dict]) -> str:
    st = (cluster.get("StorageType") or "").lower()
    if st:
        return st
    for di in insts:
        st_i = (di.get("StorageType") or "").lower()
        if st_i:
            return st_i
    return ""

def get_cluster_acu_limits(cluster: dict) -> Tuple[float, float]:
    cfg = cluster.get("ServerlessV2ScalingConfiguration") or {}
    minc = float(cfg.get("MinCapacity") or DEFAULT_MIN_ACU)
    maxc = float(cfg.get("MaxCapacity") or DEFAULT_MAX_ACU)
    return minc, maxc

def snap_acu(val: float, min_acu: float, max_acu: float, step: float = ACU_STEP) -> float:
    if val <= 0: return min_acu
    return max(min_acu, min(max_acu, round(val / step) * step))

# ---------- Coverage helpers (CPU-only) ----------
def expected_datapoints_for_window(start: dt.datetime, end: dt.datetime, period_seconds: int) -> int:
    total_secs = max(0, int((end - start).total_seconds()))
    return 1 + total_secs // max(1, period_seconds)

def coverage_ratio_from_cpu_series(series: Optional['MetricSeries'], start: dt.datetime, end: dt.datetime, period_seconds: int) -> float:
    if not series or not series.values:
        return 0.0
    expected = expected_datapoints_for_window(start, end, period_seconds)
    actual = len(series.values)
    return (actual / expected) if expected > 0 else 0.0

def earliest_timestamp(series: Optional['MetricSeries']) -> Optional[dt.datetime]:
    if not series or not series.timestamps:
        return None
    return min(series.timestamps)

# ---------- Metric fetch ----------
def fetch_metrics(cw, cluster_id: str, instance_ids: List[str], writer_instance_id: Optional[str],
                  start: dt.datetime, end: dt.datetime, period: int,
                  enable_q1_q2: bool) -> Dict[str, 'MetricSeries']:

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
            mquery(f"ci_{qid+2}","CPUUtilization",dims_inst,"Average"),
        ]
        qid += 3

    out: Dict[str, MetricSeries] = {}
    next_token = None
    while True:
        args = {"StartTime":start, "EndTime":end, "MetricDataQueries":queries,
                "ScanBy":"TimestampAscending", "MaxDatapoints":50000}
        if next_token: args["NextToken"] = next_token
        resp = cw.get_metric_data(**args)
        for r in resp.get("MetricDataResults", []):
            name = r["Id"]
            ts, vals = r.get("Timestamps", []) or [], r.get("Values", []) or []
            if name in out:
                out[name].timestamps.extend(ts); out[name].values.extend(vals)
            else:
                out[name] = MetricSeries(ts, vals)
        next_token = resp.get("NextToken")
        if not next_token: break

    # sort by timestamp asc
    for k, s in out.items():
        if s.timestamps:
            pair = sorted(zip(s.timestamps, s.values), key=lambda x: x[0])
            out[k] = MetricSeries([p[0] for p in pair], [p[1] for p in pair])
    return out

def summarize_shape(metrics: Dict[str, MetricSeries]) -> WorkloadShape:
    dbload = metrics.get("q1").values if metrics.get("q1") else []
    cpu = metrics.get("q4").values if metrics.get("q4") else []
    conn = metrics.get("q5").values if metrics.get("q5") else []
    return WorkloadShape(
        dbload_p50=percentile(dbload, 50.0) if dbload else 0.0,
        dbload_p95=percentile(dbload, 95.0) if dbload else 0.0,
        dbload_p99=percentile(dbload, 99.0) if dbload else 0.0,
        cpu_p50=percentile(cpu, 50.0) if cpu else 0.0,
        cpu_p95=percentile(cpu, 95.0) if cpu else 0.0,
        cpu_p99=percentile(cpu, 99.0) if cpu else 0.0,
        connections_p95=percentile(conn, 95.0) if conn else 0.0,
    )

def infer_period_from_series(ts: List[dt.datetime]) -> int:
    if not ts or len(ts) < 2:
        return 60
    deltas = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts)-1)]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        return 60
    deltas.sort()
    mid = len(deltas) // 2
    median = deltas[mid] if len(deltas) % 2 == 1 else (deltas[mid-1] + deltas[mid]) / 2.0
    return max(1, int(round(median)))

# ---------- CPU-based helpers ----------
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

# ---------- Spike-aware headroom (revised: lower target when spiky) ----------
def compute_effective_headroom(shape: WorkloadShape) -> float:
    h = HEADROOM_BASE
    if not SPIKE_AWARE_HEADROOM:
        return max(HEADROOM_MIN, min(HEADROOM_MAX, h))

    def lowered(h_current: float, p95: float, p99: float) -> float:
        if p95 <= 0 or p99 <= 0:
            return h_current
        ratio = p99 / max(1e-6, p95)
        if ratio < SPIKE_RATIO_THRESHOLD:
            return h_current
        capped_ratio = min(ratio, 2.0)
        ratio_excess = capped_ratio - SPIKE_RATIO_THRESHOLD
        span = max(1e-6, 2.0 - SPIKE_RATIO_THRESHOLD)
        frac = ratio_excess / span
        drop = HEADROOM_DROP_AT_THRESHOLD + (HEADROOM_DROP_MAX - HEADROOM_DROP_AT_THRESHOLD) * frac
        return max(HEADROOM_MIN, h_current - drop)

    if shape.dbload_p95 > 0 and shape.dbload_p99 > 0:
        h = lowered(h, shape.dbload_p95, shape.dbload_p99)
    elif shape.cpu_p95 > 0 and shape.cpu_p99 > 0:
        h = lowered(h, shape.cpu_p95, shape.cpu_p99)

    return max(HEADROOM_MIN, min(HEADROOM_MAX, h))

# ---------- ACU estimation (uses ACU when present; else DBLoad; else CPU) ----------
def estimate_acu_seconds(cluster_mode: str, cluster: dict, metrics: Dict[str, MetricSeries],
                         factor: float, min_acu_default: float, max_acu_default: float,
                         writer_vcpu: int) -> Tuple[float, str, Optional[MetricSeries]]:
    # Prefer real ACU for serverless-v2
    if cluster_mode == "serverless-v2":
        sv2 = metrics.get("q3")
        minc, maxc = get_cluster_acu_limits(cluster)
        if sv2 and sv2.values:
            period = infer_period_from_series(sv2.timestamps)
            clamped = [max(minc, min(maxc, v or 0.0)) for v in sv2.values]
            acu_seconds = sum(v * period for v in clamped)
            return float(acu_seconds), f"Used ServerlessDatabaseCapacity (clamped {minc}-{maxc} ACU)", sv2

    # Next, DBLoad if present
    dbload = metrics.get("q1")
    cpu_writer = metrics.get("q4")

    period = infer_period_from_series((dbload or cpu_writer or MetricSeries([], [])).timestamps)

    acu_from_dbload: List[float] = []
    if dbload and dbload.values:
        for v in dbload.values:
            raw = (v or 0.0) * factor
            acu_from_dbload.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))

    acu_from_cpu_writer: List[float] = []
    if USE_CPU_IN_ACU_ESTIMATE and cpu_writer and cpu_writer.values and writer_vcpu > 0:
        for cpu_pct in cpu_writer.values:
            u = (cpu_pct or 0.0) / 100.0
            raw = (u * writer_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
            acu_from_cpu_writer.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))

    if acu_from_dbload and acu_from_cpu_writer:
        n = min(len(acu_from_dbload), len(acu_from_cpu_writer))
        blended = [max(acu_from_dbload[i], acu_from_cpu_writer[i]) for i in range(n)]
        return float(sum(a * period for a in blended)), "Estimated ACU from DBLoad & CPU (max blend)", dbload

    if acu_from_dbload:
        return float(sum(a * period for a in acu_from_dbload)), f"Estimated ACU from DBLoad×{factor:.2f}", dbload

    if acu_from_cpu_writer:
        return float(sum(a * period for a in acu_from_cpu_writer)), "CPU-only (writer) ACU estimate", cpu_writer

    # CPU max across instances fallback
    per_inst_cpu = [series for k, series in metrics.items() if k.startswith("ci_") and series.values]
    if per_inst_cpu:
        min_len = min(len(s.values) for s in per_inst_cpu)
        effective_vcpu = writer_vcpu if writer_vcpu > 0 else 4
        period_any = infer_period_from_series(per_inst_cpu[0].timestamps)
        est: List[float] = []
        for i in range(min_len):
            cpu_pct = max((s.values[i] or 0.0) for s in per_inst_cpu)
            u = cpu_pct / 100.0
            raw = (u * effective_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
            est.append(snap_acu(raw, min_acu_default, max_acu_default, ACU_STEP))
        return float(sum(a * period_any for a in est)), "CPU-only (max across instances)", per_inst_cpu[0]

    return 0.0, "No DBLoad/ACU/CPU data; ACU=0", None

# ---------- I/O requests estimation ----------
def estimate_total_io_requests(metrics: Dict[str, MetricSeries]) -> float:
    total = 0.0
    for key, series in metrics.items():
        if key.startswith("ri_") or key.startswith("wi_"):
            period = infer_period_from_series(series.timestamps)
            total += sum((v or 0.0) * period for v in series.values)
    return total

# ---------- Cost calculators (normalized) ----------
def get_io_multiplier(storage_type: str, instance_class: str) -> float:
    if storage_type == "aurora-iopt1":
        return IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get(instance_class, IO_OPTIMIZED_MULTIPLIERS_BY_CLASS.get("default", 1.17))
    return 1.0

def serverless_monthly_equivalent(acu_seconds: float, storage_type: str,
                                  io_requests: float, window_hours: float) -> float:
    acu_hours = acu_seconds / 3600.0
    acu_price = SERVERLESS_ACU_PER_HOUR * (SERVERLESS_IOPT_MULTIPLIER if storage_type == "aurora-iopt1" else 1.0)
    compute_total = acu_hours * acu_price
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    window_total = compute_total + io_total
    return window_total * (MONTH_HOURS / max(1e-6, window_hours))

def provisioned_monthly_equivalent(compute_price_per_hour: float, hours: float, instance_count: int,
                                   storage_type: str, io_requests: float) -> float:
    compute_total = compute_price_per_hour * hours * instance_count
    io_total = 0.0
    if storage_type != "aurora-iopt1" and CHARGE_IO_FOR_STANDARD:
        io_total = (io_requests / 1_000_000.0) * IO_PRICE_PER_MILLION
    window_total = compute_total + io_total
    return window_total * (MONTH_HOURS / max(1e-6, hours))

# =========================
# Main evaluation per window/storage
# =========================
def evaluate_one_window_with_storage(*, label: str, factor: float,
                                     db_identifier: str, region: str,
                                     lookback_days: int, period_seconds: int,
                                     storage_type_used: str,
                                     cluster: dict,
                                     start_time: dt.datetime,
                                     end_time: dt.datetime) -> WindowDecision:
    rds, cw = get_clients(region)

    # refresh instance details to identify writer/current classes
    insts: List[dict] = []
    writer_id = None
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            iid = m["DBInstanceIdentifier"]
            di = rds.describe_db_instances(DBInstanceIdentifier=iid)["DBInstances"][0]
            insts.append(di)
            if m.get("IsClusterWriter"):
                writer_id = iid

    engine, mode = cluster.get("Engine", "aurora"), detect_mode(cluster)
    instance_ids = [m["DBInstanceIdentifier"] for m in cluster.get("DBClusterMembers", []) or []]
    current_classes = [di.get("DBInstanceClass") for di in insts if di.get("DBInstanceClass")]
    writer_class = None
    if writer_id:
        for di in insts:
            if di.get("DBInstanceIdentifier") == writer_id:
                writer_class = di.get("DBInstanceClass")
                break

    start, end = start_time, end_time
    hours = hours_between(start, end)

    # PI present?
    pi_enabled = any(di.get("PerformanceInsightsEnabled") for di in insts)

    # fetch metrics
    metrics = fetch_metrics(
        cw, cluster_id=db_identifier, instance_ids=instance_ids, writer_instance_id=writer_id,
        start=start, end=end, period=period_seconds, enable_q1_q2=pi_enabled
    )

    shape = summarize_shape(metrics)

    # ---------- CPU-only coverage check ----------
    cpu_writer = metrics.get("q4")
    expected_pts = expected_datapoints_for_window(start, end, period_seconds)
    actual_pts = len(cpu_writer.values) if cpu_writer else 0
    coverage = (actual_pts / expected_pts) if expected_pts > 0 else 0.0

    cluster_create = cluster.get("ClusterCreateTime")
    db_age_days = None
    db_create_iso = None
    if isinstance(cluster_create, dt.datetime):
        db_create_iso = cluster_create.isoformat()
        db_age_days = max(0.0, (end - cluster_create).total_seconds() / 86400.0)
    earliest_cpu_ts = earliest_timestamp(cpu_writer)

    if SKIP_ON_LOW_COVERAGE and coverage < COVERAGE_REQUIRED:
        # Skip and return diagnostics only
        return WindowDecision(
            label=label, factor=factor,
            lookback_days=lookback_days, period_seconds=period_seconds,
            db_identifier=db_identifier, engine=engine, mode_detected=mode,
            storage_type_used=storage_type_used, io_optimized=(storage_type_used == "aurora-iopt1"),
            instances_found=instance_ids, current_instance_classes=current_classes,
            shape=shape,
            serverless_cost=None, provisioned_best=None, provisioned_current=None,
            headroom_effective=compute_effective_headroom(shape),
            start_time=start.isoformat(), end_time=end.isoformat(),
            expected_points=expected_pts, actual_points=actual_pts,
            coverage_ratio=coverage, coverage_required=COVERAGE_REQUIRED,
            earliest_sample_time=earliest_cpu_ts.isoformat() if earliest_cpu_ts else None,
            db_create_time=db_create_iso, db_age_days=db_age_days,
            skipped=True,
            skip_reason=f"CPU coverage {coverage:.0%} < required {int(COVERAGE_REQUIRED*100)}%."
        )

    # ---------- Proceed with cost/fit ----------
    inst_count = max(1, len(instance_ids))
    writer_vcpu = INSTANCE_VCPU.get(writer_class or "", 0)

    acu_seconds, _, primary_series = estimate_acu_seconds(
        mode, cluster, metrics, factor,
        DEFAULT_MIN_ACU, DEFAULT_MAX_ACU, writer_vcpu
    )
    io_requests = estimate_total_io_requests(metrics)

    # serverless monthly-equivalent
    sv2_monthly = serverless_monthly_equivalent(acu_seconds, storage_type_used, io_requests, hours)
    serverless_cost = CostBreakdown(monthly_equivalent=sv2_monthly)

    # provisioned selection
    effective_vcpu = compute_effective_vcpu(current_classes, writer_class)

    # cost-sorted candidates
    sorted_cands: List[Tuple[str, float]] = []
    for ic, base_price in INSTANCE_HOURLY_PRICES.items():
        if ic.endswith(".medium"):  # defensive
            continue
        if ic not in CANDIDATE_CLASSES:
            continue
        mult = get_io_multiplier(storage_type_used, ic)
        sorted_cands.append((ic, base_price * mult))
    sorted_cands.sort(key=lambda x: x[1])

    basis = "AAS" if shape.dbload_p95 > 0 else ("CPU" if shape.cpu_p95 > 0 else "NONE")
    eff_headroom = compute_effective_headroom(shape)

    best_prov: Optional[ProvisionedOption] = None
    for ic, eff_price_per_hour in sorted_cands:
        vcpu = INSTANCE_VCPU.get(ic, 0)
        if basis == "AAS":
            fits = (shape.dbload_p95 <= eff_headroom * vcpu)
        elif basis == "CPU":
            req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
            fits = (req_acu <= eff_headroom * vcpu)
        else:
            fits = False

        monthly_eq = provisioned_monthly_equivalent(eff_price_per_hour, hours, inst_count, storage_type_used, io_requests)
        if fits:
            best_prov = ProvisionedOption(ic, True, monthly_eq)
            break  # first (cheapest-by-cost) fit

    # current provisioned (if any)
    current_prov = None
    if current_classes:
        curr = current_classes[0]
        if curr in INSTANCE_HOURLY_PRICES:
            mult = get_io_multiplier(storage_type_used, curr)
            curr_price = INSTANCE_HOURLY_PRICES[curr] * mult
            monthly_eq = provisioned_monthly_equivalent(curr_price, hours, inst_count, storage_type_used, io_requests)
            fits = False
            if basis == "AAS" and shape.dbload_p95 > 0:
                fits = (shape.dbload_p95 <= eff_headroom * INSTANCE_VCPU.get(curr, 0))
            elif basis == "CPU" and shape.cpu_p95 > 0:
                req_acu = cpu_p95_to_required_acu(shape.cpu_p95, effective_vcpu)
                fits = (req_acu <= eff_headroom * INSTANCE_VCPU.get(curr, 0))
            current_prov = ProvisionedOption(curr, fits, monthly_eq)

    return WindowDecision(
        label=label, factor=factor,
        lookback_days=lookback_days, period_seconds=period_seconds,
        db_identifier=db_identifier, engine=engine, mode_detected=mode,
        storage_type_used=storage_type_used, io_optimized=(storage_type_used == "aurora-iopt1"),
        instances_found=instance_ids, current_instance_classes=current_classes,
        shape=shape, serverless_cost=serverless_cost,
        provisioned_best=best_prov, provisioned_current=current_prov,
        headroom_effective=eff_headroom,
        start_time=start.isoformat(), end_time=end.isoformat(),
        expected_points=expected_pts, actual_points=actual_pts,
        coverage_ratio=coverage, coverage_required=COVERAGE_REQUIRED,
        earliest_sample_time=earliest_cpu_ts.isoformat() if earliest_cpu_ts else None,
        db_create_time=db_create_iso, db_age_days=db_age_days,
        skipped=False, skip_reason=None
    )

# =========================
# Post-processing helpers
# =========================
def _collect_candidates(flat_results: Dict) -> List[Dict]:
    out: List[Dict] = []
    for storage_key, win_map in flat_results.items():
        for window_label, factor_map in win_map.items():
            for factor_str, wd in factor_map.items():
                # skip windows marked as skipped
                if wd.get("skipped"):
                    continue
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

# =========================
# Lambda Handler
# =========================
def lambda_handler(event, context):
    try:
        # Strict input: must be exactly {"cluster_id": "..."}
        if not isinstance(event, dict) or set(event.keys()) != {"cluster_id"} or not event["cluster_id"]:
            return {"statusCode": 400, "body": json.dumps({"error": "Invalid input. Payload must be exactly {'cluster_id': '<DBClusterIdentifier>'}."})}

        db_cluster_id = event["cluster_id"]

        rds, _ = get_clients(REGION)
        cluster, insts = describe_cluster(rds, db_cluster_id)
        detected_storage = detect_storage_type(cluster, insts)  # may be ""

        # Choose storage comparison modes
        if STORAGE_COMPARISON_MODE == "auto":
            modes_to_run = [detected_storage or "aurora"]
        elif STORAGE_COMPARISON_MODE == "standard":
            modes_to_run = ["aurora"]
        elif STORAGE_COMPARISON_MODE == "iopt1":
            modes_to_run = ["aurora-iopt1"]
        elif STORAGE_COMPARISON_MODE == "both":
            modes_to_run = ["aurora", "aurora-iopt1"]
        else:
            modes_to_run = [detected_storage or "aurora"]

        results: Dict[str, Dict[str, Dict[str, dict]]] = {}

        for storage_mode in modes_to_run:
            storage_key = "standard" if storage_mode == "aurora" else "iopt1"
            results[storage_key] = {}

            for label, days, period in WINDOWS:
                results[storage_key][label] = {}

                # Freeze window timestamps once per (storage_mode, window) with -5 min skew
                end_ts = now_utc() - dt.timedelta(minutes=5)
                start_ts = end_ts - dt.timedelta(days=days)

                for factor in AAS_TO_ACU_FACTORS:
                    wd = evaluate_one_window_with_storage(
                        label=label, factor=factor,
                        db_identifier=db_cluster_id, region=REGION,
                        lookback_days=days, period_seconds=period,
                        storage_type_used=storage_mode,
                        cluster=cluster, start_time=start_ts, end_time=end_ts
                    )
                    results[storage_key][label][f"{factor:.2f}"] = asdict(wd)

        # Post-processing: cheapest picks (only among non-skipped windows)
        candidates = _collect_candidates(results)
        cheapest_overall = _pick_cheapest(candidates)
        cheapest_by_storage = {sk: _pick_cheapest([c for c in candidates if c["storage"] == sk]) for sk in ["standard", "iopt1"]}
        top3_overall = _topk(candidates, 3)

        body = {
            "cluster_id": db_cluster_id,
            "region": REGION,
            "windows": WINDOWS,
            "factors": AAS_TO_ACU_FACTORS,
            "storage_comparison_mode": STORAGE_COMPARISON_MODE,
            "detected_storage_type": detected_storage or "unknown",
            "serverless_acu_per_hour": SERVERLESS_ACU_PER_HOUR,
            "serverless_iopt_multiplier": SERVERLESS_IOPT_MULTIPLIER,
            "io_price_per_million": IO_PRICE_PER_MILLION,
            "charge_io_for_standard": CHARGE_IO_FOR_STANDARD,
            "instance_hourly_prices": INSTANCE_HOURLY_PRICES,
            "io_optimized_multipliers_by_class": IO_OPTIMIZED_MULTIPLIERS_BY_CLASS,
            "results": results,
            "postprocessor": {
                "cheapest_overall": cheapest_overall,
                "cheapest_by_storage": cheapest_by_storage,
                "top3_overall": top3_overall,
            }
        }
        return {"statusCode": 200, "body": json.dumps(body, default=str)}
    except Exception as e:
        if DEBUG:
            import traceback; traceback.print_exc()
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
