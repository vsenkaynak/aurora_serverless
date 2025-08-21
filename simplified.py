# lambda_function.py
"""
Aurora Serverless v2 vs Provisioned cost comparison (simplified, same logic)

Behavior preserved:
- Strict input: only {"cluster_id": "..."}
- Windows: 14d/30d/60d (fixed start/end; -5m CloudWatch skew)
- Metrics: DBLoad/DBLoadCPU if PI enabled; ServerlessDatabaseCapacity; CPU/IOPS series
- Coverage gate: require ≥50% writer CPU coverage per window else skip with diagnostics
- Sizing: p95 basis; spike-aware headroom (LOWER target if p99 >> p95)
- ACU estimation precedence: actual ACU (sv2) → DBLoad×factor → CPU fallback (writer, else max across instances)
- I/O: aggregate (ReadIOPS+WriteIOPS)*period; charged only on Standard storage
- Costs: normalized to 730h “monthly equivalent”
- Provisioned selection: candidates cost-sorted; first that fits with headroom; evaluate current class too
- Output: JSON only (no prints when DEBUG=False)
"""

from __future__ import annotations
import json, math
import datetime as dt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import boto3

# ========================
# Toggle logging
# ========================
DEBUG = False
def log(*a, **k):
    if DEBUG: print(*a, **k)

# ========================
# Config (edit as needed)
# ========================
REGION = "us-east-1"

# Compare storage modes: "auto" | "standard" | "iopt1" | "both"
STORAGE_COMPARISON_MODE = "both"

# Windows: (label, days, period_seconds)
WINDOWS: List[Tuple[str, int, int]] = [
    ("14d_60s", 14, 60),
    ("30d_120s", 30, 120),
    ("60d_300s", 60, 300),
]

# DBLoad (AAS) → ACU conversion sensitivity factors
AAS_TO_ACU_FACTORS: List[float] = [0.75, 1.00, 1.25]

# Serverless prices
SERVERLESS_ACU_PER_HOUR = 0.12            # $ / ACU-hour (Standard)
SERVERLESS_IOPT_MULTIPLIER = 1.17         # ACU multiplier on I/O-Optimized

# I/O price (Standard only; I/O-Optimized has $0 per I/O)
IO_PRICE_PER_MILLION = 0.20
CHARGE_IO_FOR_STANDARD = True

# ACU limits & step
DEFAULT_MIN_ACU = 0.5
DEFAULT_MAX_ACU = 64.0
ACU_STEP = 0.5

# CPU→ACU fallback headroom (separate from spike headroom used for fit checks)
USE_CPU_IN_ACU_ESTIMATE = True
CPU_TO_ACU_HEADROOM = 0.70

# Spike-aware headroom (LOWER target when spiky → more buffer)
HEADROOM_BASE = 0.70
HEADROOM_MIN = 0.50
HEADROOM_MAX = 0.80
SPIKE_RATIO_THRESHOLD = 1.30     # treat spiky if p99 >= 1.30 * p95
HEADROOM_DROP_AT_THRESHOLD = 0.05
HEADROOM_DROP_MAX = 0.15         # allow drop up to 0.15 (e.g., 0.70 → 0.55)

MONTH_HOURS = 730.0

# -------- Candidate instance classes (no .medium, no t2/t3) --------
CANDIDATE_CLASSES = [
    # r5 / r6g / r6i / r7g are the most relevant; include a couple more families for completeness
    "db.r5.large","db.r5.xlarge","db.r5.2xlarge","db.r5.4xlarge","db.r5.12xlarge","db.r5.24xlarge",
    "db.r6g.large","db.r6g.xlarge","db.r6g.2xlarge","db.r6g.4xlarge","db.r6g.8xlarge","db.r6g.12xlarge","db.r6g.16xlarge",
    "db.r6i.large","db.r6i.xlarge","db.r6i.2xlarge","db.r6i.4xlarge","db.r6i.8xlarge","db.r6i.12xlarge","db.r6i.16xlarge","db.r6i.24xlarge","db.r6i.32xlarge",
    "db.r7g.large","db.r7g.xlarge","db.r7g.2xlarge","db.r7g.4xlarge","db.r7g.8xlarge",
    "db.x2g.large","db.x2g.xlarge","db.x2g.2xlarge","db.x2g.4xlarge","db.x2g.8xlarge","db.x2g.12xlarge","db.x2g.16xlarge",
    "db.z1d.large","db.z1d.xlarge","db.z1d.2xlarge","db.z1d.3xlarge","db.z1d.6xlarge","db.z1d.12xlarge",
]

INSTANCE_VCPU: Dict[str,int] = {
    "db.r5.large":2,"db.r5.xlarge":4,"db.r5.2xlarge":8,"db.r5.4xlarge":16,"db.r5.12xlarge":48,"db.r5.24xlarge":96,
    "db.r6g.large":2,"db.r6g.xlarge":4,"db.r6g.2xlarge":8,"db.r6g.4xlarge":16,"db.r6g.8xlarge":32,"db.r6g.12xlarge":48,"db.r6g.16xlarge":64,
    "db.r6i.large":2,"db.r6i.xlarge":4,"db.r6i.2xlarge":8,"db.r6i.4xlarge":16,"db.r6i.8xlarge":32,"db.r6i.12xlarge":48,"db.r6i.16xlarge":64,"db.r6i.24xlarge":96,"db.r6i.32xlarge":128,
    "db.r7g.large":2,"db.r7g.xlarge":4,"db.r7g.2xlarge":8,"db.r7g.4xlarge":16,"db.r7g.8xlarge":32,
    "db.x2g.large":2,"db.x2g.xlarge":4,"db.x2g.2xlarge":8,"db.x2g.4xlarge":16,"db.x2g.8xlarge":32,"db.x2g.12xlarge":48,"db.x2g.16xlarge":64,
    "db.z1d.large":2,"db.z1d.xlarge":4,"db.z1d.2xlarge":8,"db.z1d.3xlarge":12,"db.z1d.6xlarge":24,"db.z1d.12xlarge":48,
}

# Starter prices (USD/hr) — adjust to your static price sheet
INSTANCE_HOURLY_PRICES: Dict[str,float] = {
    "db.r5.large":0.250,"db.r5.xlarge":0.500,"db.r5.2xlarge":1.000,"db.r5.4xlarge":2.000,"db.r5.12xlarge":6.000,"db.r5.24xlarge":12.000,
    "db.r6g.large":0.246,"db.r6g.xlarge":0.492,"db.r6g.2xlarge":0.984,"db.r6g.4xlarge":1.968,"db.r6g.8xlarge":3.936,"db.r6g.12xlarge":5.904,"db.r6g.16xlarge":7.872,
    "db.r6i.large":0.270,"db.r6i.xlarge":0.540,"db.r6i.2xlarge":1.080,"db.r6i.4xlarge":2.160,"db.r6i.8xlarge":4.320,"db.r6i.12xlarge":6.480,"db.r6i.16xlarge":8.640,"db.r6i.24xlarge":12.960,"db.r6i.32xlarge":17.280,
    "db.r7g.large":0.252,"db.r7g.xlarge":0.504,"db.r7g.2xlarge":1.008,"db.r7g.4xlarge":2.016,"db.r7g.8xlarge":4.032,
    "db.x2g.large":0.370,"db.x2g.xlarge":0.740,"db.x2g.2xlarge":1.480,"db.x2g.4xlarge":2.960,"db.x2g.8xlarge":5.920,"db.x2g.12xlarge":8.880,"db.x2g.16xlarge":11.840,
    "db.z1d.large":0.310,"db.z1d.xlarge":0.620,"db.z1d.2xlarge":1.240,"db.z1d.3xlarge":1.860,"db.z1d.6xlarge":3.720,"db.z1d.12xlarge":7.440,
}

# I/O-Optimized multipliers for provisioned compute (per class)
IO_OPT_MULT_BY_CLASS: Dict[str,float] = {**{c:1.17 for c in CANDIDATE_CLASSES}, "db.x2g.large":1.12,"db.x2g.xlarge":1.12,"db.x2g.2xlarge":1.12,"db.x2g.4xlarge":1.12,"db.x2g.8xlarge":1.12,"db.x2g.12xlarge":1.12,"db.x2g.16xlarge":1.12}

# =========================
# Data models
# =========================
@dataclass
class MetricSeries:
    ts: List[dt.datetime]
    vals: List[float]

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
    # coverage / diagnostics
    expected_points: int = 0
    actual_points: int = 0
    coverage_ratio: float = 0.0
    skipped: bool = False
    skip_reason: str = ""
    earliest_sample_time: str = ""
    db_create_time: str = ""
    db_age_days: float = 0.0

# =========================
# Utilities
# =========================
def percentile(vals: List[float], p: float) -> float:
    if not vals: return 0.0
    s = sorted(vals)
    if p <= 0: return s[0]
    if p >= 100: return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    return s[int(k)] if f == c else s[f]*(c-k) + s[c]*(k-f)

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def hours_between(a: dt.datetime, b: dt.datetime) -> float:
    return (b - a).total_seconds() / 3600.0

def snap_acu(x: float, mn: float, mx: float, step: float = ACU_STEP) -> float:
    if x <= 0: return mn
    return max(mn, min(mx, round(x/step) * step))

def get_clients(region=REGION):
    return boto3.client("rds", region_name=region), boto3.client("cloudwatch", region_name=region)

# =========================
# AWS describe / detect
# =========================
def describe_cluster(rds, cluster_id: str) -> Tuple[dict,List[dict]]:
    c = rds.describe_db_clusters(DBClusterIdentifier=cluster_id)["DBClusters"][0]
    insts = []
    for m in c.get("DBClusterMembers", []) or []:
        insts.append(rds.describe_db_instances(DBInstanceIdentifier=m["DBInstanceIdentifier"])["DBInstances"][0])
    return c, insts

def detect_mode(cluster: dict) -> str:
    if cluster.get("EngineMode") == "serverless": return "serverless-v1"
    if cluster.get("ServerlessV2ScalingConfiguration"): return "serverless-v2"
    return "provisioned"

def detect_storage_type(cluster: dict, insts: List[dict]) -> str:
    st = (cluster.get("StorageType") or "").lower()
    if st: return st
    for di in insts:
        st_i = (di.get("StorageType") or "").lower()
        if st_i: return st_i
    return ""  # unknown

def get_cluster_acu_limits(cluster: dict) -> Tuple[float,float]:
    cfg = cluster.get("ServerlessV2ScalingConfiguration") or {}
    return float(cfg.get("MinCapacity") or DEFAULT_MIN_ACU), float(cfg.get("MaxCapacity") or DEFAULT_MAX_ACU)

def writer_instance_id(insts: List[dict]) -> Optional[str]:
    for m in insts:
        if m.get("DBInstanceStatus") and any(k in str(m).lower() for k in ["writer","primary"]):
            # Not reliable; prefer cluster member Writer flag if available
            pass
    # Use the ClusterMembers writer flag if available
    return None  # we’ll derive below from cluster

def find_writer_and_classes(cluster: dict, insts: List[dict]) -> Tuple[Optional[str], List[str], List[str]]:
    writer_id = None
    if cluster.get("DBClusterMembers"):
        for m in cluster["DBClusterMembers"]:
            if m.get("IsClusterWriter"):
                writer_id = m.get("DBInstanceIdentifier")
                break
    classes = []
    ids = []
    for di in insts:
        ids.append(di.get("DBInstanceIdentifier"))
        ic = di.get("DBInstanceClass")
        if ic: classes.append(ic)
    return writer_id, ids, classes

def performance_insights_enabled(insts: List[dict]) -> bool:
    return any(di.get("PerformanceInsightsEnabled") for di in insts)

# =========================
# Metrics fetch
# =========================
def infer_period(ts: List[dt.datetime]) -> int:
    return max(1, int((ts[1]-ts[0]).total_seconds())) if ts and len(ts) > 1 else 60

def mquery(qid, metric, dims, stat, period):
    return {
        "Id": qid,
        "MetricStat": {"Metric":{"Namespace":"AWS/RDS","MetricName":metric,"Dimensions":dims},
                       "Period": period, "Stat": stat},
        "ReturnData": True
    }

def fetch_metrics(cw, cluster_id: str, instance_ids: List[str], writer_id: Optional[str],
                  start: dt.datetime, end: dt.datetime, period: int, enable_q1_q2: bool) -> Dict[str, MetricSeries]:
    dims_cluster = [{"Name":"DBClusterIdentifier","Value":cluster_id}]
    queries = []
    if enable_q1_q2:
        queries += [
            mquery("q1","DBLoad",dims_cluster,"Average",period),
            mquery("q2","DBLoadCPU",dims_cluster,"Average",period),
        ]
    queries.append(mquery("q3","ServerlessDatabaseCapacity",dims_cluster,"Average",period))
    if writer_id:
        dims_writer = [{"Name":"DBInstanceIdentifier","Value":writer_id}]
        queries += [
            mquery("q4","CPUUtilization",dims_writer,"Average",period),
            mquery("q5","DatabaseConnections",dims_writer,"Average",period),
        ]
    qid = 6
    for iid in instance_ids:
        dims_inst = [{"Name":"DBInstanceIdentifier","Value":iid}]
        queries += [
            mquery(f"ri_{qid}","ReadIOPS",dims_inst,"Average",period),
            mquery(f"wi_{qid+1}","WriteIOPS",dims_inst,"Average",period),
            mquery(f"ci_{qid+2}","CPUUtilization",dims_inst,"Average",period),
        ]
        qid += 3

    out: Dict[str, MetricSeries] = {}
    next_token = None
    while True:
        args = {"StartTime":start,"EndTime":end,"MetricDataQueries":queries,"ScanBy":"TimestampAscending","MaxDatapoints":50000}
        if next_token: args["NextToken"] = next_token
        resp = cw.get_metric_data(**args)
        for r in resp["MetricDataResults"]:
            ts, vs = r.get("Timestamps", []) or [], r.get("Values", []) or []
            if r["Id"] in out:
                out[r["Id"]].ts.extend(ts); out[r["Id"]].vals.extend(vs)
            else:
                out[r["Id"]] = MetricSeries(ts, vs)
        next_token = resp.get("NextToken")
        if not next_token: break

    # sort each series by timestamp
    for k, s in out.items():
        if s.ts:
            pair = sorted(zip(s.ts, s.vals), key=lambda x: x[0])
            out[k] = MetricSeries([p[0] for p in pair], [p[1] for p in pair])
    return out

# =========================
# Coverage, shape, headroom
# =========================
def coverage_info(metrics: Dict[str, MetricSeries], start: dt.datetime, end: dt.datetime, period: int,
                  cluster: dict) -> Tuple[int,int,float,str,str,float]:
    cpu_writer = metrics.get("q4")
    actual = len(cpu_writer.vals) if cpu_writer else 0
    expected = int((end - start).total_seconds() / period) if period > 0 else 0
    ratio = (actual / expected) if expected > 0 else 0.0
    earliest = cpu_writer.ts[0].isoformat() if (cpu_writer and cpu_writer.ts) else ""
    db_create_time = cluster.get("ClusterCreateTime")
    db_ct = db_create_time.isoformat() if db_create_time else ""
    db_age_days = (now_utc() - db_create_time).total_seconds()/86400.0 if db_create_time else 0.0
    return expected, actual, ratio, earliest, db_ct, db_age_days

def summarize_shape(metrics: Dict[str, MetricSeries]) -> WorkloadShape:
    dbload = metrics.get("q1").vals if metrics.get("q1") else []
    cpu    = metrics.get("q4").vals if metrics.get("q4") else []
    conn   = metrics.get("q5").vals if metrics.get("q5") else []
    return WorkloadShape(
        dbload_p50=percentile(dbload,50), dbload_p95=percentile(dbload,95), dbload_p99=percentile(dbload,99),
        cpu_p50=percentile(cpu,50),       cpu_p95=percentile(cpu,95),       cpu_p99=percentile(cpu,99),
        connections_p95=percentile(conn,95)
    )

def compute_effective_headroom(shape: WorkloadShape) -> float:
    # ratio from whichever signal is available
    ratios = []
    if shape.dbload_p95 > 0: ratios.append((shape.dbload_p99 or 0)/max(1e-6, shape.dbload_p95))
    if shape.cpu_p95    > 0: ratios.append((shape.cpu_p99 or 0)/max(1e-6, shape.cpu_p95))
    ratio = max(ratios) if ratios else 1.0

    headroom = HEADROOM_BASE
    if ratio >= SPIKE_RATIO_THRESHOLD:
        # linear scale extra drop up to a 0.4-wide band beyond threshold
        extra = min(1.0, max(0.0, (ratio - SPIKE_RATIO_THRESHOLD) / 0.4))
        drop = min(HEADROOM_DROP_MAX, HEADROOM_DROP_AT_THRESHOLD + (HEADROOM_DROP_MAX - HEADROOM_DROP_AT_THRESHOLD)*extra)
        headroom = max(HEADROOM_MIN, HEADROOM_BASE - drop)
    return min(HEADROOM_MAX, max(HEADROOM_MIN, headroom))

# =========================
# ACU & I/O estimation
# =========================
def estimate_acu_seconds(cluster_mode: str, cluster: dict, metrics: Dict[str, MetricSeries],
                         factor: float, writer_vcpu: int) -> Tuple[float,str]:
    # 1) serverless v2 actual ACU, clamped to min/max
    if cluster_mode == "serverless-v2":
        sv2 = metrics.get("q3")
        if sv2 and sv2.vals:
            period = infer_period(sv2.ts)
            minc, maxc = get_cluster_acu_limits(cluster)
            acusec = sum(max(minc, min(maxc, v or 0.0)) * period for v in sv2.vals)
            return float(acusec), f"Used ServerlessDatabaseCapacity (clamped {minc}-{maxc} ACU)"

    # 2) try DBLoad and writer CPU
    dbload = metrics.get("q1")
    writer = metrics.get("q4")
    base_ts = dbload.ts if (dbload and dbload.ts) else (writer.ts if (writer and writer.ts) else [])
    period = infer_period(base_ts)

    acu_db: List[float] = []
    if dbload and dbload.vals:
        for v in dbload.vals:
            acu_db.append(snap_acu((v or 0.0)*factor, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU))

    acu_cpu: List[float] = []
    if USE_CPU_IN_ACU_ESTIMATE and writer and writer.vals and writer_vcpu > 0:
        for cpu_pct in writer.vals:
            u = (cpu_pct or 0.0)/100.0
            raw = (u * writer_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
            acu_cpu.append(snap_acu(raw, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU))

    # 2a) blend both by max (safety)
    if acu_db and acu_cpu:
        n = min(len(acu_db), len(acu_cpu))
        acusec = sum(max(acu_db[i], acu_cpu[i]) * period for i in range(n))
        return float(acusec), f"Estimated ACU from DBLoad×{factor:.2f} and CPU (max)"

    # 3) no DBLoad/CPU writer → max CPU across instances
    if not acu_db and not acu_cpu:
        per_inst = [s for k,s in metrics.items() if k.startswith("ci_") and s.vals]
        if per_inst:
            min_len = min(len(s.vals) for s in per_inst)
            eff_vcpu = writer_vcpu if writer_vcpu > 0 else 4
            acusec = 0.0
            if min_len > 0:
                for i in range(min_len):
                    cpu_pct = max((s.vals[i] or 0.0) for s in per_inst)
                    u = cpu_pct/100.0
                    raw = (u * eff_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
                    acusec += snap_acu(raw, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU) * period
                return float(acusec), f"CPU-only fallback (max across instances, vCPU={eff_vcpu})"

    if acu_db:
        return float(sum(a * period for a in acu_db)), f"Estimated ACU from DBLoad×{factor:.2f}"
    if acu_cpu:
        return float(sum(a * period for a in acu_cpu)), "CPU-only (writer)"

    return 0.0, "No DBLoad/ACU/CPU data; ACU=0"

def estimate_total_io_requests(metrics: Dict[str, MetricSeries]) -> float:
    total = 0.0
    for k, s in metrics.items():
        if k.startswith("ri_") or k.startswith("wi_"):
            total += sum((v or 0.0) * infer_period(s.ts) for v in s.vals)
    return total

# =========================
# Cost calculators
# =========================
def serverless_monthly_equiv(acu_seconds: float, storage_type: str, io_requests: float, window_hours: float) -> float:
    acu_hours = acu_seconds / 3600.0
    acu_price = SERVERLESS_ACU_PER_HOUR * (SERVERLESS_IOPT_MULTIPLIER if storage_type == "aurora-iopt1" else 1.0)
    compute_total = acu_hours * acu_price
    io_total = 0.0 if storage_type == "aurora-iopt1" or not CHARGE_IO_FOR_STANDARD else (io_requests/1_000_000.0)*IO_PRICE_PER_MILLION
    return (compute_total + io_total) * (MONTH_HOURS / max(1e-6, window_hours))

def class_effective_price_per_hour(ic: str, storage_type: str) -> float:
    base = INSTANCE_HOURLY_PRICES.get(ic, 0.0)
    mult = IO_OPT_MULT_BY_CLASS.get(ic, 1.17) if storage_type == "aurora-iopt1" else 1.0
    return base * mult

def provisioned_monthly_equiv(price_per_hour: float, hours: float, inst_count: int, storage_type: str, io_requests: float) -> float:
    compute_total = price_per_hour * hours * inst_count
    io_total = 0.0 if storage_type == "aurora-iopt1" or not CHARGE_IO_FOR_STANDARD else (io_requests/1_000_000.0)*IO_PRICE_PER_MILLION
    return (compute_total + io_total) * (MONTH_HOURS / max(1e-6, hours))

# =========================
# Fit & selection
# =========================
def effective_vcpu(current_classes: List[str], writer_class: Optional[str]) -> int:
    if writer_class and INSTANCE_VCPU.get(writer_class,0) > 0:
        return INSTANCE_VCPU[writer_class]
    mx = 0
    for ic in current_classes or []:
        mx = max(mx, INSTANCE_VCPU.get(ic,0))
    return mx if mx > 0 else 4

def cpu_p95_to_required_acu(cpu_p95: float, eff_vcpu: int) -> float:
    if eff_vcpu <= 0 or cpu_p95 <= 0: return 0.0
    utilization = cpu_p95 / 100.0
    raw = (utilization * eff_vcpu) / max(1e-6, CPU_TO_ACU_HEADROOM)
    return snap_acu(raw, DEFAULT_MIN_ACU, DEFAULT_MAX_ACU)

def first_fitting_candidate(shape: WorkloadShape, headroom: float, storage_type: str) -> Optional[Tuple[str,float]]:
    # Sort by effective hourly price (cheapest first)
    sorted_classes = sorted(CANDIDATE_CLASSES, key=lambda ic: class_effective_price_per_hour(ic, storage_type))
    # Determine required capacity
    if shape.dbload_p95 > 0:
        required = shape.dbload_p95     # treat AAS p95 as “ACU-like” demand
    else:
        # CPU fallback
        # convert CPU p95 → ACU at CPU_TO_ACU_HEADROOM using writer/current effective vCPU later in handler
        return None  # will be filled at call site where eff_vcpu is known
    for ic in sorted_classes:
        if required <= headroom * INSTANCE_VCPU.get(ic, 0):
            return ic, class_effective_price_per_hour(ic, storage_type)
    return None

# =========================
# Handler
# =========================
def lambda_handler(event, context=None):
    # Strict input
    if not isinstance(event, dict) or set(event.keys()) != {"cluster_id"} or not event["cluster_id"]:
        return {"error":"invalid input: expected {\"cluster_id\": \"<db-cluster-id>\"}"}

    rds, cw = get_clients()
    cid = event["cluster_id"]

    # Describe
    cluster, insts = describe_cluster(rds, cid)
    engine = cluster.get("Engine")
    mode = detect_mode(cluster)
    storage_detected = detect_storage_type(cluster, insts) or "aurora"
    writer_id, instance_ids, current_classes = find_writer_and_classes(cluster, insts)
    writer_class = None
    for di in insts:
        if di.get("DBInstanceIdentifier") == writer_id:
            writer_class = di.get("DBInstanceClass")
            break
    eff_vcpu = effective_vcpu(current_classes, writer_class)
    pi_enabled = performance_insights_enabled(insts)

    # Storage modes to evaluate
    storage_modes = []
    if STORAGE_COMPARISON_MODE == "auto": storage_modes = [storage_detected]
    elif STORAGE_COMPARISON_MODE == "both": storage_modes = ["aurora", "aurora-iopt1"]
    elif STORAGE_COMPARISON_MODE == "standard": storage_modes = ["aurora"]
    elif STORAGE_COMPARISON_MODE == "iopt1": storage_modes = ["aurora-iopt1"]

    decisions: List[WindowDecision] = []
    for storage in storage_modes:
        for (label, days, period) in WINDOWS:
            # Freeze time window, skew -5m to avoid CW lag
            end = now_utc() - dt.timedelta(minutes=5)
            start = end - dt.timedelta(days=days)
            window_hours = hours_between(start, end)

            for factor in AAS_TO_ACU_FACTORS:
                metrics = fetch_metrics(
                    cw, cid, instance_ids, writer_id, start, end, period,
                    enable_q1_q2=pi_enabled
                )
                exp_pts, act_pts, cov_ratio, earliest, db_ct, db_age_days = coverage_info(metrics, start, end, period, cluster)

                shape = summarize_shape(metrics)
                headroom = compute_effective_headroom(shape)

                # Coverage gate (writer CPU coverage)
                skipped = cov_ratio < 0.50
                skip_reason = "Insufficient writer CPU coverage (<50%)" if skipped else ""

                serverless_cost = None
                prov_best = None
                prov_curr = None

                if not skipped:
                    # ACU seconds & IO requests
                    acu_seconds, acu_note = estimate_acu_seconds(mode, cluster, metrics, factor, eff_vcpu)
                    io_requests = estimate_total_io_requests(metrics)

                    # serverless monthly eq
                    serverless_meq = serverless_monthly_equiv(acu_seconds, storage, io_requests, window_hours)
                    serverless_cost = CostBreakdown(monthly_equivalent=serverless_meq)

                    # provisioned: choose first fitting candidate
                    # compute required from AAS p95 or CPU fallback
                    required_acu = None
                    using_cpu_fallback = False
                    if shape.dbload_p95 > 0:
                        required_acu = shape.dbload_p95
                    else:
                        required_acu = cpu_p95_to_required_acu(shape.cpu_p95, eff_vcpu)
                        using_cpu_fallback = True

                    # find first fit
                    best_ic, best_price = None, None
                    for ic in sorted(CANDIDATE_CLASSES, key=lambda c: class_effective_price_per_hour(c, storage)):
                        capacity = headroom * INSTANCE_VCPU.get(ic, 0)
                        if required_acu <= capacity:
                            best_ic, best_price = ic, class_effective_price_per_hour(ic, storage)
                            break
                    if best_ic:
                        prov_best = ProvisionedOption(
                            instance_class=best_ic,
                            fits=True,
                            monthly_equivalent=provisioned_monthly_equiv(best_price, window_hours, 1, storage, io_requests)
                        )

                    # current class eval (if any)
                    if writer_class:
                        curr_price = class_effective_price_per_hour(writer_class, storage)
                        curr_capacity = headroom * INSTANCE_VCPU.get(writer_class, 0)
                        curr_fits = (required_acu <= curr_capacity)
                        prov_curr = ProvisionedOption(
                            instance_class=writer_class,
                            fits=curr_fits,
                            monthly_equivalent=provisioned_monthly_equiv(curr_price, window_hours, 1, storage, io_requests)
                        )

                decisions.append(WindowDecision(
                    label=label, factor=factor, lookback_days=days, period_seconds=period,
                    db_identifier=cid, engine=engine, mode_detected=mode,
                    storage_type_used=storage, io_optimized=(storage=="aurora-iopt1"),
                    instances_found=instance_ids, current_instance_classes=current_classes,
                    shape=shape, serverless_cost=serverless_cost,
                    provisioned_best=prov_best, provisioned_current=prov_curr,
                    headroom_effective=headroom,
                    start_time=start.isoformat(), end_time=end.isoformat(),
                    expected_points=exp_pts, actual_points=act_pts, coverage_ratio=cov_ratio,
                    skipped=skipped, skip_reason=skip_reason,
                    earliest_sample_time=earliest, db_create_time=db_ct, db_age_days=db_age_days
                ))

    # Post-processing helpers
    def meq_of(x: WindowDecision) -> float:
        vals = []
        if x.serverless_cost: vals.append(x.serverless_cost.monthly_equivalent)
        if x.provisioned_best: vals.append(x.provisioned_best.monthly_equivalent)
        if x.provisioned_current: vals.append(x.provisioned_current.monthly_equivalent)
        return min(vals) if vals else float("inf")

    cheapest_overall = min([d for d in decisions if not d.skipped], key=meq_of, default=None)
    cheapest_by_storage = {}
    for st in {"aurora","aurora-iopt1"}:
        cands = [d for d in decisions if (not d.skipped and d.storage_type_used==st)]
        cheapest_by_storage[st] = min(cands, key=meq_of, default=None)

    top3_overall = sorted([d for d in decisions if not d.skipped], key=meq_of)[:3]

    # Render
    resp = {
        "decisions": [asdict(d) for d in decisions],
        "cheapest_overall": asdict(cheapest_overall) if cheapest_overall else None,
        "cheapest_by_storage": {k:(asdict(v) if v else None) for k,v in cheapest_by_storage.items()},
        "top3_overall": [asdict(d) for d in top3_overall],
        "notes": {
            "headroom_base": HEADROOM_BASE,
            "spike_threshold_ratio": SPIKE_RATIO_THRESHOLD,
            "headroom_drop_at_threshold": HEADROOM_DROP_AT_THRESHOLD,
            "headroom_drop_max": HEADROOM_DROP_MAX,
            "cpu_to_acu_headroom": CPU_TO_ACU_HEADROOM,
        }
    }
    return resp

# For local testing:
if __name__ == "__main__":
    print(json.dumps(lambda_handler({"cluster_id":"your-cluster-id"}), indent=2, default=str))
