For a serverless cluster you already have real ACU usage:

1 ACU ≈ ~1 vCPU worth of capacity (good enough for cost sizing).

Take a robust percentile (e.g., p95 of ACU) as the steady-state need.

Apply your headroom (e.g., / AAS_HEADROOM) and map to the nearest instance class with ≥ that many vCPUs.

Price those provisioned candidates and compare to the actual Serverless cost we computed from ACU-hours.

This does not require CPUUtilization; ACU is the stronger, “already blended” signal for serverless.
