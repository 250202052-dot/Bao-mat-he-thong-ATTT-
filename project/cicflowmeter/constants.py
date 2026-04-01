"""
Runtime constants for the Python cicflowmeter fork used in the project.

Important:
- ``CLUMP_TIMEOUT`` / ``ACTIVE_TIMEOUT`` / ``BULK_BOUND`` directly affect
  feature extraction semantics. Keep them aligned with the CICFlowMeter-style
  features used during training unless you intentionally want a different
  flow-definition regime.
- ``EXPIRED_UPDATE`` / ``PACKETS_PER_GC`` mainly affect how quickly completed
  flows are garbage-collected and emitted. These are safer knobs for improving
  realtime behaviour without substantially changing the feature space.
"""

# Emit idle flows sooner so realtime inference receives completed flows with a
# smaller delay. The previous 15s default made HTTP/CSV output feel "batched".
EXPIRED_UPDATE = 5

# Keep the original feature-clumping behaviour to stay close to training.
CLUMP_TIMEOUT = 1
ACTIVE_TIMEOUT = 5
BULK_BOUND = 4

# Trigger periodic GC more often under steady traffic so completed flows are
# flushed earlier instead of waiting for very large packet batches.
PACKETS_PER_GC = 200

# HTTP posting timeout for realtime writer mode. This is a deployment concern,
# not a feature-extraction parameter.
HTTP_POST_TIMEOUT_SECONDS = 30
