#!/usr/bin/env bash
# Usage: bash deploy/install.sh (from action-space-defenders repo) /path/to/organizer/Action_Space
# Copies our defender files into the organizers' repo and registers defend_rl / defend_lead / frame_check.
set -e
ORG="${1:?path to organizer repo}"; HERE="$(cd "$(dirname "$0")" && pwd)"
cp "$HERE/drone_deployment_infra/defend_live.py" "$ORG/drone_deployment_infra/"
cp "$HERE"/drone_deployment_infra/swarm_ops/policies/*.py "$ORG/drone_deployment_infra/swarm_ops/policies/"
mkdir -p "$ORG/team4"; cp -r "$HERE/policies" "$ORG/team4/"
INIT="$ORG/drone_deployment_infra/swarm_ops/policies/__init__.py"
grep -q defend_rl "$INIT" || cat >> "$INIT" <<'PY'

# team4 defenders
from .defend_rl import DefendRLPolicy
from .defend_lead import AssignLeadSafePolicy
from .frame_check import FrameCheckPolicy
POLICIES.update({"defend_rl": DefendRLPolicy, "defend_lead": AssignLeadSafePolicy,
                 "frame_check": FrameCheckPolicy})
PY
echo "installed. weights: $ORG/team4/policies/{adi_sparse_s3,adi_sparse_s2,aryan_s7_c3}/policy.npz"

# Optional: --patch-runner applies the takeoff keepalive patch to the organizers' swarm_policies.py
if [ "${2:-}" = "--patch-runner" ]; then
  SP="$ORG/drone_deployment_infra/swarm_ops/swarm_policies.py"
  cp -n "$SP" "$SP.orig_team4" 2>/dev/null || true
  patch "$SP" < "$HERE/patches/swarm_policies_keepalive.patch" && echo "runner patched (backup: $SP.orig_team4)"
fi
