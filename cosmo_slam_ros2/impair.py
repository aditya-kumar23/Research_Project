"""Reproducible network impairment helpers for ROS 2 paths.

Configure via environment variables (set by orchestrator or caller):
- COSMO_IMPAIR: JSON string with impairment spec, or a path to a JSON file.
- COSMO_IMPAIR_OUT_DIR: directory to write ``robustness_metrics.json``.

Spec fields (all optional):
- seed: int (default 42)
- random_loss_p: float (Bernoulli drop per message)
- burst_period_s: float, every period a drop window opens for burst_duration_s
- burst_duration_s: float length of the drop window
- blackouts: list of {rid: "a", start_s: 120, end_s: 180, mode: "sender|either"}
- bw_caps_mbps: { "default": 10.0, "a": 10.0, "b": 5.0 }

The time anchor is monotonic perf counter at ImpairmentPolicy construction.
"""
from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random


def _now_mono() -> float:
    try:
        return float(time.perf_counter())
    except Exception:
        return float(time.time())


def _read_spec_from_env() -> Optional[Dict[str, Any]]:
    spec_raw = os.environ.get("COSMO_IMPAIR")
    path = os.environ.get("COSMO_IMPAIR_FILE")
    if spec_raw and spec_raw.strip():
        # If looks like a path and exists, prefer file
        if os.path.exists(spec_raw):
            path = spec_raw
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    if spec_raw and spec_raw.strip():
        try:
            return json.loads(spec_raw)
        except Exception:
            return None
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


@dataclass
class _Bucket:
    rate_bps: float
    capacity_bytes: float
    tokens: float = 0.0
    last_mono: float = field(default_factory=_now_mono)

    def offer(self, bytes_len: int, now: Optional[float] = None) -> float:
        """Return delay seconds required before ``bytes_len`` can be sent.

        Implements a simple token-bucket: tokens accumulate at rate_bps, capped
        at capacity. Sending consumes ``bytes_len`` tokens; if insufficient,
        compute the time needed to accumulate the deficit.
        """
        if now is None:
            now = _now_mono()
        dt = max(0.0, now - self.last_mono)
        self.tokens = min(self.capacity_bytes, self.tokens + dt * self.rate_bps)
        self.last_mono = now
        deficit = float(bytes_len) - self.tokens
        if deficit <= 0.0:
            self.tokens -= float(bytes_len)
            return 0.0
        # Time to accumulate deficit at rate_bps
        wait = deficit / max(1.0, self.rate_bps)
        # Advance time and consume
        self.tokens = 0.0
        self.last_mono = now + wait
        return wait


class ImpairmentPolicy:
    def __init__(self, spec: Optional[Dict[str, Any]], *, out_dir: Optional[str] = None) -> None:
        self.spec = dict(spec or {})
        self.anchor_mono = _now_mono()
        self.seed = int(self.spec.get("seed", 42))
        self.rng = random.Random(self.seed)
        # Bernoulli loss
        self.p_loss = float(self.spec.get("random_loss_p", 0.0) or 0.0)
        # Periodic bursts
        self.burst_period = _safe_float(self.spec.get("burst_period_s"), 0.0)
        self.burst_duration = _safe_float(self.spec.get("burst_duration_s"), 0.0)
        # Warm-up window (no drops) to preserve bootstrap priors
        self.warmup_s = _safe_float(self.spec.get("warmup_s"), 0.0)
        # Blackouts per robot id
        self.blackouts = []
        for b in self.spec.get("blackouts", []) or []:
            try:
                self.blackouts.append({
                    "rid": str(b.get("rid")),
                    "start_s": _safe_float(b.get("start_s"), 0.0),
                    "end_s": _safe_float(b.get("end_s"), 0.0),
                    "mode": str(b.get("mode", "sender")),
                })
            except Exception:
                continue
        # Bandwidth caps (per sender rid) in Mbps
        caps = self.spec.get("bw_caps_mbps", {}) or {}
        self.default_mbps = _safe_float(caps.get("default"), 0.0)
        self.caps_mbps: Dict[str, float] = {k: _safe_float(v) for k, v in caps.items() if k != "default"}
        # Token buckets per sender
        self._buckets: Dict[str, _Bucket] = {}
        # Stats
        self.out_dir = out_dir or os.environ.get("COSMO_IMPAIR_OUT_DIR")
        self.stats = {
            "seed": self.seed,
            "random_loss_p": self.p_loss,
            "burst_period_s": self.burst_period,
            "burst_duration_s": self.burst_duration,
            "bw_caps_mbps": {**({"default": self.default_mbps} if self.default_mbps > 0 else {}), **self.caps_mbps},
            "drops": {"random": 0, "burst": 0, "blackout": 0},
            "throttle_seconds": {},  # per sender
            "staleness_wall_s": {},  # per sender summary (count/mean/max)
            "topics": {},
        }

    @classmethod
    def from_env(cls) -> Optional["ImpairmentPolicy"]:
        spec = _read_spec_from_env()
        if spec is None:
            return None
        out_dir = os.environ.get("COSMO_IMPAIR_OUT_DIR")
        return cls(spec, out_dir=out_dir)

    def _is_in_burst(self, now_mono: Optional[float] = None) -> bool:
        if self.burst_period <= 0.0 or self.burst_duration <= 0.0:
            return False
        if now_mono is None:
            now_mono = _now_mono()
        t = now_mono - self.anchor_mono
        # Window active if within [0, burst_duration) modulo period
        return (t % self.burst_period) < self.burst_duration

    def _is_blackout(self, sender: str, receiver: str, now_mono: Optional[float] = None) -> bool:
        if not self.blackouts:
            return False
        if now_mono is None:
            now_mono = _now_mono()
        t = now_mono - self.anchor_mono
        for b in self.blackouts:
            mode = b.get("mode", "sender")
            rid = b.get("rid")
            if mode == "sender":
                match = (sender == rid)
            elif mode == "either":
                match = (sender == rid or receiver == rid)
            else:
                match = (sender == rid)
            if match and (t >= b.get("start_s", 0.0)) and (t <= b.get("end_s", 0.0)):
                return True
        return False

    def _bucket_for(self, sender: str) -> Optional[_Bucket]:
        mbps = self.caps_mbps.get(sender, self.default_mbps)
        if mbps <= 0.0:
            return None
        b = self._buckets.get(sender)
        if b is None:
            rate_bps = float(mbps) * 1e6 / 8.0  # bytes/s
            # Capacity: allow small bursts (100 ms worth)
            capacity = max(1024.0, 0.1 * rate_bps)
            b = _Bucket(rate_bps=rate_bps, capacity_bytes=capacity)
            self._buckets[sender] = b
        return b

    def _topic_key(self, sender: str, receiver: str, topic: Optional[str]) -> str:
        if topic:
            return str(topic)
        return f"{sender}->{receiver}"

    def on_send(self, *, sender: str, receiver: str, bytes_len: int, sent_wall_time: Optional[float] = None, topic: Optional[str] = None) -> tuple[float, Optional[str]]:
        """Return (delay_s, drop_reason) for this send.

        Caller should sleep for ``delay_s`` then, if ``drop_reason`` is not None,
        drop the message and not send it.
        """
        now = _now_mono()
        topic_key = self._topic_key(sender, receiver, topic)
        topic_stats = self.stats["topics"].setdefault(topic_key, {"attempts": 0, "drops": 0, "delivered": 0})
        topic_stats["attempts"] = int(topic_stats.get("attempts", 0)) + 1
        # Optionally suppress drops during warm-up
        in_warmup = (self.warmup_s > 0.0 and (now - self.anchor_mono) < self.warmup_s)
        # Blackout dominates
        if self._is_blackout(sender, receiver, now):
            self.stats["drops"]["blackout"] += 1
            topic_stats["drops"] = int(topic_stats.get("drops", 0)) + 1
            return 0.0, "blackout"
        # Periodic burst drop window (unless warm-up)
        if not in_warmup and self._is_in_burst(now):
            self.stats["drops"]["burst"] += 1
            topic_stats["drops"] = int(topic_stats.get("drops", 0)) + 1
            return 0.0, "burst"
        # Bernoulli loss (unless warm-up)
        if not in_warmup and self.p_loss > 0.0 and self.rng.random() < self.p_loss:
            self.stats["drops"]["random"] += 1
            topic_stats["drops"] = int(topic_stats.get("drops", 0)) + 1
            return 0.0, "random"
        # Bandwidth throttle (token bucket)
        delay = 0.0
        bucket = self._bucket_for(sender)
        if bucket is not None:
            d = bucket.offer(int(bytes_len), now)
            if d > 0.0:
                delay = d
                # Accumulate per sender
                s = self.stats["throttle_seconds"].get(sender, 0.0)
                self.stats["throttle_seconds"][sender] = float(s + d)
        # Staleness on send (post-throttle) — how old is the payload at send time
        if sent_wall_time is not None:
            try:
                st = float(time.time() - float(sent_wall_time)) + delay
                entry = self.stats["staleness_wall_s"].get(sender)
                if entry is None:
                    self.stats["staleness_wall_s"][sender] = {"count": 1, "sum": st, "max": st}
                else:
                    entry["count"] = int(entry.get("count", 0)) + 1
                    entry["sum"] = float(entry.get("sum", 0.0)) + st
                    entry["max"] = float(max(float(entry.get("max", 0.0)), st))
            except Exception:
                pass
        topic_stats["delivered"] = int(topic_stats.get("delivered", 0)) + 1
        return delay, None

    def export(self) -> None:
        out_dir = self.out_dir
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            # Finalize staleness means
            if "staleness_wall_s" in self.stats:
                for k, v in list(self.stats["staleness_wall_s"].items()):
                    try:
                        cnt = int(v.get("count", 0))
                        if cnt > 0:
                            v["mean"] = float(v.get("sum", 0.0)) / cnt
                    except Exception:
                        pass
            if "topics" in self.stats:
                for key, data in list(self.stats["topics"].items()):
                    try:
                        attempts = int(data.get("attempts", 0))
                        drops = int(data.get("drops", 0))
                        delivered = int(data.get("delivered", attempts - drops))
                        eta = float(delivered) / attempts if attempts > 0 else None
                        data["delivery_rate"] = eta
                    except Exception:
                        continue
            path = os.path.join(out_dir, "robustness_metrics.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"spec": self.spec, "stats": self.stats}, f, indent=2)
        except Exception:
            pass
