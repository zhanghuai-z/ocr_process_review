"""Test factory for the explicit OCR routing observation boundary."""
from __future__ import annotations

from app.core.layout_scope import layout_snapshot_fingerprint
from app.models.ocr_routing_observation import RoutingObservationBundle


def routing_observation_bundle(snapshot, prepass, *, block_vl_observations=()):
    return RoutingObservationBundle(
        run_uid="routingrun-test",
        snapshot=snapshot,
        prepass=prepass,
        image_hash="image-hash-test",
        layout_fingerprint=layout_snapshot_fingerprint(snapshot),
        block_vl_observations=tuple(block_vl_observations),
    )
