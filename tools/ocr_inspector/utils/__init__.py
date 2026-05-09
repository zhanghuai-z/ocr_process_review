"""Coordinate transform utilities.

Coordinate spaces:
  image_space   — original image pixel coords (origin = top-left)
  scene_space   — QGraphicsScene logical coords (image at origin, 1:1 px)
  viewport_space — QWidget pixels after pan+zoom

image_space ≡ scene_space in our canvas (image placed at origin, no scene scale).
"""
from __future__ import annotations
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsView


def scene_to_image(scene_pt: QPointF) -> QPointF:
    return QPointF(scene_pt.x(), scene_pt.y())


def image_to_scene(image_pt: QPointF) -> QPointF:
    return QPointF(image_pt.x(), image_pt.y())


def viewport_to_scene(view: QGraphicsView, vp_pt: QPointF) -> QPointF:
    return view.mapToScene(vp_pt.toPoint())


def scene_to_viewport(view: QGraphicsView, scene_pt: QPointF) -> QPointF:
    return QPointF(view.mapFromScene(scene_pt))
