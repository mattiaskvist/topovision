"""Pydantic models for height extraction output."""

from typing import Literal

from pydantic import BaseModel, Field


class ContourLine(BaseModel):
    """Represents a single contour line with its geometry and height."""

    id: int = Field(
        ..., description="Unique identifier for the contour within the image."
    )
    points: list[tuple[int, int]] = Field(
        ..., description="List of (x, y) coordinates defining the contour polyline."
    )
    height: float | None = Field(
        None, description="The extracted or inferred height of the contour."
    )
    source: Literal["ocr", "inference", "unknown"] = Field(
        "unknown", description="The source of the height value."
    )


class HeightExtractionOutput(BaseModel):
    """Represents the complete output of the height extraction pipeline for an image."""

    image_path: str = Field(..., description="Path to the source image.")
    contours: list[ContourLine] = Field(..., description="List of extracted contours.")
