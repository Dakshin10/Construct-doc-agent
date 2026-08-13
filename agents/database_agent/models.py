"""
models.py
---------
Pydantic v2 model for validating the master_schema JSON payload.

All fields are required. Missing fields, wrong types, and null values
are rejected before the database tool is ever called.
"""

from pydantic import BaseModel, Field, model_validator


class MasterRecord(BaseModel):
    """
    Represents one row in the master_schema AlloyDB table.
    """

    document_id: str = Field(..., min_length=1, description="Primary key — must be non-empty string")
    project_name: str = Field(..., min_length=1, description="Project name — must be non-empty string")

    p1: int = Field(..., description="Integer metric p1")
    p2: int = Field(..., description="Integer metric p2")
    p3: int = Field(..., description="Integer metric p3")

    c1: int = Field(..., description="Integer metric c1")
    c2: int = Field(..., description="Integer metric c2")
    c3: int = Field(..., description="Integer metric c3")

    f1: int = Field(..., description="Integer metric f1")
    f2: int = Field(..., description="Integer metric f2")
    f3: int = Field(..., description="Integer metric f3")

    model_config = {
        "extra": "forbid",
        "strict": True,
    }

    @model_validator(mode="before")
    @classmethod
    def reject_none_values(cls, values: dict) -> dict:
        """Reject payloads where any required field is explicitly set to null."""
        if isinstance(values, dict):
            for field_name, value in values.items():
                if value is None:
                    raise ValueError(
                        f"Field '{field_name}' must not be null. "
                        "All fields are required with non-null values."
                    )
        return values


class SpreadFootingRecord(BaseModel):
    """
    Represents one row in the spread_footing_pad_schedule AlloyDB table.

    Fields
    ------
    footing_type      : Primary key (string, e.g. "F1")
    length            : Dimension length (string, e.g. "1300mm")
    width             : Dimension width (string, e.g. "1300mm")
    depth             : Dimension depth (string, e.g. "400mm")
    longitudinal_bars : Specification string for longitudinal bars
    lateral_bars      : Specification string for lateral bars
    """

    footing_type: str = Field(..., min_length=1, description="Footing type identifier — must be non-empty string")
    length: str = Field(..., min_length=1, description="Length with unit — must be non-empty string")
    width: str = Field(..., min_length=1, description="Width with unit — must be non-empty string")
    depth: str = Field(..., min_length=1, description="Depth with unit — must be non-empty string")
    longitudinal_bars: str = Field(..., min_length=1, description="Longitudinal reinforcement specs")
    lateral_bars: str = Field(..., min_length=1, description="Lateral reinforcement specs")

    model_config = {
        "extra": "forbid",
        "strict": True,
    }

    @model_validator(mode="before")
    @classmethod
    def reject_none_values(cls, values: dict) -> dict:
        """Reject payloads where any required field is explicitly set to null."""
        if isinstance(values, dict):
            for field_name, value in values.items():
                if value is None:
                    raise ValueError(
                        f"Field '{field_name}' must not be null. "
                        "All fields are required with non-null values."
                    )
        return values

