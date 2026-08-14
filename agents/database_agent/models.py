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


class FoundationElementRecord(BaseModel):
    """
    Represents one row in the salit_steel_foundation_diagram AlloyDB table.

    Fields
    ------
    element_type                  : Primary key (e.g. "F1", "P1", "C1", "BP1")
    category                      : Category (e.g. "Spread footing pad", "Concrete pier", "Steel column", "Base plate")
    length_mm                     : Length dimension string (e.g. "1300", "500", "203")
    width_mm                      : Width dimension string (e.g. "1300", "500", "203")
    depth_mm                      : Depth / thickness string (e.g. "400", "per section", "7.9 (wall thk.)")
    primary_reinforcement         : Primary reinforcement specification string
    secondary_reinforcement_notes : Secondary reinforcement / ties / notes specification string
    plan_instances                : QTY diagram count on layout plan (e.g. 33, 6, 0)
    """

    element_type: str = Field(..., min_length=1, description="Element type identifier (e.g. 'F1', 'P1', 'C1', 'BP1')")
    category: str = Field(default="", description="Element category description")
    length_mm: str = Field(default="", description="Length in mm")
    width_mm: str = Field(default="", description="Width in mm")
    depth_mm: str = Field(default="", description="Depth / thickness in mm")
    primary_reinforcement: str = Field(default="", description="Primary reinforcement specs")
    secondary_reinforcement_notes: str = Field(default="", description="Secondary reinforcement & notes")
    plan_instances: int = Field(default=0, description="Diagram plan instance count (QTY)")


