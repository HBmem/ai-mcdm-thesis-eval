from pydantic import BaseModel, ConfigDict

class Record(BaseModel):
    """Base contract for records crossing the AI boundary."""
    
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        frozen=True
    )