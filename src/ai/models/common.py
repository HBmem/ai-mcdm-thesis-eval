from pydantic import BaseModel, ConfigDict

class AIRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        frozen=True
    )