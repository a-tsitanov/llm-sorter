from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Action(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TranslateAction(_Action):
    type: Literal["translate"]
    to: str = Field(pattern=r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")  # язык: en, rus, pt-BR


class AnnotateAction(_Action):
    type: Literal["annotate"]


class SaveAction(_Action):
    type: Literal["save"]
    path: str | None = None


class RocketAction(_Action):
    type: Literal["rocket"]
    channel: str


Action = Annotated[
    TranslateAction | AnnotateAction | SaveAction | RocketAction,
    Field(discriminator="type"),
]
