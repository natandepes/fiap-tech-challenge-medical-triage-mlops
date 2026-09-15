from enum import IntEnum, StrEnum


class Urgency(StrEnum):
    NORMAL = "normal"
    ATTENTION = "attention"
    URGENT = "urgent"


class Condition(IntEnum):
    NEOPLASMS = 1
    DIGESTIVE_SYSTEM = 2
    NERVOUS_SYSTEM = 3
    CARDIOVASCULAR = 4
    GENERAL_PATHOLOGICAL = 5
