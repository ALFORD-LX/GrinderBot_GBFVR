from enum import Enum, auto

class State(Enum):
    MENU = auto()
    VS_WAIT = auto()
    IN_FIGHT = auto()
    NEXT_FIGHT = auto()
    IDLE = auto()