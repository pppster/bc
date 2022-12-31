from enum import Enum


class BCImageEnum(Enum):
    IMAGE = 1
    MASK = 2
    NOBACKGROUND = 3


IMAGE = BCImageEnum.IMAGE
MASK = BCImageEnum.MASK
NOBACKGROUND = BCImageEnum.NOBACKGROUND
