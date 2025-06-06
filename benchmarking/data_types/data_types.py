from enum import Enum


class ExtendedEnum(Enum):

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def enumList(cls):
        return list(map(lambda c: c, cls))


class IndexTypes(ExtendedEnum):
    CPU = "cpu"
    GPU = "gpu"

    @staticmethod
    def from_str(labelstr: str):
        if labelstr in "cpu":
            return IndexTypes.CPU
        elif labelstr in "gpu":
            return IndexTypes.GPU
        else:
            raise NotImplementedError


class WorkloadTypes(ExtendedEnum):
    INDEX = "index"
    SEARCH = "search"
    INDEX_AND_SEARCH = "index_and_search"

    @staticmethod
    def from_str(labelstr: str):
        if labelstr in "index":
            return WorkloadTypes.INDEX
        elif labelstr in "search":
            return WorkloadTypes.SEARCH
        elif labelstr in "index_and_search":
            return WorkloadTypes.INDEX_AND_SEARCH
        else:
            raise NotImplementedError
