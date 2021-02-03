from typing import *

from recordclass import RecordClass


class MethodData(RecordClass):

    prj_name: str = None
    id: int = None

    # A combination of keys is unique inside a project

    name: str = None
    return_type: str = None
    params: List[Tuple[str, str]] = None

    code: str = None  # Key

    comment: str = None  # Key
    comment_summary: str = None

    class_name: str = None  # Key
    path: str = None

    @classmethod
    def create(cls) -> "MethodData":
        obj = MethodData()
        obj.params = list()
        return obj

    def fix_repr(self):
        if self.params is None:
            self.params = list()
        # end if
        return

    # TODO: Add this property in collector instead, when we're running raw data collection the next time
    def is_abstract(self):
        # Approximate by if code ends with ";"
        return self.code.rstrip().endswith(";")
