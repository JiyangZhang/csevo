from typing import *

from recordclass import RecordClass


class MethodProjectRevision(RecordClass):

    prj_name: str = None
    revision: str = None
    method_ids: List[int] = None

    @classmethod
    def create(cls) -> "MethodProjectRevision":
        obj = MethodProjectRevision()
        obj.method_ids = list()
        return obj

    def fix_repr(self):
        if self.method_ids is None:  self.method_ids = list()
        return
