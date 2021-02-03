from typing import *

from recordclass import RecordClass


class ProjectData(RecordClass):

    name: str = None
    url: str = None

    # The output of `git log --pretty=format="%H"`, from new to old.  Not all consequent elements in this list are parent-child
    revisions: List[str] = None

    # The output of `git log --pretty=format="%H %P"`, for recovering the diff in each commit
    parent_revisions: Dict[str, List[str]] = None

    # The SHA-1 of project at specific time
    year_revisions: Dict[str, str] = dict()

    @classmethod
    def create(cls) -> "ProjectData":
        obj = ProjectData()
        obj.revisions = list()
        obj.parent_revisions = dict()
        return obj

    def fix_repr(self):
        if self.revisions is None:
            self.revisions = list()
            self.parent_revisions = dict()
        # end if
        return
