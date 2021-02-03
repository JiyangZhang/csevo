from typing import *

import bson
import datetime
from enum import Enum
import inspect
from pathlib import Path
import pydoc
import pymongo
import pymongo.client_session
import pymongo.errors
import recordclass
import typing_inspect

from seutil import LoggingUtils

from csevo.data.MethodData import MethodData
from csevo.data.MethodProjectRevision import MethodProjectRevision
from csevo.data.ProjectData import ProjectData
from csevo.Environment import Environment
from csevo.Macros import Macros


class Database:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self,
            local: bool = False,
    ):
        if local:
            self.mongodb_url = f"mongodb://127.0.0.1:{Macros.mongodb_port}"
        else:
            self.mongodb_url = f"mongodb://{Macros.mongodb_server}:{Macros.mongodb_port}"
        # end if

        # Connect to database
        self.client = pymongo.MongoClient(self.mongodb_url)

        self.init_db_data()
        return

    @property
    def db_data(self):
        return self.client.get_database("data")

    @property
    def cl_project_data(self):
        return self.db_data["ProjectData"]

    @property
    def cl_method_data(self):
        return self.db_data["MethodData"]

    @property
    def cl_method_project_revision(self):
        return self.db_data["MethodProjectRevision"]

    def init_db_data(self):
        # Create indexes for databases
        self.cl_project_data.create_index([("name", pymongo.ASCENDING)], name="key", unique=True)
        self.cl_method_data.create_index([("prj_name", pymongo.ASCENDING), ("id", pymongo.ASCENDING)], name="key", unique=True)
        self.cl_method_project_revision.create_index([("prj_name", pymongo.ASCENDING), ("revision", pymongo.ASCENDING)], name="key", unique=True)
        return

    def has_project(self, project_name: str) -> bool:
        return self.cl_project_data.find_one({"name": project_name}) is not None

    def remove_project(self, project_name: str, session: Optional[pymongo.client_session.ClientSession] = None):
        # Remove all ProjectData, MethodData, and MethodProjectRevision data associated with this project
        self.cl_project_data.delete_one({"name": project_name}, session=session)
        self.cl_method_data.delete_many({"prj_name": project_name}, session=session)
        self.cl_method_project_revision.delete_many({"prj_name": project_name}, session=session)
        return

    def ls_projects(self) -> List[str]:
        return [d["name"] for d in self.cl_project_data.find({}, projection={"_id": False, "name": True})]

    def get_project_data(self, project_name: str) -> ProjectData:
        return Database.tr_bson_to_obj(self.cl_project_data.find_one({"name": project_name}), ProjectData)

    def get_method_data_list(self, project_name: str) -> List[MethodData]:
        return Database.tr_bson_to_obj(list(self.cl_method_data.find({"prj_name": project_name})), List[MethodData])

    def get_method_project_revision_list(self, project_name: str) -> List[MethodProjectRevision]:
        return Database.tr_bson_to_obj(list(self.cl_method_project_revision.find({"prj_name": project_name})), List[MethodProjectRevision])


    # ===== Utilities

    @classmethod
    def tr_obj_to_bson(cls, obj):
        if obj is None or isinstance(obj, (int, float, str, bool, bson.objectid.ObjectId, datetime.datetime)):
            # bson-compatible primitive data
            return obj
        elif isinstance(obj, (list, set, tuple)):
            # list-like data
            return [cls.tr_obj_to_bson(item) for item in obj]
        elif isinstance(obj, dict):
            # dict-like data, keys are converted to string
            return {str(k): cls.tr_obj_to_bson(v) for k, v in obj.items() if k != "_id"}
        elif isinstance(obj, Enum):
            # Enum, use the underlying integer
            return obj.value
        elif hasattr(obj, "tr_obj_to_bson"):
            # Use transforming function if available
            return getattr(obj, "tr_obj_to_bson")(obj)
        elif isinstance(obj, recordclass.mutabletuple):
            # RecordClass, treat as dict
            return {k: cls.tr_obj_to_bson(v) for k, v in obj.__dict__.items() if k != "_id"}
        else:
            # Last effort, to str
            return str(obj)
        # end if

    @classmethod
    def tr_bson_to_obj(cls, data, clz=None):
        # print(f"tr_bson_to_obj: {data}, {clz}")
        if isinstance(clz, str):
            clz = pydoc.locate(clz)
        # end if

        if data is None:
            return None
        # end if

        # Try to convert as list
        if clz is not None:
            t = typing_inspect.get_origin(clz)
            if t in (list, set, tuple):
                l = [cls.tr_bson_to_obj(item, clz.__args__[0]) for item in data]
                if t == set:
                    l = set(l)
                elif t == tuple:
                    l = tuple(l)
                # end if
                return l
            elif isinstance(data, list):
                return [cls.tr_bson_to_obj(item, clz) for item in data]
            # end if
        # end if

        # Try to use transforming function
        if clz is not None and hasattr(clz, "tr_bson_to_obj"):
            return getattr(clz, "tr_bson_to_obj")(data)
        # end if

        # Try other types
        if clz is not None and inspect.isclass(clz):
            if issubclass(clz, recordclass.mutabletuple):
                # RecordClass
                field_values = dict()
                for f, t in get_type_hints(clz).items():
                    if f == "_id":
                        field_values["_id"] = data.get("_id")
                    elif f in data:
                        field_values[f] = cls.tr_bson_to_obj(data.get(f), t)
                    # end if
                # end for
                return clz(**field_values)
            elif issubclass(clz, Enum):
                # Enum
                return clz(data)
            elif clz == bson.objectid.ObjectId or clz == datetime.datetime:
                # Bson compatible types
                return data
            else:
                # Try constructor
                try:
                    return clz(data)
                except:
                    pass
                # try
            # end if
        # end if

        if isinstance(data, dict):
            # dict
            return {k: cls.tr_bson_to_obj(v, clz) for k, v in data.items()}
        else:
            return data
        # end if

