from os import listdir, path
from tqdm import tqdm
from typing import *
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from seutil import LoggingUtils, IOUtils, BashUtils

from csevo.data.MethodData import MethodData
from csevo.Environment import Environment
from csevo.filter.AbstractFilter import AbstractFilter
from csevo.Macros import *


class BetaFilter(AbstractFilter):
    """Filter function for method naming task."""
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    def __init__(self):
        super(AbstractFilter, self).__init__()
        return

    def process_data_concurrent(self, data_dir=Macros.repos_results_dir):
        """Process data concurrently."""
        projects = [Path(data_dir/proj) for proj in listdir(data_dir)]
        num_proj = len(projects)
        processed = 0
        with ThreadPoolExecutor(8) as executor:
            futures = [executor.submit(self.process_data, proj) for proj in projects]
            for f in tqdm(as_completed(futures), total=num_proj):
                pass

    def process_data(self, project_dir):
        try:
            revision_data = IOUtils.load(project_dir/ "collector" / "method-project-revision.json")
            method_data = IOUtils.load(project_dir / "collector" / "method-data.json")
            output_dir = project_dir / "collector"
            method_project_evo = []
            for year in BetaFilter.YEARS[:-1]:
                curr_time = f"{year}_Jan_1"
                curr_method_ids = \
                    [year_data["method_ids"] for year_data in revision_data if year_data["year"] == curr_time][0]
                next_time = f"{year + 1}_Jan_1"
                next_method_ids = \
                    [year_data["method_ids"] for year_data in revision_data if year_data["year"] == next_time][0]
                new_method_ids = list(set(next_method_ids) - set(curr_method_ids))
                filtered_method_ids = BetaFilter.beta_filter(new_method_ids, curr_method_ids, method_data)
                method_project_evo.append({
                    "prj_name": revision_data[0]["prj_name"],
                    "time": f"{curr_time}-{next_time}",
                    "method_ids": filtered_method_ids
                })

            IOUtils.dump(output_dir/"method-project-beta-filtered.json", IOUtils.jsonfy(method_project_evo), IOUtils.Format.json)
            return
        except:
            self.logger.info(f"Unexpected error: {sys.exc_info()[0]}")
            return

    @staticmethod
    def beta_filter(new_method_ids: List[int], curr_method_ids: List[int], method_data) -> List[int]:
        filtered_method_ids = list()
        for method_id in new_method_ids:
            new_method = method_data[method_id]
            assert new_method["id"] == method_id
            filtered_method_id = -1
            for curr_method_id in curr_method_ids:
                curr_method = method_data[curr_method_id]
                assert curr_method["id"] == curr_method_id
                if new_method["code"] == curr_method["code"] and new_method["class_name"] == curr_method["class_name"]:
                    filtered_method_id = method_id
                    break
            if filtered_method_id == -1:
                filtered_method_ids.append(method_id)
        return filtered_method_ids