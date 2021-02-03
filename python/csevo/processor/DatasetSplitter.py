from typing import *

import collections
import copy
import random
from tqdm import tqdm

from seutil import LoggingUtils, IOUtils

from csevo.collector.Database import Database
from csevo.collector.DataCollector import DataCollector
from csevo.data.MethodData import MethodData
from csevo.data.MethodProjectRevision import MethodProjectRevision
from csevo.data.ProjectData import ProjectData
from csevo.Environment import Environment
from csevo.Macros import Macros


class DatasetSplitter:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    def __init__(self, database: Database):
        self.database = database
        self.output_dir = Macros.data_dir / "split"
        IOUtils.rm_dir(self.output_dir)
        IOUtils.mk_dir(self.output_dir)

        self.statistics = dict()
        return

    def split_dataset(self):
        """
        Splits the dataset randomly to train/val/test subsets by project, such that:
        1. each project appears in and only appears in one of the subset;
        2. set size (number of evolution-aware data) of train:val:test is roughly 8:1:1;
        3. duplication of <code, comment> pairs are removed.
        """
        # 0. Clean up statistics dict
        self.statistics.clear()

        # 1. Figure out which projects are available
        project_names = self.get_available_projects()

        # 2. Load data and perform data cleaning: remove abstract methods; remove duplicates
        project_names, project_2_data_ids_evolution, project_2_data_ids_latest = self.load_and_clean_data(project_names)

        # 3. Do the splitting
        split_data_ids_evolution, split_data_ids_latest, split_project_names = self.perform_splitting(project_names, project_2_data_ids_evolution, project_2_data_ids_latest)

        # 4. Save the splitting info to file
        for data_type in [Macros.train, Macros.val, Macros.test]:
            IOUtils.dump(self.output_dir / f"evolution-{data_type}.json", split_data_ids_evolution[data_type], IOUtils.Format.json)
            IOUtils.dump(self.output_dir / f"latest-{data_type}.json", split_data_ids_latest[data_type], IOUtils.Format.json)
            IOUtils.dump(self.output_dir / f"projects-{data_type}.txt", "".join([p+"\n" for p in split_project_names[data_type]]), IOUtils.Format.txt)
        # end for

        # 5. Get small version dataset with 10% downsampling
        sampled_split_data_ids_evolution, sampled_split_data_ids_latest = self.downsample(split_data_ids_evolution, split_data_ids_latest, split_project_names, sample_rate=0.1, sample_name="small")
        for data_type in [Macros.train, Macros.val, Macros.test]:
            IOUtils.dump(self.output_dir / f"evolution-{data_type}-small.json", sampled_split_data_ids_evolution[data_type], IOUtils.Format.json)
            IOUtils.dump(self.output_dir / f"latest-{data_type}-small.json", sampled_split_data_ids_latest[data_type], IOUtils.Format.json)
        # end for

        # 6. Save the splitting statistics to results_dir
        IOUtils.dump(self.output_dir / f"statistics.json", self.statistics, IOUtils.Format.jsonPretty)

        return

    def get_available_projects(self) -> List[str]:
        project_urls = IOUtils.load(Macros.data_dir/"projects.txt", IOUtils.Format.txt).splitlines()
        project_names = DataCollector.urls_to_names(project_urls)

        project_names_in_db = self.database.ls_projects()
        projects_not_collected = [p for p in project_names if p not in project_names_in_db]
        if len(projects_not_collected) > 0:
            self.logger.warning(f"Ignoring {len(projects_not_collected)} projects whose data is not collected.")
            IOUtils.dump(self.output_dir/"projects-not-collected.txt", "".join([p+"\n" for p in projects_not_collected]), IOUtils.Format.txt)
        # end if
        project_names = [p for p in project_names if p in project_names_in_db]

        return project_names

    def load_and_clean_data(self, project_names: List[str]) -> Tuple[
        List[str],  # project_names
        Dict[str, List[int]],  # project_2_data_ids_evolution
        Dict[str, List[int]],  # project_2_data_ids_latest
    ]:
        # Perform two cleanings:
        # 1. Remove duplicates: keep first seen data, by order specified by projects list and method id in each project;
        # 2. Remove abstract methods;
        projects_no_collected_data = list()
        projects_no_data_after_cleaning = list()
        project_2_data_ids_evolution: Dict[str, List[int]] = dict()
        project_2_data_ids_latest: Dict[str, List[int]] = dict()
        self.statistics["removed-abstract-method"] = 0
        self.statistics["removed-duplicates-within-project"] = 0
        self.statistics["removed-duplicates-between-projects"] = 0
        seen_data_hashes: Set[int] = set()

        for prj_name in tqdm(project_names):
            project_2_data_ids_evolution[prj_name] = list()
            method_data_list: List[MethodData] = self.database.get_method_data_list(prj_name)
            if len(method_data_list) == 0:
                projects_no_collected_data.append(prj_name)
                continue
            # end if

            prj_seen_data_hashes: Set[int] = set()

            for method_data in method_data_list:
                # Remove abstract methods
                if method_data.is_abstract():
                    self.statistics["removed-abstract-method"] += 1
                    continue
                # end if

                # Remove duplicate methods within project
                h = hash((method_data.code, method_data.comment_summary))
                if h in prj_seen_data_hashes:
                    self.statistics["removed-duplicates-within-project"] += 1
                    continue
                # end if
                prj_seen_data_hashes.add(h)

                # Remove duplicate methods across projects
                if h in seen_data_hashes:
                    self.statistics["removed-duplicates-between-projects"] += 1
                    continue
                # end if
                seen_data_hashes.add(h)

                project_2_data_ids_evolution[prj_name].append(method_data.id)
            # end for

            if len(project_2_data_ids_evolution[prj_name]) == 0:
                projects_no_data_after_cleaning.append(prj_name)
                del project_2_data_ids_evolution[prj_name]
                continue
            # end if

            # Obtain the data ids for the latest revision
            project_data: ProjectData = self.database.get_project_data(prj_name)
            latest_revision = project_data.revisions[0]
            method_project_revision: MethodProjectRevision = Database.tr_bson_to_obj(self.database.cl_method_project_revision.find_one({"prj_name": prj_name, "revision": latest_revision}), MethodProjectRevision)
            latest_data_ids = method_project_revision.method_ids
            project_2_data_ids_latest[prj_name] = [did for did in project_2_data_ids_evolution[prj_name] if did in latest_data_ids]
        # end for

        # Report about no data projects
        if len(projects_no_collected_data) > 0:
            self.logger.warning(f"Ignoring {len(projects_no_collected_data)} projects who have no data collected.")
            IOUtils.dump(self.output_dir/"projects-no-collected-data.txt", "".join([p+"\n" for p in projects_no_collected_data]), IOUtils.Format.txt)
            project_names = [p for p in project_names if p not in projects_no_collected_data]
        # end if
        if len(projects_no_data_after_cleaning) > 0:
            self.logger.warning(f"Ignoring {len(projects_no_data_after_cleaning)} projects who have no data after cleaning.")
            IOUtils.dump(self.output_dir/"projects-no-data-after-cleaning.txt", "".join([p+"\n" for p in projects_no_data_after_cleaning]), IOUtils.Format.txt)
            project_names = [p for p in project_names if p not in projects_no_data_after_cleaning]
        # end if

        return project_names, project_2_data_ids_evolution, project_2_data_ids_latest

    def perform_splitting(self, project_names: List[str], project_2_data_ids_evolution: Dict[str, List[int]], project_2_data_ids_latest: Dict[str, List[int]]) -> Tuple[
        Dict[str, List[Tuple[str, List[int]]]],  # split_data_ids_evolution
        Dict[str, List[Tuple[str, List[int]]]],  # split_data_ids_latest
        Dict[str, List[str]],  # split_project_names
    ]:
        split_data_ids_evolution: Dict[str, List[Tuple[str, List[int]]]] = collections.defaultdict(list)
        split_data_ids_latest: Dict[str, List[Tuple[str, List[int]]]] = collections.defaultdict(list)
        split_project_names: Dict[str, List[str]] = collections.defaultdict(list)

        # Reset random seed to ensure reproducibility
        random.seed(Environment.random_seed)

        # Shuffle the projects
        random.shuffle(project_names)
        data_count = sum([len(l) for l in project_2_data_ids_evolution.values()])

        # Prepare statistics (and counters)
        for data_type in [Macros.train, Macros.val, Macros.test]:
            self.statistics[f"num-data-evolution-{data_type}"] = 0
            self.statistics[f"num-data-latest-{data_type}"] = 0
            self.statistics[f"num-project-{data_type}"] = 0
        # end for

        def add_project(data_type: str):
            pn = project_names.pop()
            data_ids_evolution = project_2_data_ids_evolution[pn]
            split_data_ids_evolution[data_type] += [(pn, data_ids_evolution)]
            self.statistics[f"num-data-evolution-{data_type}"] += len(data_ids_evolution)

            data_ids_latest = project_2_data_ids_latest[pn]
            split_data_ids_latest[data_type] += [(pn, data_ids_latest)]
            self.statistics[f"num-data-latest-{data_type}"] += len(data_ids_latest)

            split_project_names[data_type].append(pn)
            self.statistics[f"num-project-{data_type}"] += 1
            return

        # Take projects to train set until it exceeds train_ratio
        while self.statistics[f"num-data-evolution-{Macros.train}"] / data_count < self.TRAIN_RATIO:
            add_project(Macros.train)
        # end while

        # Take projects to test set until it exceeds test_ratio
        while self.statistics[f"num-data-evolution-{Macros.test}"] / data_count < self.TEST_RATIO:
            add_project(Macros.test)
        # end while

        # Take remaining projects to val set
        while len(project_names) > 0:
            add_project(Macros.val)
        # end while
        return split_data_ids_evolution, split_data_ids_latest, split_project_names

    def downsample(self,
            split_data_ids_evolution: Dict[str, List[Tuple[str, List[int]]]],
            split_data_ids_latest: Dict[str, List[Tuple[str, List[int]]]],
            split_project_names: Dict[str, List[str]],
            sample_rate: float,
            sample_name: str = "small",
    ) -> Tuple[
        Dict[str, List[Tuple[str, List[int]]]],  # sampled_split_data_ids_evolution
        Dict[str, List[Tuple[str, List[int]]]],  # sampled_split_data_ids_latest
    ]:
        """
        Downsamples the dataset to get smaller dataset.

        Ensures each project has at least one data remaining in the set.
        Tries not to change the projects data distribution.
        Tries to down-sample by the specified rate, but may retain a bit more data.
        Ensures that after sampling, any data in split_data_ids_latest is in split_data_ids_evolution.
        """
        sampled_split_data_ids_evolution: Dict[str, List[Tuple[str, List[int]]]] = dict()
        sampled_split_data_ids_latest: Dict[str, List[Tuple[str, List[int]]]] = dict()

        # Reset random seed to ensure reproducibility
        random.seed(Environment.random_seed)

        for data_type in [Macros.train, Macros.val, Macros.test]:
            # Prepare final results
            sampled_split_data_ids_evolution[data_type] = list()
            sampled_split_data_ids_latest[data_type] = list()

            # Prepare statistics
            self.statistics[f"num-data-evolution-{data_type}-{sample_name}"] = 0
            self.statistics[f"num-data-latest-{data_type}-{sample_name}"] = 0

            data_ids_evolution: List[Tuple[str, List[int]]] = split_data_ids_evolution[data_type]
            data_ids_latest: List[Tuple[str, List[int]]] = split_data_ids_latest[data_type]

            # Sample for each project
            for pn in split_project_names[data_type]:
                data_ids_project_evolution: List[int] = [pn_dids[1] for pn_dids in data_ids_evolution if pn_dids[0] == pn][0]
                data_ids_project_latest: List[int] = [pn_dids[1] for pn_dids in data_ids_latest if pn_dids[0] == pn][0]

                # Sample the data in latest, add to both sampled-evolution and sampled-latest set
                orig_size_latest = len(data_ids_project_latest)
                # Sample size: at least 1, unless original size is 0
                sample_size_latest = min(max(int(sample_rate*orig_size_latest), 1), orig_size_latest)
                sampled_data_ids_project_latest = random.sample(data_ids_project_latest, sample_size_latest)
                sampled_data_ids_project_evolution = copy.copy(sampled_data_ids_project_latest)

                # Sample the data in evolution but not in latest, add to sampled-evolution set
                data_ids_project_evolution_only = [did for did in data_ids_project_evolution if did not in data_ids_project_latest]
                orig_size_evolution_only = len(data_ids_project_evolution_only)
                # Sample size: at least 1, unless original size is 0
                sample_size_evolution_only = min(max(int(sample_rate*orig_size_evolution_only), 1), orig_size_evolution_only)
                sampled_data_ids_project_evolution.extend(random.sample(data_ids_project_evolution_only, sample_size_evolution_only))
                sampled_data_ids_project_evolution.sort()

                # Update statistics and add to final results
                sampled_split_data_ids_evolution[data_type].append((pn, sampled_data_ids_project_evolution))
                sampled_split_data_ids_latest[data_type].append((pn, sampled_data_ids_project_latest))
                self.statistics[f"num-data-evolution-{data_type}-{sample_name}"] += len(sampled_data_ids_project_evolution)
                self.statistics[f"num-data-latest-{data_type}-{sample_name}"] += len(sampled_data_ids_project_latest)
            # end for
        # end for

        return sampled_split_data_ids_evolution, sampled_split_data_ids_latest
