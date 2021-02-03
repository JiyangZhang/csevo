from typing import *

from datetime import datetime
from pathlib import Path
import pymongo
import pymongo.errors
import re
from tqdm import tqdm
import traceback
from urllib.error import HTTPError
from urllib.request import urlopen

from seutil import LoggingUtils, IOUtils, BashUtils, TimeUtils, GitHubUtils

from csevo.collector.Database import Database
from csevo.data.MethodData import MethodData
from csevo.data.MethodProjectRevision import MethodProjectRevision
from csevo.data.ProjectData import ProjectData
from csevo.Environment import Environment
from csevo.Macros import Macros


class DataCollector:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    MAX_REVISIONS = 50000
    # YEARS = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
    YEARS = [2020]
    def __init__(self, database: Optional[Database] = None):
        self.database = database

        self.repos_downloads_dir: Path = Macros.repos_downloads_dir
        self.repos_results_dir: Path = Macros.repos_results_dir
        IOUtils.mk_dir(self.repos_downloads_dir)
        IOUtils.mk_dir(self.repos_results_dir)

        # Load collected projects list
        collected_projects_file = Macros.data_dir/"projects-collected.txt"
        self.collected_projects_list = list()
        if collected_projects_file.exists():
            self.collected_projects_list += IOUtils.load(collected_projects_file, IOUtils.Format.txt).splitlines()
        # end if
        return

    def collect_projects(self, project_urls_file: Path,
            skip_collected: bool,
            beg: int = None,
            cnt: int = None,
    ):
        # 1. Load urls
        project_urls = IOUtils.load(project_urls_file, IOUtils.Format.txt).splitlines()
        # for deepcom projects:
        url_list = [f"https://github.com/{d.split('_')[0]}/{d.split('_')[1]}.git" for d in project_urls]
        project_urls = url_list

        invalid_project_urls = list()

        # Limit the number of projects to collect
        if beg is None:  beg = 0
        if cnt is None:  cnt = len(project_urls)

        project_urls = project_urls[beg:beg+cnt]

        for pi, project_url in enumerate(project_urls):
            self.logger.info(f"Project {beg+pi+1}/{len(project_urls)}({beg}-{beg+cnt}): {project_url}")

            try:
                # Project name is user_repo
                user_repo = self.parse_github_url(project_url)
                if user_repo is None:
                    self.logger.warning(f"URL {project_url} is not a valid GitHub repo URL.")
                    invalid_project_urls.append(project_url)
                    continue
                # end if
                project_name = f"{user_repo[0]}_{user_repo[1]}"

                if skip_collected and self.is_project_collected(project_name, project_url):
                    self.logger.info(f"Project {project_name} already collected.")
                    continue
                # end if

                # Query if the repo exists and is public on GitHub - private repo will block and waste time on git clone
                if not self.check_github_url(project_url):
                    self.logger.warning(f"Project {project_name} no longer available.")
                    invalid_project_urls.append(project_url)
                    continue
                # end if

                self.collect_project(project_name, project_url)
            except KeyboardInterrupt:
                self.logger.warning(f"KeyboardInterrupt")
                break
            except:
                self.logger.warning(f"Collection for project {project_url} failed, error was: {traceback.format_exc()}")
        # end for

        return

    def is_project_collected(self, project_name, project_url):
        return project_name in self.collected_projects_list or project_url in self.collected_projects_list

    def data_cut(self, data_size: int):
        """cut down the dataset to data_size, then save the projects list to data_dir"""
        collected_projects_file = Macros.data_dir / "projects-github.txt"
        self.collected_projects_list = list()
        if collected_projects_file.exists():
            self.collected_projects_list += IOUtils.load(collected_projects_file, IOUtils.Format.txt).splitlines()
        # end if
        project_name_list = list()
        for project_url in self.collected_projects_list:
            user_repo = self.parse_github_url(project_url)
            project_name_list.append(f"{user_repo[0]}_{user_repo[1]}")
        all_used_projects = [str(x).split("/")[-1] for x in Macros.repos_results_dir.iterdir() if x.is_dir()]
        # Find the overlapping projects and select the top data_size projects
        overall_project_num = 0
        reduced_project_list = list()
        for p in project_name_list:
            if p in all_used_projects and overall_project_num < data_size:
                # load the revision data
                filtered_methods = IOUtils.load(Macros.repos_results_dir/p/"collector"/"method-project-revision.json")
                new_method_ids = [delta_data["method_ids"] for delta_data in filtered_methods if delta_data["year"] == "2020_Jan_1"][0]
                if len(new_method_ids) > 0:
                    reduced_project_list.append(p)
                    overall_project_num += 1
                    all_used_projects.remove(p)
        IOUtils.dump(Macros.data_dir/f"projects-github-{data_size}.json", reduced_project_list, IOUtils.Format.jsonNoSort)

    def collect_project(self, project_name: str, project_url: str):
        Environment.require_collector()

        # 0. Download repo
        downloads_dir = self.repos_downloads_dir / project_name
        results_dir = self.repos_results_dir / project_name

        # Remove previous results if any
        IOUtils.rm_dir(results_dir)
        IOUtils.mk_dir(results_dir)

        # Clone the repo if not exists
        if not downloads_dir.exists():
            with IOUtils.cd(self.repos_downloads_dir):
                with TimeUtils.time_limit(300):
                    BashUtils.run(f"git clone {project_url} {project_name}", expected_return_code=0)
                # end with
            # end with
        # end if

        project_data = ProjectData.create()
        project_data.name = project_name
        project_data.url = project_url

        # 1. Get list of revisions
        with IOUtils.cd(downloads_dir):
            git_log_out = BashUtils.run(f"git log --pretty=format:'%H %P'", expected_return_code=0).stdout
            for line in git_log_out.splitlines()[:self.MAX_REVISIONS]:
                shas = line.split()
                project_data.revisions.append(shas[0])
                project_data.parent_revisions[shas[0]] = shas[1:]
            # end for
        # end with

        # 2. Get revisions in different year
        with IOUtils.cd(downloads_dir):
            for year in self.YEARS:
                git_log_out = BashUtils.run(f"git rev-list -1 --before=\"Jan 1 {year}\" origin",
                                            expected_return_code=0).stdout
                project_data.year_revisions[str(year)+"_Jan_1"] = git_log_out.rstrip()
            # end for
        # end with

        project_data_file = results_dir / "project.json"
        IOUtils.dump(project_data_file, IOUtils.jsonfy(project_data), IOUtils.Format.jsonPretty)

        # 2. Start java collector
        # Prepare config
        log_file = results_dir / "collector-log.txt"
        output_dir = results_dir / "collector"

        config = {
            "collect": True,
            "projectDir": str(downloads_dir),
            "projectDataFile": str(project_data_file),
            "logFile": str(log_file),
            "outputDir": str(output_dir),
            "year": True  # To indicate whether to collect all evo data or yearly data
        }
        config_file = results_dir / "collector-config.json"
        IOUtils.dump(config_file, config, IOUtils.Format.jsonPretty)

        self.logger.info(f"Starting the Java collector. Check log at {log_file} and outputs at {output_dir}")
        rr = BashUtils.run(f"java -jar {Environment.collector_jar} {config_file}", expected_return_code=0)
        if rr.stderr:
            self.logger.warning(f"Stderr of collector:\n{rr.stderr}")
        # end if

        # 3. In some cases, save collected data to appropriate location or database
        # TODO private info
        # On luzhou server for user pynie, move it to a dedicated location at /user/disk2
        if BashUtils.run(f"hostname").stdout.strip() == "luzhou" and BashUtils.run(f"echo $USER").stdout.strip() == "pynie":
            alter_results_dir = Path("/home/disk2/pynie/csevo-results")/project_name
            IOUtils.rm_dir(alter_results_dir)
            IOUtils.mk_dir(alter_results_dir.parent)
            BashUtils.run(f"mv {results_dir} {alter_results_dir}")
            self.logger.info(f"Results moved to {alter_results_dir}")
        # end if

        # -1. Remove repo
        IOUtils.rm_dir(downloads_dir)
        return

    RE_GITHUB_URL = re.compile(r"https://github\.com/(?P<user>[^/]+)/(?P<repo>.+?)(\.git)?")

    @classmethod
    def parse_github_url(cls, github_url) -> Tuple[str, str]:
        """
        Parses a GitHub repo URL and returns the user name and repo name. Returns None if the URL is invalid.
        """
        m = cls.RE_GITHUB_URL.fullmatch(github_url)
        if m is None:
            return None
        else:
            return m.group("user"), m.group("repo")
        # end if

    @classmethod
    def parse_projects(cls, project_list_file):
        """
        Parse the project list file provided by DeepCom and return the github url file.
        """
        project_list = IOUtils.load(project_list_file, IOUtils.Format.txt).splitlines()
        git_urls = list()
        for project in project_list:
            project_name = project.split("_", 1)
            git_urls.append(f"https://github.com/{project_name[0]}/{project_name[1]}.git")
        IOUtils.dump(Macros.data_dir/"DeepCom-projects-github.txt", "".join([url+"\n" for url in git_urls]),
                     IOUtils.Format.txt)

    @classmethod
    def urls_to_names(cls, project_urls: List[str]) -> List[str]:
        project_names = list()
        for url in project_urls:
            u, r = DataCollector.parse_github_url(url)
            project_names.append(f"{u}_{r}")
        # end for
        return project_names

    @classmethod
    def check_github_url(cls, github_url):
        try:
            urlopen(github_url)
            return True
        except HTTPError:
            return False
        # end try

    def get_github_top_repos(self):
        urls = list()
        stars = list()
        create_times = list()

        # 1000 top starred projects
        repositories = GitHubUtils.search_repos(q="topic:java language:java", sort="stars", order="desc", max_num_repos=1000)
        urls += [repo.clone_url for repo in repositories]
        stars += [repo.stargazers_count for repo in repositories]
        create_times += [datetime.timestamp(repo.created_at) for repo in repositories]

        IOUtils.dump(Macros.data_dir/"projects-github.txt", "".join([url+"\n" for url in urls]), IOUtils.Format.txt)
        IOUtils.dump(Macros.data_dir/"projects-github-stars.txt", "".join([str(star)+"\n" for star in stars]), IOUtils.Format.txt)
        IOUtils.dump(Macros.data_dir/"projects-github-create-times.txt", "".join([str(create_time)+"\n" for create_time in create_times]), IOUtils.Format.txt)
        return

    def store_repo_results(self, repos_results_dir: Path, force_update: bool = False):
        self.logger.info(f"Scanning {repos_results_dir} for results ...")
        pds = sorted(list(repos_results_dir.iterdir()))
        for pd in tqdm(pds):
            if not pd.is_dir():  continue

            project_name = pd.name

            if (pd/"project.json").exists() and (pd/"collector"/"method-data.json").exists() and (pd/"collector"/"method-project-revision.json").exists():
                if self.database.has_project(project_name):
                    if force_update:
                        self.logger.info(f"Removing existing data for project {project_name}")
                        self.database.remove_project(project_name)
                    else:
                        self.logger.info(f"Skipping project {project_name}")
                        continue
                    # end if
                # end if

                try:
                    self.store_repo_results_project(pd)
                except KeyboardInterrupt:
                    raise
                except pymongo.errors.OperationFailure as e:
                    self.logger.warning(f"Project {project_name} fail to add to database, pymongo error: {e.details}")
                    continue
                except:
                    self.logger.warning(f"Project {project_name} fail to add to database, exception: {traceback.format_exc()}")
            else:
                self.logger.warning(f"Project {project_name} has incomplete data at {pd}")
            # end if
        # end for
        return

    def store_repo_results_project(self, repo_results_project_dir: Path):
        # Assuming the project doesn't already exist in the database (or that does not matter), and the data in this directory is complete
        # Also fix some problems in collected files:
        #   1. some MethodData's prj_name == None issue here
        #   2. cap the number of revisions to consider to MAX_REVISIONS

        project_data = IOUtils.load(repo_results_project_dir / "project.json", IOUtils.Format.json)
        project_name = project_data["name"]

        project_data["revisions"] = project_data["revisions"][:self.MAX_REVISIONS]
        project_data["parent_revisions"] = {k: v for k, v in project_data["parent_revisions"].items() if k in project_data["revisions"]}
        IOUtils.dump(repo_results_project_dir / "project.json", project_data, IOUtils.Format.jsonPretty)

        method_project_revision_list = IOUtils.load(repo_results_project_dir / "collector" / "method-project-revision.json", IOUtils.Format.json)
        method_project_revision_list = method_project_revision_list[:self.MAX_REVISIONS]
        IOUtils.dump(repo_results_project_dir / "collector" / "method-project-revision.json", method_project_revision_list, IOUtils.Format.json)

        method_data_list = IOUtils.load(repo_results_project_dir / "collector" / "method-data.json", IOUtils.Format.json)
        method_data_indexed = set()
        method_data_idx_to_remove = list()

        with self.database.client.start_session() as session:
            with session.start_transaction():
                self.database.cl_project_data.insert_one(project_data, session=session)

                for method_project_revision in method_project_revision_list:
                    self.database.cl_method_project_revision.insert_one(method_project_revision, session=session)
                    method_data_indexed.update(method_project_revision["method_ids"])
                # end for

                for md_idx, method_data in enumerate(method_data_list):
                    if method_data["id"] not in method_data_indexed:
                        method_data_idx_to_remove.append(md_idx)
                        continue
                    # end if

                    if method_data["prj_name"] is None:
                        method_data["prj_name"] = project_name
                    # end if

                    self.database.cl_method_data.insert_one(method_data, session=session)
                    del method_data["_id"]
                # end for
            # end with
        # end with

        for md_idx in reversed(method_data_idx_to_remove):
            del method_data_list[md_idx]
        # end for
        IOUtils.dump(repo_results_project_dir / "collector" / "method-data.json", method_data_list, IOUtils.Format.jsonNoSort)
        return
