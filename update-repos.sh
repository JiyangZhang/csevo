#!/bin/bash
# This script clones/updates the auxiliary data and results repositories.

readonly _DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly DATA_REPO_NAME="csevo-data"
readonly DATA_BRANCH_NAME="data"
readonly RESULTS_REPO_NAME="csevo-results"
readonly RESULTS_BRANCH_NAME="results"

function update_repo() {
        local repo_name="$1"; shift
        local branch_name="$1"; shift
        
        ( cd "$_DIR"
          if [[ -e "${repo_name}" ]]; then
                  if [[ ! -d "${repo_name}" || ! -d "${repo_name}/.git" ]]; then
                          echo "ERROR: Path ${repo_name} already exists but is not a proper git repository!" 1>&2
                          return -1
                  fi

                  ( cd "${repo_name}"
                    git pull
                    if [[ $? -ne 0 ]]; then
                            echo "ERROR: ${repo_name} update fail, please fix errors!" 1>&2
                            return -1
                    fi
                  )
          else
                  mkdir "${repo_name}"
                  local git_url="$(git config --get remote.origin.url)"
                  ( cd "${repo_name}"
                    git clone --single-branch -b "${branch_name}" -- "${git_url}" .
                    if [[ $? -ne 0 ]]; then
                            echo "ERROR: ${repo_name} clone fail, please fix errors!" 1>&2
                            return -1
                    fi
                  )
          fi
          
        )
}

update_repo "${DATA_REPO_NAME}" "${DATA_BRANCH_NAME}"
update_repo "${RESULTS_REPO_NAME}" "${RESULTS_BRANCH_NAME}"
