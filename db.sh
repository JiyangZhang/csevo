#!/bin/bash
# This script should only be executed by pynie on luzhou

readonly _DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

readonly DB_DIR="/home/disk2/pynie/csevo-db"
readonly DB_MAIN_DIR="${DB_DIR}/main"
readonly DB_DUMP_DIR="${DB_DIR}/dump"
readonly DB_LOG_FILE="${DB_DIR}/log.txt"
readonly DB_PORT=20144

mkdir -p ${DB_DIR}
mkdir -p ${DB_MAIN_DIR}
mkdir -p ${DB_DUMP_DIR}


function main() {
        local action=$1; shift

        case $action in
        server)
                mongod --dbpath "${DB_MAIN_DIR}" \
                       --port ${DB_PORT} \
                       --logpath "${DB_LOG_FILE}" \
                       --config ${_DIR}/db.conf
                ;;
        connect-local)
                mongo --port ${DB_PORT}
                ;;
        dump)
                local target="$(date +%Y%m%d-%s).tgz"
                ( cd ${DB_DIR}/
                  tar czf ${target} --owner=0 --group=0 main
                  mv ${target} ${DB_DUMP_DIR}
                  echo "Database dumped at ${DB_DUMP_DIR}/${target}"
                )
                ;;
        load)
                local dumpname=$1; shift
                if [[ -f ${DB_DUMP_DIR}/${dumpname}.tgz ]]; then
                        local bakcup_dir="${DB_DIR}/backup.$(date +%Y%m%d-%s)"
                        mv ${DB_MAIN_DIR} ${backup_dir}
                        ( cd ${DB_DUMP_DIR}
                          tar xzf ${dumpname}.tgz
                          mv main ${DB_MAIN_DIR}
                        )
                        echo "Database dump ${dumpname} loaded. Current one backed up at ${backup_dir}"
                else
                        echo "No file named ${dumpname}.tgz found in ${DB_DUMP_DIR}"
                fi
        esac
}


main "$@"
