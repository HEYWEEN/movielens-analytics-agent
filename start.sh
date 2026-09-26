#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runtime_dir="$project_dir/../.runtime"
hadoop_dir="$runtime_dir/hadoop-3.4.2"
bundled_jar="$hadoop_dir/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar"
mode=${1:-auto}

case "$mode" in
  auto|--local|--hadoop) ;;
  -h|--help)
    printf 'Usage: ./start.sh [--local|--hadoop]\n'
    printf '  default  Use configured Hadoop if available, otherwise local verification mode.\n'
    printf '  --local  Run the Python verification engine.\n'
    printf '  --hadoop  Require Hadoop Streaming; fail if unavailable.\n'
    printf 'Set LAB2_PORT to change the port (default: 8765).\n'
    exit 0 ;;
  *) printf 'Unknown option: %s\n' "$mode" >&2; exit 2 ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
  printf 'Python 3.9+ is required.\n' >&2
  exit 1
fi
if ! python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then
  printf 'Python 3.9+ is required.\n' >&2
  exit 1
fi

if [ ! -f "$project_dir/ml-1m/users.dat" ] ||
   [ ! -f "$project_dir/ml-1m/movies.dat" ] ||
   [ ! -f "$project_dir/ml-1m/ratings.dat" ]; then
  printf 'Warning: course data is missing from %s/ml-1m; the page will open, but tasks cannot run.\n' "$project_dir" >&2
fi

port=${LAB2_PORT:-8765}
case "$mode" in
  --local)
    printf 'Starting local verification mode at http://127.0.0.1:%s\n' "$port"
    export LAB2_ENGINE=local
    unset LAB2_HADOOP_LOCAL
    exec python3 "$project_dir/server.py" ;;
esac

if [ -n "${HADOOP_STREAMING_JAR:-}" ] && [ -f "$HADOOP_STREAMING_JAR" ] &&
   command -v hadoop >/dev/null 2>&1 && command -v hdfs >/dev/null 2>&1; then
  printf 'Starting configured Hadoop mode at http://127.0.0.1:%s\n' "$port"
  export LAB2_ENGINE=hadoop
  exec python3 "$project_dir/server.py"
fi

if [ -f "$bundled_jar" ] && command -v java >/dev/null 2>&1; then
  printf 'Starting bundled Hadoop standalone mode at http://127.0.0.1:%s\n' "$port"
  exec "$project_dir/hadoop_local.sh" server
fi

if [ "$mode" = --hadoop ]; then
  printf 'Hadoop is unavailable. Configure HADOOP_STREAMING_JAR, hadoop, and hdfs, or install the bundled runtime.\n' >&2
  exit 1
fi

printf 'Hadoop is unavailable; starting local verification mode at http://127.0.0.1:%s\n' "$port"
printf 'Use ./start.sh --hadoop for a Hadoop-only demonstration.\n'
export LAB2_ENGINE=local
unset LAB2_HADOOP_LOCAL
exec python3 "$project_dir/server.py"
