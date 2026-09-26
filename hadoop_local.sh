#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
runtime_dir="$project_dir/../.runtime"
hadoop_dir="$runtime_dir/hadoop-3.4.2"
streaming_jar="$hadoop_dir/share/hadoop/tools/lib/hadoop-streaming-3.4.2.jar"

if [ ! -f "$streaming_jar" ]; then
  echo "Hadoop 3.4.2 not found at $hadoop_dir" >&2
  exit 1
fi

if [ -z "${JAVA_HOME:-}" ]; then
  if [ -x /usr/libexec/java_home ]; then
    JAVA_HOME=$(/usr/libexec/java_home)
  else
    echo "Set JAVA_HOME to a compatible JDK before starting Hadoop." >&2
    exit 1
  fi
fi
export JAVA_HOME
export PATH="$hadoop_dir/bin:$PATH"
export HADOOP_STREAMING_JAR="$streaming_jar"
export LAB2_HADOOP_LOCAL=1
export LAB2_HDFS_BASE="$runtime_dir/hadoop-jobs"
export LAB2_ENGINE=hadoop

case "${1:-server}" in
  server) exec python3 "$project_dir/server.py" ;;
  run) exec python3 "$project_dir/runner.py" --engine hadoop ;;
  *) echo "Usage: $0 [server|run]" >&2; exit 2 ;;
esac
