#!/bin/bash
#
# Open up psql after starting Postgres database and handle killing on exit
#
. kamiak_config.sh

# Kill DB on exit
kill_db()
{
	if [[ ! -z $postgres_pid ]]; then
        echo "Stopping PostgreSQL"
		kill $postgres_pid
    fi
    exit
}
trap 'kill_db' EXIT

# Start DB
export PATH="$(pwd)/$bindir:$PATH"
export LD_LIBRARY_PATH="$(pwd)/$libdir:$LD_LIBRARY_PATH"
postgres -D "$remotedir/$mimicdir/data" &
postgres_pid=$!
sleep 5 # Make sure DB is up

# Interactive access to database
psql 'host=/tmp user=postgres dbname=mimic options=--search_path=mimiciii'
