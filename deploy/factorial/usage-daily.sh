#!/bin/sh
# Keep each finished day's counts of factori.al's use, before the access log forgets the day.
# setup.sh installs this as /etc/cron.daily/factorial-usage; usage.py says what is counted.
exec /usr/local/bin/factorial-usage --keep
