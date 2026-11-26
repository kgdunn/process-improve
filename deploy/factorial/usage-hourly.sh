#!/bin/sh
# Write how many readers each of the last 90 days had, numbers alone, where the footer of every page reads them.
# setup.sh installs this as /etc/cron.hourly/factorial-usage-public; usage.py says what is written (--public).
exec /usr/local/bin/factorial-usage --public /var/lib/factorial/public/usage.json
