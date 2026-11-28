# factori.al

Factorial, the browser app in [`web/`](../../web), is served from https://factori.al/: plain
files on the server that ran the old kgdunn/factorial app, behind the same Caddy. Each push to
`main` builds the app and starts it in Chromium; once it passes, the `deploy-factorial` job in
[`docs.yml`](../../.github/workflows/docs.yml) sends that same build to the server, and
`check-factorial` runs the smoke tests against factori.al itself.

## How a deploy works

- The workflow logs in as `factorial` with a key that can run one command,
  [`factorial-deploy`](factorial_deploy.py), whatever it asks: upload a build, switch to it,
  roll back, or report which build is live. It cannot open a shell, copy files, or forward a
  port.
- Each build gets a folder, `/srv/factorial/releases/<commit>`, and the link
  `/srv/factorial/current` names the one Caddy serves. A switch is one rename of that link:
  a reader gets one build or the other, never a mix, and nothing restarts or reloads.
- A build must be the app built from the commit it is uploaded as (its `wheels/manifest.json`
  names it), and may hold only files and folders, inside its own folder.
- The last five builds stay on the server. **Actions → Factorial: roll back → Run workflow**
  switches back to the one before, or to a commit you name.

## The copy a reader's browser keeps

The app registers a service worker, [`web/sw.js`](../../web/sw.js), which serves a copy of the
page kept in the reader's browser, so it opens without a network. The worker holds no list of
files and no version, and reads none: the app's own files come from this server whenever it
answers, so a deploy or a roll back reaches a reader at their next visit as before, and only
files whose address names their version (the wheel's folder, Pyodide's release) are served from
the copy. The page itself makes the copy and drops what earlier builds left in it
([`web/offline.js`](../../web/offline.js)).

Should the worker ever serve a page wrongly, there is a way out that needs no fix to it. Build
the app with `--no-service-worker` and deploy that build:

```bash
uv run python scripts/build_web_app.py -o site --no-service-worker
```

Its `sw.js` is [`web/sw-retire.js`](../../web/sw-retire.js). A browser that has the worker asks
for `sw.js` again at the reader's next visit, whatever the page's own code does, and gets that
one: it deletes the copies and unregisters. The build's `wheels/manifest.json` says
`"service_worker": false`, so the page registers none again. A later build without the flag
brings the worker back. A roll back to a build from before the worker existed leaves a reader's
worker in place and harmless: it goes on fetching the app's files from the server.

The build asks PyPI once for the exact files of the packages the page installs on demand (plotly,
openpyxl), and writes their addresses into the manifest: the page then installs those files, and
asks no package index. A build that cannot reach PyPI says so and goes on without them; its page
installs the same pins through the index, and offers no copy of the charts and workbooks.

[`factorial.caddy`](factorial.caddy) sends everything under `wheels/<digest>/` with a year's
`Cache-Control`, since that folder's address changes with its contents. A server set up before
that rule covered the page's own Python as well as the wheel serves it correctly all the same,
revalidated at each visit: run `setup.sh` again, and then the command it ends with, to install the rule.

## What the server keeps

Usage, and the encrypted study of each room, which it cannot read ([Rooms](#rooms)). Caddy writes one line per request to
`/var/log/caddy/factorial-access.log`: the time, the path, the status, the browser's
`User-Agent`, and the page that linked here. The reader's address is cut to its network
before the line is written (the first 24 bits of IPv4, 48 of IPv6), and lines are kept for
366 days, so a year can be set beside the one before. There are no cookies, no tracking script
and no database. A design shared by link
travels after the `#`, which a browser never sends, so no design reaches the log.

So that usage can be followed for longer than the log is kept, a daily job adds each finished
day's counts to `/var/lib/factorial/usage.csv`: a row a day of numbers (each page's views,
readers, Python starts, link previews), with no address, browser or path of anyone's in it.

The footer of every page draws how many readers each of the last 90 finished days had, then the
last 24 hours as the line's last point, with that count beside it: the last 24 hours rather than
today so far, which would start each day near nothing. An hourly job writes those numbers, and when they were written, to
`/var/lib/factorial/public/usage.json`, which Caddy serves as `https://factori.al/usage.json` in
place of the build's own, a file of no days. It reads the kept counts and a day or two of the log,
not a year of it:

```bash
sudo factorial-usage --public /var/lib/factorial/public/usage.json   # what the hourly job runs
```

The one exception is a report a reader sends with the app's Feedback button, and only when they
press Send: the relay emails it to the maintainer, and keeps no copy and no line of it
([Feedback](#feedback)). Its request is logged as any other, without its body. A reader who gives
an address gets a receipt of fixed text; to send an address at most one a day, the relay holds a
keyed hash of it in memory for a day.

The other exception is a study a reader shares in a room: the encrypted study locker keeps it, encrypted in the
reader's browser with a key the server never holds ([Rooms](#rooms)). A room's requests are logged
as any other, with neither their body nor the header that names the room.

```bash
sudo factorial-usage                        # the last 28 days, a row a day
sudo factorial-usage --days 90 --by week    # a row a week
sudo factorial-usage --days 366 --by month  # a year, a row a month
```

Each page has a column: the app (Home), the concept map (Map), the design catalog's pages
(Catalog), the multivariate page, About, the list of what Factorial does (Features), and any
other page (Other). Readers are distinct networks and browsers in a day, and over a week or a
month their average a day; Python counts the app's worker starting Python. Under the table, a
line of bars for each page shows its use over the period. Robots, link-preview fetchers and the deploy's own checks are not counted.

Under the bars, the report says how long Python took to start, from opening the app to Python
ready: the median, and the time 9 in 10 starts took no longer than, for first visits and
returning readers apart. The log can time a start only when the browser downloaded the app's
Python (`wheels/<digest>/bootstrap.zip`), which is the last file the worker asks factori.al for:
a reader's first visit, and their first after a deploy. A returning reader's browser already has
Pyodide, so the gap between the two rows is what downloading it adds. A start that finds the
archive kept asks for nothing to time. The time leaves out the second or so it takes to unpack
and import the archive.

Then it splits each start in three, where the log holds the worker's requests that mark the
seams: the page, from opening it to the worker reading `runtime.json`; Pyodide and its packages,
from there to the worker asking for the process-improve wheel; and process-improve, from there
to the end of the archive's download. Each part's median is given, so the parts need not add up
to the median start.

```text
From opening the app to Python ready, where the browser downloaded the app's Python:
  First visit  median 8.0 s, 9 in 10 within 30.0 s, of 5
  Returning    median 2.5 s, 9 in 10 within 3.0 s, of 2

Where the time went, the median of each part (medians, so they need not add up to the one above):
  First visit  page 1.5 s, Pyodide 6.0 s, process-improve 1.5 s, of 2
```

Last, the report lists where readers came from: the site that linked here, as the browser names
it. Many apps open a link in a browser of their own that names no site, so a link you post can
name the place itself:

```text
https://factori.al/?ref=linkedin
https://factori.al/?example=worked&ref=newsletter
```

The report lists such visits as `?ref=linkedin`, apart from the rest. A tag is up to 24 letters,
digits, `-` or `_`, read in lower case; any other is not counted. Use a word for a place, never
one for each person: the report counts visits, not people. The app drops the tag from its
address once open, so a reload, or the address copied on, is not counted again.

A place is listed only for a reader whose browser went on to ask for the page's stylesheet or
scripts. That keeps referrer spam out: a robot that sends a made-up site as the page that
linked here, to be listed in reports like this one, asks for the page alone. Its page views are
still in the table. A robot that runs a whole browser would be listed.
The report reads the log for the days it holds and the kept counts for older days. It is the
`factorial-usage` that `setup.sh` installed: run `setup.sh` again to install a newer one.

## Setting it up, once

**1. The files and a deploy key**, on your machine, in a folder outside the repository so the
key cannot be committed (`~/process-improve` stands for your checkout):

```bash
mkdir -p ~/factorial-deploy && cd ~/factorial-deploy
git -C ~/process-improve fetch origin
git -C ~/process-improve archive origin/main deploy/factorial | tar -x --strip-components=1
ssh-keygen -t ed25519 -N '' -C 'github-actions@kgdunn/process-improve' -f factorial_deploy
cp factorial_deploy.pub factorial/
scp -r factorial deploy@factori.al:factorial-setup
```

The folder goes to `~/factorial-setup`, not `~/factorial`, where the old app is checked out.

**2. The server**, logged in as `deploy`:

```bash
sudo sh factorial-setup/setup.sh factorial-setup/factorial_deploy.pub
```

[`setup.sh`](setup.sh) makes the `factorial` account and `/srv/factorial`, installs
`factorial-deploy` and `factorial-usage` in `/usr/local/bin`, puts the key where only root can
change it, and installs the SSH block ([`sshd.conf`](sshd.conf)) once `sshd -t` accepts it.
It installs the site's Caddy block, [`factorial.caddy`](factorial.caddy), and makes its log
owned by Caddy. It installs the feedback relay and its unit, and makes the relay's settings files
once ([Feedback](#feedback)). It installs the encrypted study locker and its unit, and starts it ([Rooms](#rooms)). It ends by printing the value of `FACTORIAL_KNOWN_HOSTS` for step 5.
It needs the server's Python to be 3.12 or later, as Ubuntu 24.04's is. Running it again is safe.

**3. Check the key**, from your machine:

```bash
ssh -o IdentitiesOnly=yes -i ~/factorial-deploy/factorial_deploy factorial@factori.al status
```

It prints `live: none` and `kept:`: the key logs in, and runs the deploy command. Without
`status` it answers `refused: this key may only: ...`.

**4. Caddy.** In `/etc/caddy/Caddyfile`, replace the old app's whole `factori.al { ... }` block
(the one with `reverse_proxy`) with `import /etc/caddy/factorial.caddy`, then:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

If the Caddyfile's global options turn the admin endpoint off (`admin off`), a reload cannot
reach Caddy: use `sudo systemctl restart caddy`, which drops the server's other sites for a
second too. `setup.sh` reads the Caddyfile and ends with the command of the two that works
there; run again later, it names a command only when the site's block changed. A deploy never
needs either. factori.al answers 404 until the first deploy.

**5. GitHub**, in the settings of kgdunn/process-improve:

- **Environments → New environment** `factorial`. Under deployment branches, allow `main`
  only. Add the environment secret `FACTORIAL_SSH_KEY`: the whole of
  `~/factorial-deploy/factorial_deploy`, the private half, from `-----BEGIN` to `-----END`
  and its line.
- **Secrets and variables → Actions → Variables → New repository variable**
  `FACTORIAL_KNOWN_HOSTS`: the line `setup.sh` printed, `factori.al ssh-ed25519 AAAA...`. It
  pins the server's host key, so the workflow refuses any other machine answering as
  factori.al. It has to be a repository variable: the deploy jobs read it to decide whether
  to run, before they enter the environment.

With the GitHub CLI, the secret goes in without copy and paste (the environment must exist):

```bash
gh secret set FACTORIAL_SSH_KEY --repo kgdunn/process-improve --env factorial < ~/factorial-deploy/factorial_deploy
```

Then delete the private key, `rm ~/factorial-deploy/factorial_deploy`: GitHub holds the copy
the workflow needs, and a new key is a rerun of steps 1 and 2. The deploy jobs are skipped
until `FACTORIAL_KNOWN_HOSTS` is set. The next push to `main` deploys; to deploy now, run
**Actions → Docs → Run workflow** on `main`.

The runners connect from GitHub's addresses, so SSH (port 22) has to stay open to the
internet, as it is now.

## Feedback

The app's Feedback dialog sends a reader's report to `https://factori.al/api/feedback`, and
Caddy passes it to [`feedback_relay.py`](feedback_relay.py), which emails it to the maintainer
through the email provider that sends the site's mail. Each report gets a reference, such as
`FB-7K3Q2M`, in the email's subject and on the reader's screen. When the reader gives an address,
the relay also sends it a receipt: fixed text with the reference, the topic and the day, and
nothing the reader wrote, at most one a day to an address. The relay listens on `127.0.0.1:8790`
only, as an account systemd makes for it at each start
([`factorial-feedback.service`](factorial-feedback.service)). It keeps nothing and logs no part
of a report, and it takes at most 10 reports an hour from one network, and sends at most 200
emails a day in all, reports and receipts together.

**Setting it up**, once, on the server, before the old app is retired: its mail settings are the
ones the relay needs, under the same names, in `/home/deploy/factorial/.env`. `setup.sh` made
the relay's two files, empty, where only root can read them.

```bash
sudo grep -E '^SMTP_(HOST|PORT|USER|FROM_EMAIL)=' /home/deploy/factorial/.env    # the old app's settings
sudo nano /etc/factorial/feedback.env    # copy those in, and set FEEDBACK_TO, the address reports go to
sudo sh -c "grep '^SMTP_PASSWORD=' /home/deploy/factorial/.env | cut -d= -f2- | tr -d '\n' > /etc/factorial/smtp-password"
sudo systemctl enable --now factorial-feedback
```

The relay reads `smtp-password` as it is, less a final line break: if the old `.env` quotes the
password, take the quotes off there. Then check that Caddy reaches the relay:

```bash
curl -s -o /dev/null -w '%{http_code}\n' -X POST -H 'Origin: https://factori.al' \
  -H 'Content-Type: application/json' -d '{}' https://factori.al/api/feedback
```

It prints `400`, a report the relay refused; `502` means the relay is not running, and
`sudo journalctl -u factorial-feedback` says why. Last, send a report from https://factori.al
with the Feedback button, and check that it reaches `FEEDBACK_TO`.

**A change to the relay** reaches the server only when `setup.sh` runs again, since the deploy
key cannot change it: from your machine, a fresh copy of this folder with the deploy key's
public half (step 1 without `ssh-keygen`), and then step 2, which restarts the relay.

```bash
cd ~/factorial-deploy && git -C ~/process-improve fetch origin
rm -rf factorial && git -C ~/process-improve archive origin/main deploy/factorial | tar -x --strip-components=1
cp factorial_deploy.pub factorial/
ssh deploy@factori.al rm -rf factorial-setup && scp -r factorial deploy@factori.al:factorial-setup
```

**On the server itself**, with the checkout at `~/process-improve`, nothing is copied over the
network, and the deploy key is read back from where `setup.sh` put it, so the key GitHub Actions
deploys with stays the one installed:

```bash
mkdir -p ~/factorial-deploy && cd ~/factorial-deploy && git -C ~/process-improve fetch origin
rm -rf factorial && git -C ~/process-improve archive origin/main deploy/factorial | tar -x --strip-components=1
sed 's/^restrict //' /etc/ssh/authorized_keys/factorial > factorial/factorial_deploy.pub
sudo sh factorial/setup.sh factorial/factorial_deploy.pub
```

Then send a report with the Feedback button: its email is written by the relay just installed.

## Rooms

The app's "Work on it together" shares a study in a room: a study that two or more browsers keep
in step (`web/room.js`). Each browser encrypts the study with a key that only the room's link holds,
after its `#`, and sends it to `https://factori.al/api/room`. Caddy passes the request to
[`encrypted_study_locker.py`](encrypted_study_locker.py), the encrypted study locker, which keeps one file of
encrypted bytes for each room in `/var/lib/factorial-study-locker`, and hands it back. The locker never holds
the key, so neither it, nor
anyone who reads its folder, can read a study. It listens on `127.0.0.1:8791`, runs as an account
systemd makes for it at each start, which can write only to that folder
([`factorial-study-locker.service`](factorial-study-locker.service)), and needs no settings.

What it takes, and for how long it keeps it:

- An encrypted study of 4, 16 or 64 KiB once padded, and nothing else: an image or a file does not fit.
- One file for each room, which each change replaces: no history, and no list of rooms.
- A room nobody changes for 30 days is removed.
- At most 256 MiB in all: past that, a new room is refused until old ones expire.
- From one network (a /24 of IPv4, a /48 of IPv6): 20 new rooms a day, 600 changes and 3600
  reads an hour.

A room's name, or the token that changes it, travels in the `Authorization` header, which the
access log leaves out, with the versions a browser names (`If-Match`, `If-None-Match`). The locker
logs the status of each request, and nothing that names a room.

To remove one room, from its link (whoever reports a room gives the link, and with it the key):

```bash
sudo env STATE_DIRECTORY=/var/lib/factorial-study-locker factorial-study-locker forget 'https://factori.al/#room=...'
```

To stop every room at once, `sudo systemctl stop factorial-study-locker`: the app then says the room
cannot be reached, and each browser keeps its own copy of every study it was sharing.

Check that Caddy reaches the locker:

```bash
curl -s -H 'Authorization: Room AAAAAAAAAAAAAAAAAAAAAA' https://factori.al/api/room
```

It prints the locker's refusal, `{"ok": false, "error": "there is no such room: ..."}`; an empty
answer with status `502` means the locker is not running, and `sudo journalctl -u factorial-study-locker`
says why.

## Retiring the old app

The old app ran from `/home/deploy/factorial` under Docker Compose, with nightly backups to
Hetzner Object Storage. Once factori.al serves Factorial:

```bash
cd /home/deploy/factorial && docker compose down    # add -v to delete the database volume too
sudo rm /etc/cron.d/doe-backup /etc/default/doe-backup /etc/logrotate.d/doe-backup
sudo rm /etc/sudoers.d/doe-caddy-reload /etc/caddy/active_backend.caddy
sudo rm -r /var/lib/factorial /home/deploy/.aws
```

`/etc/default/doe-backup` and `~/.aws` hold the bucket's credentials: revoke that access key
in the Hetzner console too. The bucket's Object Lock keeps the backups already taken until
their retention ends; the bucket can be deleted after that.

## On the server

```bash
readlink /srv/factorial/current          # the live build
cat /srv/factorial/history               # every switch, and every rollback
sudo -u factorial env SSH_ORIGINAL_COMMAND=rollback factorial-deploy /srv/factorial
```

The last line rolls back without GitHub, as the workflow would.

## The files

| File | Installed as | What it is |
|---|---|---|
| [`factorial_deploy.py`](factorial_deploy.py) | `/usr/local/bin/factorial-deploy` | The deploy key's one command |
| [`usage.py`](usage.py) | `/usr/local/bin/factorial-usage` | The usage report |
| [`usage-daily.sh`](usage-daily.sh) | `/etc/cron.daily/factorial-usage` | Keeps each finished day's counts |
| [`usage-hourly.sh`](usage-hourly.sh) | `/etc/cron.hourly/factorial-usage-public` | Writes the public `usage.json` the footer draws |
| [`sshd.conf`](sshd.conf) | `/etc/ssh/sshd_config.d/factorial.conf` | What sshd allows the deploy account |
| [`factorial.caddy`](factorial.caddy) | `/etc/caddy/factorial.caddy` | The site, its headers and its log |
| [`feedback_relay.py`](feedback_relay.py) | `/usr/local/bin/factorial-feedback` | The relay that emails a reader's report |
| [`factorial-feedback.service`](factorial-feedback.service) | `/etc/systemd/system/factorial-feedback.service` | Runs the relay |
| [`feedback.env.example`](feedback.env.example) | `/etc/factorial/feedback.env`, if missing | The relay's settings |
| [`encrypted_study_locker.py`](encrypted_study_locker.py) | `/usr/local/bin/factorial-study-locker` | The locker that keeps each room's encrypted study |
| [`factorial-study-locker.service`](factorial-study-locker.service) | `/etc/systemd/system/factorial-study-locker.service` | Runs the locker |
| [`setup.sh`](setup.sh) | (run once) | Installs each file above |

[`tests/deploy`](../../tests/deploy) tests the two commands, the feedback relay and the locker, including
what a leaked key, or anyone who finds their address, could send.
