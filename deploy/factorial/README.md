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

## What the server keeps

Usage, and nothing else. Caddy writes one line per request to
`/var/log/caddy/factorial-access.log`: the time, the path, the status, the browser's
`User-Agent`, and the page that linked here. The reader's address is cut to its network
before the line is written (the first 24 bits of IPv4, 48 of IPv6), and lines are kept for
90 days. There are no cookies, no tracking script and no database. A design shared by link
travels after the `#`, which a browser never sends, so no design reaches the log.

The one exception is a report a reader sends with the app's Feedback button, and only when they
press Send: the relay emails it to the maintainer, and keeps no copy and no line of it
([Feedback](#feedback)). Its request is logged as any other, without its body.

```bash
sudo factorial-usage            # visits, catalog visits, readers, Python sessions and link previews per day
sudo factorial-usage --days 90
```

Robots, link-preview fetchers and the deploy's own checks are not counted as visits. A visit to
the design catalog (`/catalog/`) is counted in a column of its own, and its readers among the app's.
The report is the `factorial-usage` that `setup.sh` installed: run `setup.sh` again to install a
newer one.

## Setting it up, once

**1. The files and a deploy key**, on your machine, in a folder outside the repository so the
key cannot be committed (`~/code/process-improve` stands for your checkout):

```bash
mkdir -p ~/factorial-deploy && cd ~/factorial-deploy
git -C ~/code/process-improve fetch origin
git -C ~/code/process-improve archive origin/main deploy/factorial | tar -x --strip-components=1
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
once ([Feedback](#feedback)). It ends by printing the value of `FACTORIAL_KNOWN_HOSTS` for step 5.
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
second too. A deploy never needs either. factori.al answers 404 until the first deploy.

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
through the email provider that sends the site's mail. The relay listens on `127.0.0.1:8790`
only, as an account systemd makes for it at each start
([`factorial-feedback.service`](factorial-feedback.service)). It keeps nothing and logs no part
of a report, and it takes at most 10 reports an hour from one network and 200 a day in all.

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
cd ~/factorial-deploy && git -C ~/code/process-improve fetch origin
rm -r factorial && git -C ~/code/process-improve archive origin/main deploy/factorial | tar -x --strip-components=1
cp factorial_deploy.pub factorial/
ssh deploy@factori.al rm -r factorial-setup && scp -r factorial deploy@factori.al:factorial-setup
```

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
| [`sshd.conf`](sshd.conf) | `/etc/ssh/sshd_config.d/factorial.conf` | What sshd allows the deploy account |
| [`factorial.caddy`](factorial.caddy) | `/etc/caddy/factorial.caddy` | The site, its headers and its log |
| [`feedback_relay.py`](feedback_relay.py) | `/usr/local/bin/factorial-feedback` | The relay that emails a reader's report |
| [`factorial-feedback.service`](factorial-feedback.service) | `/etc/systemd/system/factorial-feedback.service` | Runs the relay |
| [`feedback.env.example`](feedback.env.example) | `/etc/factorial/feedback.env`, if missing | The relay's settings |
| [`setup.sh`](setup.sh) | (run once) | Installs the seven above |

[`tests/deploy`](../../tests/deploy) tests the two commands and the relay, including what a
leaked key, or anyone who finds the relay's address, could send.
