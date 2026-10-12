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

```bash
sudo factorial-usage            # visits, readers, Python sessions and link previews per day
sudo factorial-usage --days 90
```

Robots, link-preview fetchers and the deploy's own checks are not counted as visits.

## Setting it up, once

**1. A deploy key**, on your machine, in a checkout of this repository:

```bash
ssh-keygen -t ed25519 -N '' -C 'github-actions@kgdunn/process-improve' -f factorial_deploy
scp -r deploy/factorial factorial_deploy.pub deploy@factori.al:
```

**2. The server**, logged in as `deploy`:

```bash
sudo sh factorial/setup.sh factorial_deploy.pub
```

[`setup.sh`](setup.sh) makes the `factorial` account and `/srv/factorial`, installs
`factorial-deploy` and `factorial-usage` in `/usr/local/bin`, puts the key where only root can
change it, and installs the SSH block ([`sshd.conf`](sshd.conf)) once `sshd -t` accepts it.
It also installs the site's Caddy block, [`factorial.caddy`](factorial.caddy). It needs the
server's Python to be 3.12 or later, as Ubuntu 24.04's is. Running it again is safe.

**3. Caddy.** In `/etc/caddy/Caddyfile`, replace the old app's `factori.al { ... }` block
(the one with `reverse_proxy`) with `import factorial.caddy`, then:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy
```

factori.al answers 404 until the first deploy.

**4. GitHub**, in the settings of kgdunn/process-improve:

- **Environments → New environment** `factorial`. Under deployment branches, allow `main`
  only. Add the secret `FACTORIAL_SSH_KEY`: the whole of `factorial_deploy`, the private half.
- **Secrets and variables → Actions → Variables → New repository variable**
  `FACTORIAL_KNOWN_HOSTS`: the server's host key, so the workflow refuses any other machine
  answering as factori.al. Print it on the server, where it cannot have been tampered with:

  ```bash
  printf 'factori.al %s\n' "$(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)"
  ```

Then delete `factorial_deploy` from your machine: GitHub holds the copy the workflow needs.
The deploy jobs are skipped until `FACTORIAL_KNOWN_HOSTS` is set. The next push to `main`
deploys; to deploy now, run **Actions → Docs → Run workflow** on `main`.

The runners connect from GitHub's addresses, so SSH (port 22) has to stay open to the
internet, as it is now.

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
| [`setup.sh`](setup.sh) | (run once) | Installs the four above |

[`tests/deploy`](../../tests/deploy) tests the two commands, including what a leaked key could
send.
