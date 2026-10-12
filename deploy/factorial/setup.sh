#!/bin/sh
# Prepare factori.al's server for deploys from GitHub Actions. Run it once, as root, from a copy
# of this folder, with the deploy key's public half (README.md, "Setting it up, once"):
#
#     sudo sh factorial-setup/setup.sh factorial-setup/factorial_deploy.pub
#
# Running it again is safe: it creates what is missing and replaces the files it installs.
# It changes nothing in /etc/caddy/Caddyfile; README.md says what to change there.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
key=${1:?usage: sudo sh setup.sh <the deploy key, .pub>}
grep -q '^ssh-ed25519 ' "$key" || { echo "$key is not an ed25519 public key" >&2; exit 1; }
python3 -c 'import tarfile; tarfile.data_filter' 2>/dev/null \
    || { echo "python3 has no tarfile data filter: the deploy command needs Python 3.12 or later" >&2; exit 1; }

# The account the deploy key logs in as. It has no password; sshd.conf gives it one command.
id factorial >/dev/null 2>&1 || useradd --system --create-home --shell /bin/sh factorial

# The site: a folder per build, and the link `current` that Caddy serves, which the deploy switches.
install -d -o factorial -g factorial -m 755 /srv/factorial /srv/factorial/releases

# The deploy key's one command, and the usage report.
install -o root -g root -m 755 "$here/factorial_deploy.py" /usr/local/bin/factorial-deploy
install -o root -g root -m 755 "$here/usage.py" /usr/local/bin/factorial-usage

# The deploy key, where only root can change it, and restricted again on its own line.
install -d -o root -g root -m 755 /etc/ssh/authorized_keys
printf 'restrict %s\n' "$(cat "$key")" > /etc/ssh/authorized_keys/factorial
chmod 644 /etc/ssh/authorized_keys/factorial

# sshd reads the new block only once the whole configuration checks out, so a mistake here
# cannot lock anyone out.
install -o root -g root -m 644 "$here/sshd.conf" /etc/ssh/sshd_config.d/factorial.conf
sshd -t
systemctl try-reload-or-restart ssh

# Caddy: the site's block, and its access log. Caddy writes the log as the caddy account, so it
# is made here, owned by caddy: `sudo caddy validate` would otherwise make it root's, and Caddy
# could then not open it.
install -o root -g root -m 644 "$here/factorial.caddy" /etc/caddy/factorial.caddy
install -d -o caddy -g caddy -m 750 /var/log/caddy
touch /var/log/caddy/factorial-access.log
chown caddy:caddy /var/log/caddy/factorial-access.log
chmod 640 /var/log/caddy/factorial-access.log

echo
echo "Done. Next, in /etc/caddy/Caddyfile, replace the whole factori.al { ... } block with:"
echo "    import /etc/caddy/factorial.caddy"
echo "then:"
echo "    sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy"
echo
echo "The value of the GitHub repository variable FACTORIAL_KNOWN_HOSTS:"
printf 'factori.al %s\n' "$(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)"
