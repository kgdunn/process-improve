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
[ -d /var/log/caddy ] || install -d -o caddy -g caddy -m 750 /var/log/caddy  # other sites may log there too
touch /var/log/caddy/factorial-access.log
chown caddy:caddy /var/log/caddy/factorial-access.log
chmod 640 /var/log/caddy/factorial-access.log

# The feedback relay and its unit. Its settings and the SMTP password are made once, empty, where
# only root can read them, and kept after that: running this again never touches what was filled in.
install -o root -g root -m 755 "$here/feedback_relay.py" /usr/local/bin/factorial-feedback
install -o root -g root -m 644 "$here/factorial-feedback.service" /etc/systemd/system/factorial-feedback.service
install -d -o root -g root -m 700 /etc/factorial
[ -e /etc/factorial/feedback.env ] || install -o root -g root -m 600 "$here/feedback.env.example" /etc/factorial/feedback.env
[ -e /etc/factorial/smtp-password ] || install -o root -g root -m 600 /dev/null /etc/factorial/smtp-password
systemctl daemon-reload
# A relay already set up is restarted, so it runs the copy just installed; one not yet set up waits for its settings.
if grep -q '^FEEDBACK_TO=.' /etc/factorial/feedback.env; then
    systemctl enable --quiet factorial-feedback
    systemctl restart factorial-feedback
    feedback="The feedback relay runs the copy just installed."
else
    feedback="The feedback relay waits for its settings: fill in /etc/factorial/feedback.env and
/etc/factorial/smtp-password (README.md, \"Feedback\"), then: sudo systemctl enable --now factorial-feedback"
fi

echo
echo "$feedback"
echo
echo "Done. Next, in /etc/caddy/Caddyfile, replace the whole factori.al { ... } block with:"
echo "    import /etc/caddy/factorial.caddy"
echo "then:"
echo "    sudo caddy validate --config /etc/caddy/Caddyfile && sudo systemctl reload caddy"
echo "(restart instead of reload if the Caddyfile turns the admin endpoint off: admin off)"
echo
echo "The value of the GitHub repository variable FACTORIAL_KNOWN_HOSTS:"
printf 'factori.al %s\n' "$(cut -d' ' -f1,2 /etc/ssh/ssh_host_ed25519_key.pub)"
