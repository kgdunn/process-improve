- Factorial says which version it is at the foot of the page, as "Version 88ccccb, 13 November 2026.": the first
  seven characters of the commit it was built from, as a link to that commit on GitHub, and the day of the commit.
  The build's manifest gains `committed`, the commit's day, beside `commit`. A page that was not built, such as a
  plain preview, shows no version.
- A report from Feedback names the same version, the whole commit and then the day, on the last line of its email:
  "Version: 88ccccbbe212e09c53d9f69b3ee7193262cc165a, 13 November 2026". The line was "App commit", first under
  "What the page sent with it", and held the commit alone. The dialog's list of what will be sent, and the report
  as text for an issue or an email, end with the same line. The email is written by the relay, so its last line
  is there once `deploy/factorial/setup.sh` has run on the server again; until then the email keeps its
  "App commit" line, which holds the commit and the day.
