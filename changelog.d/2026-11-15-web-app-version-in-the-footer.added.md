- Factorial says which version it is at the foot of the page, as "Version 88ccccb, 13 November 2026.": the first
  seven characters of the commit it was built from, as a link to that commit on GitHub, and the day of the commit.
  The build's manifest gains `committed`, the commit's day, beside `commit`. A report from Feedback names the same
  version: its "App commit" line, which held the commit, holds the whole commit and then the day, as in
  "88ccccbbe212e09c53d9f69b3ee7193262cc165a, 13 November 2026". The relay takes it as it is, so the server needs
  no change. A page that was not built, such as a plain preview, shows no version.
