- **The footer of every page of factori.al shows how many people used Factorial each day.** Under "Behind it", a
  line of the last 90 days, then a red dot on the last 24 hours, and that count beside it: "89 users, last 24
  hours". It leads to what the server counts, and how. A user is a network and a browser in a day, as the server's
  usage report counts a reader. The server writes the counts, numbers alone, to a public `usage.json` each hour
  (`deploy/factorial/usage.py --public`); a copy of Factorial at another address has none, and leaves the line out.
