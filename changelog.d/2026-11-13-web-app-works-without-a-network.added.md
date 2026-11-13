- **factori.al opens without a network.** Once the page has been opened with one, a copy of it is kept
  in the browser: its own files, Python and its packages, and the wheel. The copy costs no extra
  download, since the visit has just fetched it all. The foot of the page says what the copy holds
  and what it takes of the device's storage (about 170 MB), read from the copy itself each visit, so
  it never claims what a browser has since dropped. A button
  there adds the charts and the workbooks, a download of about 12 MB, after which a study can be
  designed, analysed, drawn and saved in a lab with no signal; another removes the copy, and a copy
  that could not be finished says why. With a network the page's own files always come from the
  server, so a newer page is never held back, and a build made with `--no-service-worker` retires the
  copy in every browser, whatever state the page's own code is in. Stop now starts Python again from
  the copy, in seconds. (#1230)
