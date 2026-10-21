- **Factorial's studies can leave the browser as a file, and come back.** Under "Your studies", "Download
  all" saves every study as one JSON file, and each study's "Share" hands its own file to the share sheet
  (Mail, Messages, AirDrop, Files) where the browser can share a file, or saves it where it cannot (the button
  then reads "Download"). A dropzone under the list opens such a file in any browser: a file of one study
  opens it on the page, a file of several adds them all; a study already kept there is not saved twice, and
  none is overwritten. The file (`format`, `version`, and each study with its `id`, times, `sha256` and the
  study whole) is built to grow: a reader skips keys it does not know, and a study's own format number rises
  when it holds something an earlier Factorial could not keep. "How Factorial works" describes it.
