# The reference catalogue: what it holds, and on what terms

The code of process-improve is under the MIT licence. The JSON files in this folder are not
code. Each one holds what a published source reports, so that software can be checked
against it:

- the facts it reports: numbers, design matrices and experimental data, as printed;
- a short quotation beside each number that shows where it is printed.

**The MIT licence does not extend to these files' contents.** Each file states the terms of
its source in `source.rights`, and `reference_sources()` returns them:

- A journal article, a book under copyright (such as Davies, *The Design and Analysis of
  Industrial Experiments*) or a slide deck is protected by copyright and is not reproduced here.
  Its facts are held for citation and for checking software, and its quotations are short and
  attributed. Read the source itself through its DOI or URL.
- The book *Process Improvement using Data* is under the Creative Commons
  Attribution-ShareAlike 4.0 licence, which its quotations and data keep.
- The NIST/SEMATECH e-Handbook is a work of the US government.

No source document, figure or image is included.

One file is the exception: `topics.json` is the catalogue's own vocabulary of subject-matter
topics, and part of this package under its MIT licence. The topics given to each entry are the
catalogue's classification, not something its source prints.

If you hold rights in a source and want something here changed or removed, open an issue at
<https://github.com/kgdunn/process-improve/issues>.
