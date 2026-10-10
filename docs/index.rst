.. rst-class:: pi-landing

process-improve
===============

.. container:: pi-tagline

   Designed experiments, multivariate models and process monitoring in Python: the companion
   package to the free textbook `Process Improvement using Data <https://learnche.org/pid>`_.

.. image:: /_static/readme/banner-light.png
   :class: only-light pi-banner
   :alt: Four designs from one generate_design call: a definitive screening design, a central composite design, a constrained I-optimal design and a MaxPro space-filling design.

.. image:: /_static/readme/banner-dark.png
   :class: only-dark pi-banner
   :alt: Four designs from one generate_design call: a definitive screening design, a central composite design, a constrained I-optimal design and a MaxPro space-filling design.

.. container:: pi-install

   .. code-block:: bash

      pip install process-improve

.. container:: pi-buttons

   :doc:`Get started <quickstart>`
   `Try it in your browser <app/>`__
   :doc:`User guide <user_guide/index>`
   :doc:`API reference <api/index>`
   `The textbook <https://learnche.org/pid>`__


What it does
------------

.. container:: pi-cards

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-flask pi-icon" aria-hidden="true"></i>

      .. rubric:: Designed experiments

      Factorial, response surface, definitive screening, OMARS, optimal and mixture designs
      from one ``generate_design`` call, and the analysis of the results.

      :doc:`Plan an experimental programme <user_guide/doe_strategy>`

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-scale-balanced pi-icon" aria-hidden="true"></i>

      .. rubric:: Judge a design before you run it

      Efficiency, prediction variance, correlation between coefficients and bias from omitted
      terms, all computed for the model you intend to fit.

      :doc:`Evaluate a design <user_guide/design_evaluation>`

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-circle-nodes pi-icon" aria-hidden="true"></i>

      .. rubric:: Multivariate models

      PCA, PLS and TPLS in scikit-learn style, with scores, loadings, contributions, and
      Hotelling's T² and SPE limits for spotting unusual observations.

      :doc:`Latent variable models <user_guide/multivariate>`

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-chart-line pi-icon" aria-hidden="true"></i>

      .. rubric:: Process monitoring

      Shewhart, EWMA and CUSUM control charts, and adaptive PCA and PLS models that update
      as new samples arrive.

      :doc:`Monitoring reference <api/monitoring>`

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-industry pi-icon" aria-hidden="true"></i>

      .. rubric:: Batch processes

      A batch bioreactor simulator, batch trajectory features, and mid-course correction of
      a batch that is still running.

      :doc:`The batch simulator <user_guide/batch_simulator>`

   .. container:: pi-card

      .. raw:: html

         <i class="fa-solid fa-users pi-icon" aria-hidden="true"></i>

      .. rubric:: Sensory panels

      Validate descriptive panel data, check each assessor, and relate the attribute scores
      back to the products.

      :doc:`Descriptive panel data <user_guide/sensory_panel>`


Try it without installing anything
----------------------------------

.. container:: pi-feature

   .. image:: /_static/readme/app-demo.gif
      :alt: The designed-experiments app in a browser: choosing factors, generating the design, and reading the analysis of the results.

   .. container::

      The `designed-experiments app <app/>`__ runs process-improve inside your browser.
      Design an experiment, download the run sheet, fill in your results, and upload it for
      the analysis, with nothing to install.

      `Open the app <app/>`__


Sixteen designs, one function call
----------------------------------

.. container:: pi-feature

   .. image:: /_static/readme/design-gallery-light.png
      :class: only-light
      :alt: Sixteen design families, each drawn as its runs: supersaturated, Plackett-Burman, fractional factorial, definitive screening, full factorial, central composite, Box-Behnken, OMARS, Taguchi, D-optimal, constrained I-optimal, constrained mixture, Latin hypercube, uniform, MaxPro and Sobol designs.

   .. image:: /_static/readme/design-gallery-dark.png
      :class: only-dark
      :alt: Sixteen design families, each drawn as its runs: supersaturated, Plackett-Burman, fractional factorial, definitive screening, full factorial, central composite, Box-Behnken, OMARS, Taguchi, D-optimal, constrained I-optimal, constrained mixture, Latin hypercube, uniform, MaxPro and Sobol designs.

   .. container::

      Screening, response surface, optimal, mixture and space-filling designs all come from
      ``generate_design(factors, design_type=...)``. Each construction is checked against its
      published definition by the test suite, so a change that breaks a design fails the
      tests.

      :doc:`See every design family <user_guide/doe_coverage>`


Learn the methods
-----------------

.. container:: pi-feature

   .. raw:: html

      <a href="https://learnche.org/pid"><img src="https://learnche.org/pid/_static/hero.png"
         alt="Process Improvement using Data, a free online textbook by Kevin G. Dunn, beside one of its Python examples with the Run in browser button and the plot a click draws."></a>

   .. container::

      The free textbook `Process Improvement using Data <https://learnche.org/pid>`_, by the
      same author, explains the statistics behind the package: designed experiments, process
      monitoring, regression and latent variable models. Its Python examples use
      process-improve, and over 180 of them run in your browser with a click.

      `Read the textbook <https://learnche.org/pid>`__


.. toctree::
   :hidden:
   :caption: Contents

   quickstart
   architecture
   scaling
   user_guide/index
   api/index

.. toctree::
   :hidden:
   :caption: Applied DoE

   applied_doe/index

.. toctree::
   :hidden:
   :caption: Case studies

   user_guide/case_studies/index

.. toctree::
   :hidden:
   :caption: Development

   development/index
