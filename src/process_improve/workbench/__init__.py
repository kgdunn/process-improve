"""The dataset workbench: refine a table step by step, and keep the Python that repeats it.

A :class:`Recipe` is a source file and a list of steps (drop, keep or rename columns,
drop or filter rows, add a computed column, transform a column, read text as numbers,
identify the rows). It applies its steps, writes a script that repeats them with pandas
alone, and round-trips through JSON.

A :class:`Variant` is a model (PCA or PLS) over a recipe's table. It leaves its
excluded rows out of the fit and projects them through it, and its :class:`Fit`
explains why a group of rows differs from the rest (:meth:`Fit.explain_group`).

A :class:`Project` keeps the source file, its recipe, the variants and the reader's
saved selections in one zip file, and refits and checks every variant on opening.
"""

from process_improve.workbench.project import Opened, Project
from process_improve.workbench.recipe import (
    STEPS,
    TRANSFORMS,
    AddColumn,
    DropColumns,
    DropRows,
    FilterRows,
    KeepColumns,
    Recipe,
    RenameColumns,
    SetIndex,
    SourceFile,
    Step,
    ToNumeric,
    Transform,
    step_from_dict,
)
from process_improve.workbench.variant import Fit, GroupExplanation, Variant

__all__ = [
    "STEPS",
    "TRANSFORMS",
    "AddColumn",
    "DropColumns",
    "DropRows",
    "FilterRows",
    "Fit",
    "GroupExplanation",
    "KeepColumns",
    "Opened",
    "Project",
    "Recipe",
    "RenameColumns",
    "SetIndex",
    "SourceFile",
    "Step",
    "ToNumeric",
    "Transform",
    "Variant",
    "step_from_dict",
]
