"""The dataset workbench: refine a table step by step, and keep the Python that repeats it.

A :class:`Recipe` is a source file and a list of steps (drop, keep or rename columns,
drop or filter rows, add a computed column, transform a column, read text as numbers,
identify the rows). It applies its steps, writes a script that repeats them with pandas
alone, and round-trips through JSON.
"""

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

__all__ = [
    "STEPS",
    "TRANSFORMS",
    "AddColumn",
    "DropColumns",
    "DropRows",
    "FilterRows",
    "KeepColumns",
    "Recipe",
    "RenameColumns",
    "SetIndex",
    "SourceFile",
    "Step",
    "ToNumeric",
    "Transform",
    "step_from_dict",
]
