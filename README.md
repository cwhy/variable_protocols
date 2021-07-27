# Variable Protocols
A way to specify common variables of data inputs in the field of data science and machine learning.

## Installation
`pip install variale_protocols`

## Specifications:
### Principles
* Only one representation for each variable type
* See `variable_protocols/variable_protocols.py`
### Guideline
You have a var, 
```
the var is a BaseVariable <─────────────────────────────────────────────────────┐
               │      │                                                         │
           (no)│      └─── (yes)──> `BaseVariable`                              │
               v                                                                │
 the var consist of a group                                                     │
  of same `Variable`s ────(no)─> `VariableGroup`, for all the variable types ───┤
              (yes)│                                                            │ 
                   v                                                            │
the var has a fixed length ─(no)──> `VariableList`, for all the variable types ─┤
              (yes)│                                                            │
                   v                                                            │
   the group has N >= 2 dimensions ─(yes)────────> Nd `VariableTensor`          │
               (no)│                                                            │
                   v                                                            │
   the group is shuffle-able ─(no)───────────────>  1d `VariableTensor`         │
              (yes)│                                                            │
                   v                                                            │
             `VariableGroup` of length 1 [1], for all the variable types ───────┘

```
You got a variable.
* [1] To prevent `NamedVariable`s of the same types collapsing inside a `VariableGroups`,
 which would be the same as an 1D `VariableTensor` we forbid 1D `VariableTensor`s and
 use `VariableGroup` instead.

## Examples
see `test/examples.py`

## Goals
* specify common variables of data inputs in the field of data science and machine learning.
* structural typing, only sum types and product types (In the general sense, not ML family ADT)
* there should be one way to type one variable group
* serialization in the future (probably in Dhall)

## NonGoals
* compatability with any current stuff
* easy to comprehend (simple is not easy)

## Todo
* For now the set nature of VariableGroup does not allow
 shuffle-able variables of the same tensor type