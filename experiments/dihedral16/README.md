# Dihedral-16 Composition Control

This control uses the 16 symmetries of a regular octagon. The target is the ordered product of a sequence of group elements. Context sequences have lengths 2--3 and evaluation sequences lengths 4--10.

The coarse row contains length plus counts over eight merged categories. Category index `4 * reflected + (rotation mod 4)` merges elements separated by four rotations and discards sequence order. The absolute row contains length plus ten 16-column positional blocks. The comparison therefore removes both order and part of element identity; it is not an order-only ablation.

