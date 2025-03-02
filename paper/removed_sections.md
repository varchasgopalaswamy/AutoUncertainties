# *Removed Sections (Not Currently in Use)*

## Pint Integration

`AutoUncertainties` is compatible with the `Pint` package and its implementation of `Quantity` objects. Support
is achieved with the components described in the following subsections. 

### `UncertaintyQuantity`

The `UncertaintyQuantity` class is an extension of `PlainQuantity` from `Pint`, and is designed to wrap `Uncertainty` 
objects into `Quantity` objects. Wrapping `Uncertainty` objects with `UncertaintyQuantity` (rather than with the 
standard `pint.Quantity` class) ensures better feature continuity between `Pint` and `AutoUncertainties`. 

```python
from auto_uncertainties.pint import UncertaintyQuantity
from auto_uncertainties import Uncertainty
u = Uncertainty(1.0, 0.5)
q = UncertaintyQuantity(u, 'radian')
```

### `UncertaintyRegistry`

The `UncertaintyRegistry` class is a custom `Pint` `UnitRegistry` in which the default `Quantity` class is mapped to
`UncertaintyQuantity`.

### `from_quantities`

The `Uncertainty` class contains the `from_quantities` method, which returns an `UncertaintyQuantity` object 
created from two input `Quantity` objects, one representing the central value(s), and one representing the error(s). 