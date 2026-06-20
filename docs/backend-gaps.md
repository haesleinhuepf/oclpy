# Backend gaps

These issues should be created in `haesleinhuepf/oclpy`. They are kept here
until the GitHub integration has issue-write permission.

## Support Array API bool, float64, int64, and uint64 dtypes in pyclesperanto

Python Array API compliance requires boolean, 64-bit integer, and
double-precision floating dtypes. Current pyclesperanto dtype validation and
host transfer do not support `bool`, `float64`, `int64`, or `uint64`.

Acceptance: pyclesperanto can allocate, upload, download, and operate on these
dtypes; oclpy enables them without a host fallback.

## Remove pyclesperanto's three-dimensional array limit

`pyclesperanto.Array.empty` rejects shapes with more than three dimensions,
while the Array API permits arbitrary-rank arrays. This blocks batched matrix
multiplication and fully general broadcasting.

Acceptance: buffer arrays support arbitrary rank and oclpy's dimension guard
and associated expected-failure test can be removed.

## Add general device-side reshape and axis reductions

The Array API requires reshape and reductions over arbitrary axes with
`keepdims`. Current support is limited to backend-dependent reshape and
whole-array scalar reductions.

Acceptance: reshape avoids host transfers, and `sum`, `min`, and `max` accept
one or multiple axes with standard dtype and `keepdims` behavior.
