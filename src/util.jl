using Spec

"""
Split `xs` into equivalance classes defined by `f``, where two elements
are in the same class if `f(x) == f(y)``.

# Arguments
- `xs`: A list of elements
- `f`: Mapping from elements of `xs` to a value defining equivalence class

# Returns
- a dictionary mapping each equivalence class to a list of elements in that class

Example
```julia
julia> splitby([1, 2, 3, 4, 5, 6], iseven)
Dict{Bool,Array{Int64,1}} with 2 entries:
  false => [1, 3, 5]
  true  => [2, 4, 6]
```
"""
function splitby(xs, f)
  # Should produce as concrete types as inferreable
  f_xs = map(f, xs)
  d = Dict{valtype(f_xs), Vector{eltype(xs)}}()
  for (x, fx) in zip(xs, f_xs)
    if haskey(d, fx)
      push!(d[fx], x)
    else
      d[fx] = [x]
    end
  end
  return d
end
@pre splitby(xs, f) = !isempty(xs) "xs must be non-empty"