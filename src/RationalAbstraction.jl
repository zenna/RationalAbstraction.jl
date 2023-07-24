module RationalAbstraction

using Flux
using Random

include("util.jl")

ŝ(prob, α, s, u) = s(α(prob))

"""
Train a rational abstraction model.

# Arguments
- `rng`: A random number generator
- `s`: A function that maps a problem to a solution
- `ŝ`: A function that abstracts a problem and solves it, producing a solution
- `u`: A utility function
- `problem_rv`: A random variable that generates problems
- `batch_size`: The number of problems to sample from `problem_rv` at each iteration
"""
function train!(rng, s, ŝ, u, problem_rv, embed; batch_size, nepochs = 5)
  # Sample a batch of problems

  opt_states = Dict(query_type => Flux.setup(Momentum(0.1), model) for query_type in keys(ŝ))

  querty_ps = p -> p.query

  ## Oone optimiser or one per query type?
  ## One optimiser for now

  for i = 1:nepochs
    @info "Epoch $i"
    ps = rand(rng, problem_rv, batch_size)
    split_ps = splitby(ps, querty_ps)

    for query_type in keys(split_ps)
      @info "Handling $query_type"
      opt = opt_states[query_type]

      ps = split_ps[query_type]
      ps_embed = embed.(ps) # This will produce a vector of vectors but we want a matrix
      sub_batch_size = length(ps_embed)
      data = reshape(vcat(ps_embed...), (:, sub_batch_size))
      net = ŝ[query_type]
      @show r̂ = net(data)
      @show r = s.(ps)

      ## 
      Optimisers.update!(opt, net, gs[1])
      @show u(r̂, r, query_type)
    end
  end
end

function train!(loss, model, data, opt; cb = nothing)
  isnothing(cb) || error("""train! does not support callback functions.
                            For more control use a loop with `gradient` and `update!`.""")
  @withprogress for (i,d) in enumerate(data)
    d_splat = d isa Tuple ? d : (d,)
    l, gs = Zygote.withgradient(m -> loss(m, d_splat...), model)
    if !isfinite(l)
      throw(DomainError("Loss is $l on data item $i, stopping training"))
    end
    opt, model = Optimisers.update!(opt, model, gs[1])
    @logprogress Base.haslength(data) ? i/length(data) : nothing
  end
end

include("ball.jl")

end # module RationalAbstraction
