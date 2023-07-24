## Model
####################

@enum Color red blue green

struct Ball
  position::Float64
  color::Color
  weight::Float64
end

struct Ramp
  color::Color
end

struct Model
  ball::Ball
  ramp::Ramp
  line::Float64
end

# Assume for simplicity a finite number of queries
abstract type Query end

struct WillBallPassLine <: Query end
struct WhereWillBallEnd <: Query end

query_type(::WillBallPassLine) = Bool
query_type(::WhereWillBallEnd) = Float64 # FIXME: Should this be a 64 bit float?

struct Problem{Q <: Query}
  model::Model
  query::Q
end

"""
Coarse simulation: Will the ball rolling down the ramp
cross the line?

The ball and ramp colors indicate different materials with
different friction coefficients, i.e. red is high friction,
blue is low friction, and green is no friction.
"""
function simulate(model::Model)
  # define friction coefficients for each color
  friction_coefficients = Dict(red => 0.3, blue => 0.6, green => 1.0)
  
  # get the conversion efficiency for the ball and ramp
  ball_efficiency = friction_coefficients[model.ball.color]
  ramp_efficiency = friction_coefficients[model.ramp.color]
  
  # calculate the total conversion efficiency
  total_efficiency = ball_efficiency * ramp_efficiency
  # @show total_efficiency

  # calculate the potential energy of the ball at the top of the ramp
  potential_energy = model.ball.weight * 9.81 * model.ball.position
  # @show potential_energy

  # calculate the kinetic energy the ball will have at the bottom, considering conversion efficiency
  kinetic_energy = potential_energy * total_efficiency
  # @show kinetic_energy
  
  # assuming the kinetic energy will all convert into translational energy, 
  # we can get the final speed using: kinetic_energy = 1/2 * mass * velocity^2
  final_speed = sqrt(2 * kinetic_energy / model.ball.weight)
  # @show final_speed

  # calculate the final distance traveled by the ball using v^2 = u^2 + 2*a*s
  # since initial speed u is zero, the equation simplifies to s = v^2 / (2*a)
  final_distance = final_speed^2 / (2 * 9.81)
  # @show final_distance

  # Return everything we might need for the queries as a named tuple
  return (total_efficiency=total_efficiency,
          potential_energy=potential_energy, 
          kinetic_energy=kinetic_energy,
          final_speed=final_speed,
          final_distance=final_distance, 
          model=model)
end

function simulate(model, query::WillBallPassLine)
  res = simulate(model)
  # check if the ball crosses the line
  if res.final_distance > res.model.line
    return true
  else
    return false
  end
end

function simulate(model, query::WhereWillBallEnd)
  res = simulate(model)
  # return the final distance the ball travels
  return res.final_distance
end

## Randomness and Random Variables
####################

function generate_random_model(rng::AbstractRNG)
    # Randomly generate a position, weight, and angle
    position = rand(rng) * 100
    weight = rand(rng) * 10
    angle = rand(rng) * 90

    # Randomly select a color
    colors = [red, blue, green]
    ball_color = rand(rng, colors)
    ramp_color = rand(rng, colors)

    # Randomly generate a line
    line = rand(rng) * 200

    # Create the Ball, Ramp, and Model instances
    ball = Ball(position, ball_color, weight)
    ramp = Ramp(ramp_color)
    model = Model(ball, ramp, line)

    return model
end

function generate_random_query(rng::AbstractRNG)
    # Randomly select a query
    queries = [WillBallPassLine(), WhereWillBallEnd()]
    query = rand(rng, queries)
    return query
end

generate_random_problem(rng::AbstractRNG) = Problem(generate_random_model(rng), generate_random_query(rng))

struct ProblemRv end

## Create some neural networks
function Base.rand(rng::AbstractRNG, ::ProblemRv, n::Integer)
  [generate_random_problem(rng) for i in 1:n] 
end

## Embedding
####################

"Embed as a `T` valued vector"
struct EmbedAsVector{T} end

"Embed a problem into a vector of features"
embed(problem::Problem, mode) = combine((embed(problem.model, mode), embed(problem.query, mode)), mode)

"Embed a model into a vector of features"
embed(model::Model, mode) = combine((embed(model.ball, mode), embed(model.ramp, mode), embed(model.line, mode)), mode)

"Embed a ball into a vector of features"
embed(ball::Ball, mode) = combine((embed(ball.position, mode), embed(ball.color, mode), embed(ball.weight, mode)), mode)
"Embed colors as one-hot vectors"
embed(color::Color, mode::EmbedAsVector{Float64}) = Float64.(color .== [red, blue, green])

"Embed a ramp into a vector of features"
embed(ramp::Ramp, mode) = combine((embed(ramp.color, mode),), mode)

embed(xs::Float64, mode::EmbedAsVector{Float64}) = [xs]

combine(xs::NTuple{N, Vector{Float64}}, mode::EmbedAsVector{Float64}) where N = vcat(xs...)

"Embed a query into a vector of features"
embed(query::WillBallPassLine, mode::EmbedAsVector{Float64}) = [0.0, 1.0]

"Embed a query into a vector of features"
embed(query::WhereWillBallEnd, mode::EmbedAsVector{Float64}) = [1.0, 0.0]

## Utility
####################

# Utility is a function of the predicted and actual outcomes and depends on the query
function u(predicted::Float64, actual::Float64, query::WhereWillBallEnd)
  return -(predicted - actual)^2
end

function u(predicted::Float64, actual::Bool, query::WillBallPassLine)
  actual = Float64(actual)  # convert actual to Float64 for mathematical operations
  return -actual * log(predicted) - (1 - actual) * log(1 - predicted)
end

function ubatch(predicted::Array, actual::BitVector, query::WillBallPassLine)
  actual_ = reshape(actual, (1, :))
  Flux.binarycrossentropy(predicted,  actual_)
end

ubatch(predicted::Array, actual::Array, query::WhereWillBallEnd) =
  Flux.mse(predicted, reshape(actual, (1, :)))

## FIrst let's create a neural network for the abstraction function α
function train_ball(; batch_size=10, embedding_size=10, abstraction_size=10, output_size=1)
  rng = MersenneTwister(1234)
  embed_(x) = embed(x, EmbedAsVector{Float64}())
  embed_prob_len = length(embed(generate_random_problem(rng), EmbedAsVector{Float64}()))
  abstraction_size = 10
  s_range_len = 10 # THis should be 

  ## Create the neural networks
  α = Chain(Dense(embed_prob_len, 5, relu), Dense(5, abstraction_size), softmax)
  ŝ = Chain(Dense(abstraction_size, 5, relu), Dense(5, s_range_len, relu))

  s_shared_inner_layer_size = 5
  ŝ_shared = Dense(abstraction_size, s_shared_inner_layer_size, relu)
  # Boolean part should 
  ŝ_bool = Chain(Dense(s_shared_inner_layer_size, 1, sigmoid))

  # Real part should be a linear layer
  ŝ_real = Chain(Dense(s_shared_inner_layer_size, 1))

  ss = Dict(WillBallPassLine() => Chain(α, ŝ_shared, ŝ_bool),
            WhereWillBallEnd() => Chain(α, ŝ_shared, ŝ_real))

  # Is this correct? 

  s(prob) = simulate(prob.model, prob.query)
  train!(rng, s, ss, ubatch, ProblemRv(), embed_; batch_size = batch_size)
end