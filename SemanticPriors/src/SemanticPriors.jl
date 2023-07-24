module SemanticPriors

using Random
using Spec
using ProgressLogging

# 1. Define the integer language
# Write a small interpreter that executes an integer program
# Enumerate these programs in order to find the optimal program with respect to some loss function
# Finds the posterior distribution over integer sequences given some data (using a language model) 

# """### The Language

# Let’s consider a simple yet concrete setup, and consider a language $\mathcal{A}$ for representing integer sequences:

# $$
# \begin{align*}
# P &= x_0 = n, x_i = E\\
# E &= n \mid x_{i-1} \mid E*E \mid E+E\\
# n &= 1 \mid 2 \mid \dots

# \end{align*}
# $$

# An example program $P_1$ in this language is:

# $$
# \begin{align*}
# x_0 &= 0\\
# x_i &= x_{i-1} + 1
# \end{align*}
# $$

# Here’s another program, $P_2$:

# $$
# \begin{align*}
# x_0 &= 2\\
# x_i &= x_{i-1} * x_{i-1}
# \end{align*}
# $$
# """

"An expression in the language"
abstract type Exp end

struct IntLiteral <: Exp
  value::Int
end

struct Program <: Exp
  x0exp::IntLiteral
  XiExp::Exp
end

"X_{i-1}"
struct XnegiExp <: Exp end

"Current index `i`"
struct CurrentIndexExp <: Exp end

struct MulExpr <: Exp
  left::Exp
  right::Exp
end

struct AddExpr <: Exp
  left::Exp
  right::Exp
end

function example_program_1()
  Program(IntLiteral(0), AddExpr(XnegiExp(), IntLiteral(1)))
end

function example_program_2()
  Program(IntLiteral(2), MulExpr(XnegiExp(), XnegiExp()))
end


"Pretty print a program `p`"
function pretty_print(p::Program)
  println("x0 = ", p.x0exp.value)
  println("xi = ", pretty_print(p.XiExp))
  println()
end

function pretty_print(e::Exp)
  if e isa IntLiteral
    return string(e.value)
  elseif e isa XnegiExp
    return "x_{i-1}"
  elseif e isa CurrentIndexExp
    return "i"
  elseif e isa MulExpr
    return string(pretty_print(e.left), "*", pretty_print(e.right))
  elseif e isa AddExpr
    return string(pretty_print(e.left), "+", pretty_print(e.right))
  else
    error("Unknown expression type $(typeof(e))")
  end
end

# """### The Interpreter

"Interpret the program `p` for `nsteps` steps. returning vector of values of all steps"
function interpret(p::Program, nsteps)
  x0 = p.x0exp.value
  xi = x0
  values = [xi]
  for i in 1:nsteps - 1
    xi = interpret_exp(p.XiExp, xi, i)  # pass current index to the interpreter
    push!(values, xi)
  end
  return values
end

"Interpret the expression `e` given a value for `x_{i-1}`"
function interpret_exp(p::Exp, xi_1, i)
  if p isa IntLiteral
    return p.value
  elseif p isa XnegiExp
    return xi_1
  elseif p isa CurrentIndexExp
    return i
  elseif p isa MulExpr
    return interpret_exp(p.left, xi_1, i) * interpret_exp(p.right, xi_1, i)
  elseif p isa AddExpr
    return interpret_exp(p.left, xi_1, i) + interpret_exp(p.right, xi_1, i)
  else
    error("Unknown expression type $(typeof(p))")
  end
end

## """### The Sampler

"Sample a program from the prior"
function sample_program(rng::Random.AbstractRNG, maxn = 10)
  x0 = rand(rng, 0:maxn)
  return Program(IntLiteral(x0), sample_exp(rng))
end

sample_program(maxn = 10) = sample_program(Random.GLOBAL_RNG, maxn)

function sample_exp(rng, maxn = 10)
  # Add sampling for CurrentIndexExp
  p = rand(rng)
  if p < 0.25
    return IntLiteral(rand(rng, 0:maxn))
  elseif p < 0.5
    return XnegiExp()
  elseif p < 0.66
    return CurrentIndexExp()
  elseif p < 0.75
    return MulExpr(sample_exp(rng), sample_exp(rng))
  elseif p < 1.0
    return AddExpr(sample_exp(rng), sample_exp(rng))
  else
    return CurrentIndexExp()
  end
end

function program_size(p::Program)
  return exp_size(p.XiExp)
end

function exp_size(p::Exp)
  if p isa IntLiteral
    return 1
  elseif p isa XnegiExp
    return 1
  elseif p isa CurrentIndexExp
    return 1
  elseif p isa MulExpr
    return 1 + exp_size(p.left) + exp_size(p.right)
  elseif p isa AddExpr
    return 1 + exp_size(p.left) + exp_size(p.right)
  else
    error("Unknown expression type $(typeof(p))")
  end
end

# """### The Enumerator
using HTTP, JSON

# Define your API key and endpoint
const url = "https://api.openai.com/v1/chat/completions"
const api_key = ENV["OPENAI_API_KEY"]

# Define the headers
const headers = Dict(
    "Content-Type" => "application/json",
    "Authorization" => "Bearer " * api_key
)

extract_result(jsonmsg) = jsonmsg["choices"][1]["message"]["content"]
parse_result(jsonmsg) = parse_string_seq(extract_result(jsonmsg))

seq_as_string(seq::Vector{Int}) = join(string.(x for x in seq), " ")

"Convert string in form '1 2 3' to vector of integers"
function parse_string_seq(s)
  split_s = split(s)
  return parse.(Int, split_s)
end

function call_gpt(message; temperature = 0.9, model = "gpt-3.5-turbo", max_retries = 5)
  data = Dict(
      "model" => model,
      "messages" => [Dict("role" => "user", "content" => "$message")],
      "temperature" => temperature
  )

  # Convert the data to JSON
  body = JSON.json(data)

  for attempt in 1:max_retries
      try
          # Make the POST request
          response = HTTP.request("POST", url, headers, body)

          result = String(response.body)
          result_dict = JSON.Parser.parse(result)
          return parse_result(result_dict)
      catch e
          println("Error: $e")
          # If we've reached the maximum number of retries, rethrow the exception
          if attempt == max_retries
              rethrow(e)
          end
          # Otherwise, print a message and continue with the next iteration of the loop
          println("Attempt $attempt failed, retrying...")
          continue
      end
  end
end

"Sample (using GPT) from the posterior over integer sequences conditioned on prefix"
function sample_posterior(prefix, seq_len = 10)
  string_seq = seq_as_string(prefix)
  res = call_gpt("Complete the sequence of numbers so that the total number of numbers is  $seq_len (respond with the sequence in full and nothing else): $string_seq")
end
@pre sample_posterior(prefix, seq_len) = seq_len > length(prefix) "The sequence length must be at least as long as the prefix"

"Find the program that is optimal, according to loss function ℓ, conditional on prefix"
function sample_best_program(rng::AbstractRNG, prefix, seq_len, ℓ, nprograms_to_try; maxn = 10)
  @show seq = sample_posterior(prefix, seq_len)

  best_program = nothing
  best_loss = Inf

  for i in 1:nprograms_to_try
    p = sample_program(rng, maxn)
    result = interpret(p, seq_len)
    # @show result
    # @show seq
    if result == seq
      # return p

      loss = ℓ(p)
      if loss < best_loss
        best_loss = loss
        best_program = p
        pretty_print(p)
        println("Loss: ", loss)
      end  
    end
    # @show i
  end

  return best_program
end

function test_everything()
  nsamples = 10
  progs = []
  @withprogress for i = 1:nsamples
    res = sample_best_program(Random.GLOBAL_RNG, [1, 2, 4], 10, program_size, 10000)
    push!(progs, res)
  end
  progs
end

end # module SemanticPriors
