# paper at https://doi.org/10.1016/j.ejor.2005.10.060

using Catlab, AlgebraicPetri
using JuMP, HiGHS

to_graphviz(Presentation(LabelledPetriNet))

pn = @acset LabelledPetriNet begin
    S=6
    sname=Symbol.("p" .* string.(1:6))
    T=5
    tname=Symbol.("t" .* string.(1:5))
end

pn_is = Symbol.(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p6"])
pn_it = Symbol.(["t1", "t2", "t1", "t2", "t3", "t4", "t4", "t5"])

pn_ot = Symbol.(["t1", "t1", "t1", "t2", "t3", "t4", "t5"])
pn_os = Symbol.(["p1", "p3", "p2", "p4", "p5", "p6", "p2"])

for i in eachindex(pn_is)
    s_ix = only(incident(pn, pn_is[i], :sname))
    t_ix = only(incident(pn, pn_it[i], :tname))
    add_part!(pn, :I, it=t_ix, is=s_ix)
end

for i in eachindex(pn_os)
    s_ix = only(incident(pn, pn_os[i], :sname))
    t_ix = only(incident(pn, pn_ot[i], :tname))
    add_part!(pn, :O, ot=t_ix, os=s_ix)
end

to_graphviz(pn)
# to_graphviz(pn, graph_attrs=Dict(:rankdir=>"TD"))

# paper follows convention that matrices are places (rows) X transitions (cols)
# M places N transitions

"""
Generate integer programming model 1 from the paper (eq set 9)
"""
function gen_model_1(pn, K, m0, mf; optimizer=HiGHS.Optimizer)

    tm = TransitionMatrices(pn)
    # pre is C- and post is C+ matrices in paper
    pre, post = transpose(tm.input), transpose(tm.output)
    C = post .- pre

    mod = JuMP.Model(optimizer)

    # eqn 9d
    # X[i,t] i is index of transition, t index of step
    X = @variable(
        mod,
        X[1:nt(pn), 1:K] ≥ 0,
        Int
    )

    # eqn 9b
    for i in 1:K
        @constraint(
            mod,
            reduce(+, [C * X[:,j] for j in 1:i-1], init=zeros(AffExpr, ns(pn))) - (pre * X[:,i]) ≥ -m0            
        )
    end

    # eqn 9c
    @constraint(
        mod,
        reduce(+, [C * X[:,i] for i in 1:K]) == mf - m0
    )

    # obj 1 (vanishing identically)

    # # obj 2 (L1 norm of steps)
    # @objective(
    #     mod,
    #     Min,
    #     sum(X)
    # )

    return mod, X
end

function kmin_val(m)
    (2*m[1]) + 9
end

# initial and final states from sec 4 of paper
m0 = zeros(Int, ns(pn))
m0[1] = 4
m0[2] = 2

mf = deepcopy(m0)
mf[1] = 0
mf[5] = 17

Kmin = kmin_val(m0)

mod, X = gen_model_1(pn, Kmin, m0, mf)

optimize!(mod)