# GeneralisedKoornwinderPolynomials.jl
[![Build Status](https://github.com/JiajieYao1106/GeneralisedKoornwinderPolynomials.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JiajieYao1106/GeneralisedKoornwinderPolynomials.jl)

A Julia package for solving partial differential equations on domains bounded by planar algebraic curves.

This experimental package implements generalised Koornwinder polynomials on the `GeneralisedKoornwinderDomain` `{(x, y)∈ ℝ² | γ*ρ(x) ≤ y ≤ δ*ρ(x), α ≤ x ≤ β}`
with respect to the weight `(β-x)ᵃ (x-α)ᵇ (y-γ*ρ(x))ᶜ (δ*ρ(x)-y)ᵈ`. Here `ρ(x)` is either a polynomial (`if ϵ == 0`) or the square root of a polynomial (`if ϵ == 1`), with `θ = deg(ρ(x)²)` in both cases.

> ⚠️**TODO**: Implement lazy storage and manipulation of infinite-dimensional matrices/vectors.

# Example Usage

Here is an example of how the package can be used. We consider the Poisson problem
```math 
\Delta  u(x,y) = f(x,y) \:\:\text{in} \:\:\Omega, \quad  u(x,y) = 0 \:\:\text{on} \:\:\partial \Omega,
```
where $f(x,y) = e^{-100(x^2 + y^2)}$ and $\Omega$ is a fish-shaped domain 
```math
\left\{ (x,y)\in ℝ² | y^2 \leq 0.09 (0.5 + x)(1 + 6x^2 - 20x^4 + 15x^6), \quad 0 \leq x \leq 1\right\}.
```
```julia
julia> using GeneralisedKoornwinderPolynomials, LinearAlgebra, Plots

julia> using SparseArrays, BlockBandedMatrices, BlockArrays, StaticArrays

julia> α, β, γ, δ, θ, ϵ = 0.0, 1.0, -0.3, 0.3, 7, 1;

julia> ρ(x)  = sqrt(Complex((0.5 + x) * (1 + 6x^2 - 20x^4 + 15x^6)));

julia> dρ(x) = nothing;

julia> Fρ1(x) = (0.5 + x) * (1 + 6x^2 - 20x^4 + 15x^6);  # ρ²(x) 

julia> Fρ2(x) = 0.5 * (1 + 6x + 18x^2 - 40x^3 - 100x^4 + 45x^5 + 105x^6);  # ρ'(x) * ρ(x)

julia> gkd1 = GeneralisedKoornwinderDomain(α, β, γ, δ, ρ, θ, ϵ, dρ, Fρ1, Fρ2);
```

In our sparse spectral method, we construct and solve the discretised linear system
```math
\Delta_{W,(1,1,1,1)}^{(1,1,1,1)} \,\tilde{u}_W = \tilde{f}_H.
```

💻STEP ONE: construct the sparse matrix representation $\Delta_{W,(1,1,1,1)}^{(1,1,1,1)}$ of the Laplace operator $\Delta$
```julia
julia> N = 30;

julia> ops1 = Semiclassical_Operatormatrices(gkd1, 0, 0, 0, 0, N);

julia> Laplacian_W = Koornwinder_Laplacian(gkd1, ops1, N)
31×31-blocked 496×496 BandedBlockBandedMatrix{Float64} with block-bandwidths (12, 12) and sub-block-bandwidths block-bandwidths (2, 2) with data 25×31-blocked 125×496 BlockedMatrix{Float64}:
 -0.781887   │  -0.133796    0.0        │
 ────────────┼──────────────────────────┼  …    ⋅        │    ⋅           ⋅   │   ⋅ 
 ───────────┼────────────────────┼─────
 -0.133796   │  -1.16        0.0        │       ⋅        │    ⋅           ⋅   │   ⋅ 
  0.0        │   0.0        -1.72929    │
 ────────────┼──────────────────────────┼       ⋅        │    ⋅           ⋅   │   ⋅ 
 ───────────┼────────────────────┼─────
  0.135412   │  -0.440601    0.0        │       ⋅        │    ⋅           ⋅   │   ⋅ 
  0.0        │   0.0        -0.166728   │       ⋅        │    ⋅           ⋅   │   ⋅ 
  0.0494787  │   0.0210104   0.0        │
 ────────────┼──────────────────────────┼  …    ⋅        │    ⋅           ⋅   │   ⋅ 
 ───────────┼────────────────────┼─────
 -0.308997   │  -0.0719209   0.0        │       ⋅        │    ⋅           ⋅   │   ⋅ 
  0.0        │   0.0         0.159073   │       ⋅        │    ⋅           ⋅   │   ⋅ 
  0.0        │   0.135513    0.0        │       ⋅        │    ⋅           ⋅   │   ⋅ 
   ⋅         │    ⋅          0.0562969  │
 ────────────┼──────────────────────────┼       ⋅        │    ⋅           ⋅   │   ⋅ 
 ───────────┼────────────────────┼─────
  ⋮                                  ⋱                               ⋮
   ⋅         │    ⋅           ⋅         │       ⋅        │    ⋅           ⋅   │   ⋅ 
   ⋅         │    ⋅           ⋅         │       ⋅        │    ⋅           ⋅   │   ⋅ 
   ⋅         │    ⋅           ⋅         │       ⋅        │    ⋅           ⋅   │   ⋅ 
   ⋅         │    ⋅           ⋅         │  …   0.0       │    ⋅           ⋅   │   ⋅ 
   ⋅         │    ⋅           ⋅         │      0.0       │   0.0          ⋅   │   ⋅ 
   ⋅         │    ⋅           ⋅         │     -0.160335  │   0.0         0.0  │   ⋅ 
   ⋅         │    ⋅           ⋅         │      0.0       │  -0.00132314  0.0  │  0.0
   ⋅         │    ⋅           ⋅         │      0.0       │   0.0         0.0  │  0.0
   ⋅         │    ⋅           ⋅         │  …    ⋅        │   0.0         0.0  │  0.0
```
💻STEP TWO: compute $\tilde{f}_H$ by expanding the RHS function $f(x,y)$ in the Koornwinder basis $H^{(1,1,1,1)}$
```julia
julia> f_RHS(x,y) = exp(-100 * (x^2 + y^2));

julia> koornwinder_coef_f = Koornwinder_analysis_transform(gkd1, ops1, f_RHS, 1, 1, 1, 1, N)
31-blocked 496-element BlockedVector{Float64}:
  0.00047033271603753787
 ───────────────────────
 -0.001028220501601279  
  0.0                   
 ───────────────────────
  0.0015122662867041597 
  0.0                   
 -0.000139383654910867  
 ───────────────────────
 -0.0017355651127501806 
  0.0                   
  0.0003412666118793805 
 -0.0                   
 ───────────────────────
  ⋮
 -1.5524148786142254e-10
  0.0                   
  9.829666102620701e-12 
  0.0                   
 -2.0129693280679017e-13
 -0.0                   
  4.350115865517139e-15 
  0.0                   
 -2.2650306367515262e-19
```
💻STEP Three: compute $\tilde{u}_W$ by solving the above discretised linear system
```julia
julia> koornwinder_coef_u  = Laplacian_W[Block.(1:N), Block.(1:N)] \ koornwinder_coef_f[Block.(1:N)]
30-blocked 465-element BlockedVector{Float64}:
 -0.0016736416215110655 
 ───────────────────────
  0.002334983323327797  
 -0.0                   
 ───────────────────────
 -0.0026328080298425663 
 -0.0                   
  0.0001803660551994913 
 ───────────────────────
  0.002367964566209051  
 -0.0                   
 -0.0002889796103613488 
 -0.0                   
 ───────────────────────
  ⋮
 -0.0                   
  3.095711930709763e-10 
 -0.0                   
  4.267158906321072e-11 
 -0.0                   
  2.2882080932322378e-11
 -0.0                   
  1.5212299279017298e-11
 -0.0             
```
💻STEP Four: obtain the solution function $u(x,y)$ from $\tilde{u}_W$ using the weighted Koornwinder basis $W^{(1,1,1,1)}$
```julia
julia> u_approx = Koornwinder_synthesis_transform(gkd1, ops1, koornwinder_coef_u, 1, 1, 1, 1, N-1)
(::GeneralisedKoornwinderPolynomials.var"#u_approx#23"{QuasiArrays.ApplyQuasiVector{Float64, typeof(*), Tuple{MultivariateOrthogonalPolynomials.RectPolynomial{Float64, Tuple{QuasiArrays.SubQuasiArray{Float64, 2, ClassicalOrthogonalPolynomials.ChebyshevT{Float64}, Tuple{ContinuumArrays.AffineMap{Float64, QuasiArrays.Inclusion{Float64, IntervalSets.ClosedInterval{Float64}}, QuasiArrays.Inclusion{Float64, DomainSets.ChebyshevInterval{Float64}}}, Base.Slice{InfiniteArrays.OneToInf{Int64}}}, false}, QuasiArrays.SubQuasiArray{Float64, 2, ClassicalOrthogonalPolynomials.ChebyshevT{Float64}, Tuple{ContinuumArrays.AffineMap{Float64, QuasiArrays.Inclusion{Float64, IntervalSets.ClosedInterval{Float64}}, QuasiArrays.Inclusion{Float64, DomainSets.ChebyshevInterval{Float64}}}, Base.Slice{InfiniteArrays.OneToInf{Int64}}}, false}}}, BlockedVector{Float64, LazyArrays.CachedArray{Float64, 1, Vector{Float64}, FillArrays.Zeros{Float64, 1, Tuple{InfiniteArrays.OneToInf{Int64}}}}, Tuple{BlockedOneTo{Int64, ArrayLayouts.RangeCumsum{Int64, InfiniteArrays.OneToInf{Int64}}}}}}}, typeof(Fρ1)}) (generic function with 1 method)
julia> u_approx(0.1, 0.2)   # Evaluate
-0.0002722868634110348
```
📣Now we can plot the solution for this Poisson problem!
```julia
julia> x_vals = range(α, β, length=500); y_vals = range(-0.61, 0.61, length=500);

julia> z = [γ*sqrt(Fρ1(x)) ≤ y ≤ δ*sqrt(Fρ1(x)) ? u_approx(x,y) : NaN for x in x_vals, y in y_vals];

julia> contourf(x_vals, y_vals, z'; aspect_ratio=:equal, xlims=(α,β), ylims=(-0.61,0.61), color=:inferno, linewidth = 0, levels=50, xlabel="\$x\$", ylabel="\$y\$")
```
![README_FIG](/README_FIG.png)

📌When $\Omega$ is a smooth domain, i.e., it is a `DegenerateKoornwinderDomain`, the corresponding generalised Koornwinder polynomials reduce to a degenerate form. For implementation details, see [DegenerateKoornwinderPolynomials.jl](/src/DegenerateKoornwinderPolynomials.jl).

📌See the experiments file for more examples:

[Poisson equation on a bowling-pin-shaped smooth domain](/experiments/BowlingPinDomain_Poisson.ipynb)

[Screened Poisson equation on a fish-shaped domain](/experiments/FishDomain_ScreenedPoisson.ipynb)

[Variable-coefficient Helmholtz equation on a vest-shaped domain](/experiments/VestDomain_Helmholtz)

[Biharmonic equation on a curvilinear trapezium](/experiments/CurvilinearTrapezium_Biharmonic.ipynb)
