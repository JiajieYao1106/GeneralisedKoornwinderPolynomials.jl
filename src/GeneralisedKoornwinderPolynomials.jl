module GeneralisedKoornwinderPolynomials
using ClassicalOrthogonalPolynomials, LinearAlgebra, SparseArrays, BandedMatrices, LazyBandedMatrices, Plots
using MultivariateOrthogonalPolynomials, StaticArrays
using ClassicalOrthogonalPolynomials: expand, coefficients, recurrencecoefficients
using FastTransforms, Test, BenchmarkTools, LazyArrays, QuadGK, SpecialFunctions
using BlockBandedMatrices, BlockArrays
using Polynomials, ForwardDiff
import Base:\

export Koornwinder_Multiplication_X, Koornwinder_Multiplication_Y
export Koornwinder_Laplacian, Koornwinder_Biharmonic, Koornwinder_analysis_transform, Koornwinder_synthesis_transform, Δ
export Koornwinder_Conversion_NW2, Koornwinder_Conversion_W2, GeneralisedKoornwinderDomain, Semiclassical_Operatormatrices

"""
    The generalised Koornwinder domain is defined as
    Ω = {(x, y)∈ ℝ² | γ·ρ(x) ≤ y ≤ δ·ρ(x), α ≤ x ≤ β}, 
    where ρ(x) is either a polynomial or the square root of a polynomial and θ = deg(ρ²) ≥ 2.
"""
struct GeneralisedKoornwinderDomain
    α::Float64
    β::Float64
    γ::Float64
    δ::Float64
    ρ::Function        # In case (ii), please use the form sqrt(Complex(...)) instead of sqrt(...)
    θ::Int             # deg(ρ²)
    ϵ::Int             # Indicator parameter: 0 in case (i) and 1 in case (ii)
    dρ::Function       # ρ' in case (i) and `nothing` in case (ii)
    Fρ1::Function      # ρ²
    Fρ2::Function      # ρ * ρ'
end
# Do not use anonymous functions; they might lead to errors when using `Koornwinder_analysis_transform`.

function Δ(f::Function)                     # Compute Δf(x,y) = ∂²f/∂x² + ∂²f/∂y²
    return function (x::Real, y::Real)
        g(v::SVector{2, T}) where T = f(v[1], v[2])
        v = @SVector [x, y]
        H = ForwardDiff.hessian(g, v)
        return tr(H)
    end
end

function MATPOLY(F::Function, J::AbstractMatrix)     # Compute F(J), where F must be a polynomial (real)

    coeffs = F(Polynomial([0,1])).coeffs
    n = size(J,1)
    result = coeffs[end] * I(n)
    
    for k in (length(coeffs)-1):-1:1
        result = J * result + coeffs[k] * I(n) # Horner
    end

    return result
end

"""
    Hierachically compute the operator matrices of the semiclassical OPs 
    orthonormal w.r.t. (β-x)ᵃ(x-α)ᵇ ρ(x)^{c+d+2k+1} on [α,β], for 0≤k≤N. 
"""

function first_connection_matrix(gkd::GeneralisedKoornwinderDomain, a, b, c, d, N)

    α, β, ρ, θ, ϵ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.ρ, gkd.θ, gkd.ϵ, gkd.Fρ1, gkd.Fρ2

    if ϵ==0 || ϵ==1                             # Clenshaw-based Cholesky
        T = chebyshevt(α..β)
        x = axes(T, 1)

        u = T * (T \ (ρ.(x) .^(c+d+1)))                                 
        Q = Normalized(jacobi(a, b, α..β))
        W₀ = Q \ (u .* Q)                                               
        R₀ = cholesky(Symmetric(W₀)).U
    else                                        # moment-based Cholesky
        deg = θ + 1                    
        μ₀_initial = zeros(deg)                 

        S  = Normalized(jacobi(a, b, α..β))
        S1 = Normalized(jacobi(a + 1, b + 1, α..β))
        x  = axes(S, 1)
        Ls = (S1 \ S)' * 0.5 * (β - α)
        Ds = (S1 \ (Derivative(x) * S)) * 0.5 * (β - α) 
        Js = jacobimatrix(S)
        Fρ1Js = S \ (Fρ1.(x) .* S)              # ρ² ∘ J(w_S^{(a,b)}) 
        Gρ = z -> Fρ2(z) * (z - α) * (β - z)
        GρJs = S \ (Gρ.(x) .* S)                # (ρρ'σₛ) ∘ J(w_S^{(a,b)}) 

        μ₀_initial, _ = quadgk(y -> sqrt((2 / (β - α))^(a + b)) * (β - y)^a * (y - α)^b * ρ(y)^(c+d+1) .* S[y, 1:deg], α, β; rtol=1e-14, atol=1e-15)

        MAT = -(c+d+1) * GρJs[1:N, 1:N] + Fρ1Js[1:N, 1:N] * Ls[1:N, 1:N] * Ds[1:N, 1:N]
        MAT_aug = [MAT; Matrix(I, deg, N)]
        b_aug = vcat(zeros(N), μ₀_initial)
        μ₀ = (MAT_aug \ b_aug)[1:N]
        p₀ = sqrt((2 / (β - α))^(a + b)) * S[(α+β)/2, 1]

        W₀ = GramMatrix(real.(μ₀), Js[1:N,1:N], p₀)
        R₀ = (cholesky(W₀).L)'
    end

    return W₀, R₀
end


function semiclassical_jacobimatrices(gkd::GeneralisedKoornwinderDomain, a, b, c, d, N)

    α, β, ρ, Fρ1 = gkd.α, gkd.β, gkd.ρ, gkd.Fρ1   

    jacobi_matrices = Vector{BandedMatrix{Float64}}(undef, N+1)   
    connection_matrices = Vector{BandedMatrix{Float64}}(undef, N+1)    

    T = chebyshevt(α..β)
    x = axes(T, 1)
    u = T * (T \ (ρ.(x) .^(c+d+1)))                                 
    Q = Normalized(jacobi(a, b, α..β))
    JS = jacobimatrix(Q)                                           
    uQ = Q \ (u .* Q)                                               
    R = cholesky(Symmetric(uQ[1:2*N+2, 1:2*N+2])).U                 
    
    # Compute the first semiclassical Jacobi matrix J₀ = J(w^{(a,b,c+d+1)}_R) in optimal complexity
    J₀ = BandedMatrix(Zeros(2*N+1, 2*N+1), (1,1))
    
    J₀[1, 1] = (R[1, 1] * JS[1, 1] + R[1, 2] * JS[2, 1]) / R[1, 1]

    for i in 1:2*N
        J₀[i, i+1] = (R[i+1, i+1] * JS[i+1, i]) / R[i, i]
        J₀[i+1, i] = J₀[i, i+1]
    end

    for i in 2:(2*N+1)
        J₀[i, i] = (R[i, i] * JS[i, i] + R[i, i+1] * JS[i+1, i]  - J₀[i, i-1] * R[i-1, i]) / R[i, i]
    end

    jacobi_matrices[1] = J₀
    connection_matrices[1] = real.(R[1:N+1, 1:N+1]) 
    
    # compute the semiclassical Jacobi matrices J = J(w^{(a,b,c+d+2k+1)}_R) in optimal complexity
    J = BandedMatrix(Zeros(N+1, N+1), (1, 1))

    for k in 1:N
        vJ₀ = MATPOLY(Fρ1, J₀)         
        R₀ = cholesky(Symmetric(vJ₀)).U                      
        J = BandedMatrix(Zeros(2*N+1-k, 2*N+1-k), (1, 1))
    
        J[1, 1] = (R₀[1, 1] * J₀[1, 1] + R₀[1, 2] * J₀[2, 1]) / R₀[1, 1]

        for i in 1:(2*N-k)
            J[i, i+1] = (R₀[i+1, i+1] * J₀[i+1, i]) / R₀[i, i]
            J[i+1, i] = J[i, i+1]
        end

        for i in 2:(2*N+1-k)
            J[i, i] = (R₀[i, i] * J₀[i, i] + R₀[i, i+1] * J₀[i+1, i]  - J[i, i-1] * R₀[i-1, i]) / R₀[i, i]
        end

        jacobi_matrices[k+1] = J
        connection_matrices[k+1] = R₀[1:N+1, 1:N+1]   
        
        J₀ = J
    end

    return jacobi_matrices, connection_matrices

end


function semiclassical_raisingmatrices(gkd::GeneralisedKoornwinderDomain, J_set, λ1, λ2, λ3, N)  # λ3 is a positive odd number!

    α, β, Fρ1 = gkd.α, gkd.β, gkd.Fρ1

    T_set = Vector{BandedMatrix{Float64}}(undef, N+1) 
    f = z -> (β - z)^λ1 * (z - α)^λ2 * Fρ1(z)^((λ3 - 1) ÷ 2)

    for k in 0:N
        Jr = MATPOLY(f, J_set[k+1])  
        Rr = cholesky(Symmetric(Jr)).U
        T_set[k+1] = Rr       
    end

    return T_set
end


function semiclassical_derivativematrices(gkd::GeneralisedKoornwinderDomain, R_set1, R_set2, a, b, c, d, N)

    α, β, ρ, θ = gkd.α, gkd.β, gkd.ρ, gkd.θ
    D_set = Vector{BandedMatrix{Float64}}(undef, N+1)

    T = chebyshevt(α..β)
    x = axes(T, 1)
    H = Normalized(jacobi(a, b, α..β))
    y = axes(H,1)
    S = Normalized(jacobi(a+1, b+1, α..β))
    Ds = (S \ (Derivative(y) * H) )* 0.5*(β-α)                      
    u = T * (T \ (ρ.(x) .^(c+d+1)))                                 
    uH = H \ (u .* H)                                              
    R = cholesky(Symmetric(uH[1:N, 1:N])).U
    ζu = T * (T \ (ρ.(x) .^(c+d+3)))                                
    ζuS = S \ (ζu .* S)                                             
    Rd = cholesky(Symmetric(ζuS[1:N, 1:N])).U 

    # compute the first semiclassical derivative matrix D₀ = D(w^{(a,b,2c+1)}_R) in optimal complexity
    D₀ = BandedMatrix(Zeros(N, N), (-1, θ+1))

    # D₀ = Rd[1:N, 1:N] * Ds[1:N, 1:N] / R[1:N, 1:N] => cubic complexity × 
    for i in 1:N-1         
        A = zeros(min(N-i,θ+1), min(N-i,θ+1))
        vec_b = zeros(min(N-i,θ+1))
        
        for j in 1:min(N-i,θ+1)
            for k in 1:j
                A[j, k] = R[i+k, i+j]
            end
            vec_b[j] = Rd[i, i+j-1] * Ds[i+j-1, i+j] 
        end
        
        solution = LowerTriangular(A) \ vec_b
        
        for j in 1:min(N-i,θ+1)
            D₀[i, i+j] = solution[j]
        end
    end

    D_set[1] = D₀

    # compute the semiclassical derivative matrices D = D(w^{(a,b,c+d+2k+1)}_R) in optimal complexity
    D = BandedMatrix{Float64}(undef, (N, N), (-1, θ+1))
    
    # D = R_set2[k+1][1:N, 1:N] * D₀ / R_set1[k+1][1:N, 1:N] => cubic complexity × 
    for k in 1:N
 
        for i in 1:N-1
            mat1 = zeros(min(N-i,θ+1), min(N-i,θ+1))
            mat2 = zeros(min(N-i,θ+1), min(N-i,θ+1))
            vec1 = zeros(min(N-i,θ+1))
            vec2 = zeros(min(N-i,θ+1))
            
            for j in 1:min(N-i,θ+1)
                for s in 1:j
                    mat1[j, s] = R_set1[k+1][i+s, i+j]
                    mat2[j, s] = D₀[i+s-1, i+j]
                end
                vec1[j] = R_set2[k+1][i, i+j-1]
            end
    
            vec2 = mat2 * vec1
            solution = LowerTriangular(mat1) \ vec2
            
            for j in 1:min(N-i,θ+1)
                D[i, i+j] = solution[j]
            end
        end

        D_set[k+1] = copy(D) 
        D₀ = D
        
    end

    return D_set

end

"""
    Compute the operator matrices of the classical OPs 
    orthonormal w.r.t. (δ-x)ᵈ(x-γ)ᶜ on [γ,δ].
"""
function classical_jacobimatrix(gkd::GeneralisedKoornwinderDomain, c, d)
    γ, δ = gkd.γ, gkd.δ
    P = Normalized(jacobi(d, c, γ..δ))
    return jacobimatrix(P)
end

function classical_raisingmatrix(gkd::GeneralisedKoornwinderDomain, c, d)
    γ, δ = gkd.γ, gkd.δ
    P = Normalized(jacobi(d, c, γ..δ))
    P1 = Normalized(jacobi(d+1, c+1, γ..δ))
    return (P1 \ P) * 0.5 * (δ - γ)
end

function classical_derivativematrix(gkd::GeneralisedKoornwinderDomain, c, d)
    γ, δ = gkd.γ, gkd.δ
    P = Normalized(jacobi(d, c, γ..δ))
    P1 = Normalized(jacobi(d+1, c+1, γ..δ))
    x = axes(P,1)
    return (P1 \  (Derivative(x) * P)) * 0.5 * (δ - γ)
end


"""
    Wrap all required operator matrices for univariate OPs into a structure.
"""
const BMVec = Vector{BandedMatrix{Float64}}

struct SemiclassicalOperatorMatrixSets
    J_set1::BMVec; R_set0::BMVec; R_set1::BMVec; J_set2::BMVec; J_set3::BMVec; J_set4::BMVec; 
    J_set5::BMVec; J_set6::BMVec; J_set7::BMVec; J_set8::BMVec; J_set9::BMVec; J_set10::BMVec;
    J_set11::BMVec; J_set12::BMVec; J_set13::BMVec; J_set14::BMVec; D_set1::BMVec;
    T_set1::BMVec; T_set2::BMVec; T_set3::BMVec; T_set4::BMVec;
    CJ1::AbstractMatrix{Float64}; CJ2::AbstractMatrix{Float64};
    CR1::AbstractMatrix{Float64}; CD1::AbstractMatrix{Float64};
    FCM::AbstractMatrix{Float64}; Gs::AbstractMatrix{Float64}
end

function Semiclassical_Operatormatrices(gkd::GeneralisedKoornwinderDomain, a, b, c, d, N)

    α, β, γ, δ, ρ, θ, ϵ, dρ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.γ, gkd.δ, gkd.ρ, gkd.θ, gkd.ϵ, gkd.dρ, gkd.Fρ1, gkd.Fρ2

    J_set1, R_set0 = semiclassical_jacobimatrices(gkd, a, b, c, d, N+1)
    J_set2 = ϵ == 0 ? MATPOLY.(ρ, J_set1) : BandedMatrix{Float64}[]
    J_set3, _ = semiclassical_jacobimatrices(gkd, a+1, b+1, c, d, N)
    J_set4 = ϵ == 0 ? MATPOLY.(ρ, J_set3) : BandedMatrix{Float64}[]
    J_set5 = MATPOLY.(z -> (β - z)*(z - α), J_set1)
    J_set6, R_set1 = semiclassical_jacobimatrices(gkd, a+1, b+1, c+1, d+1, N+1)
    J_set7 = ϵ == 0 ? MATPOLY.(ρ, J_set6) : BandedMatrix{Float64}[]
    J_set8 = ϵ == 0 ? MATPOLY.(dρ, J_set3) : BandedMatrix{Float64}[]
    J_set11 = ϵ == 0 ? MATPOLY.(dρ, J_set6) : BandedMatrix{Float64}[]
    J_set9 = MATPOLY.(Fρ1, J_set3)
    J_set12 = MATPOLY.(Fρ1, J_set6)
    J_set10 = MATPOLY.(Fρ2, J_set3)
    J_set13 = MATPOLY.(Fρ2, J_set6)
    J_set14, _ =  semiclassical_jacobimatrices(gkd, a, b, c+1, d+1, N)
    
    D_set1 = semiclassical_derivativematrices(gkd, R_set0, R_set1, a, b, c, d, N+1)

    T_set1 = semiclassical_raisingmatrices(gkd, J_set1, 0, 0, 3, N) 
    T_set2 = semiclassical_raisingmatrices(gkd, J_set14, 1, 1, 1, N)
    T_set3 = semiclassical_raisingmatrices(gkd, J_set1, 1, 1, 1, N)
    T_set4 = semiclassical_raisingmatrices(gkd, J_set1, 1, 1, 3, N)

    CJ1 = classical_jacobimatrix(gkd, a, b)
    CJ2 = classical_jacobimatrix(gkd, a+1, b+1)
    CR1 = classical_raisingmatrix(gkd, a, b)
    CD1 = classical_derivativematrix(gkd, a, b)

    FCM = first_connection_matrix(gkd, a+1, b+1, c+1, d+1, 2*N+2)[2]
    Gs = first_connection_matrix(gkd, a+1, b+1, 0, 0, 2*N+2)[1]

    return SemiclassicalOperatorMatrixSets(
        J_set1, R_set0, R_set1, J_set2, J_set3, J_set4, J_set5, J_set6, J_set7, J_set8,
        J_set9, J_set10, J_set11, J_set12, J_set13, J_set14, D_set1,
        T_set1, T_set2, T_set3, T_set4, CJ1, CJ2, CR1, CD1, FCM, Gs
    )
end



"""
    Compute the multiplication/conversion/differentiation matrices of the Generalised 
    Koornwinder Polynomials orthonormal w.r.t. (β-x)ᵃ (x-α)ᵇ (y-γ*ρ(x))ᶜ (δ*ρ(x)-y)ᵈ on Ω.
"""
@views function Koornwinder_Multiplication_X(ops::SemiclassicalOperatorMatrixSets, N)   # J_x^{(a,b,c,d)} 
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)

    HX = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (1, 1), (0, 0))
    @inbounds for n in 0:N
        for k in 0:n
            if n >= 1 && n-k >= 1
                view(HX, Block(n, n+1))[k+1,k+1] = ops.J_set1[k+1][n-k, n-k+1]
            end

            view(HX, Block(n+1, n+1))[k+1,k+1] = ops.J_set1[k+1][n-k+1, n-k+1]

            if n <= N-1
                view(HX, Block(n+2, n+1))[k+1,k+1] = ops.J_set1[k+1][n-k+1, n-k+2]
            end
        end
    end
    return HX
end


@views function Koornwinder_Multiplication_Y(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)   # J_y^{(a,b,c,d)}
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)
    θ = gkd.θ
    ϵ = gkd.ϵ

    HY = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (θ-1, θ-1), (1, 1))
    @inbounds for n in 0:N
        for k in 0:n
            if ϵ == 0 
                for j in (-θ÷2) : (θ÷2) 
                    if 1<=k+1<=n+j+1<=N+1 
                        view(HY, Block(n+j+1, n+1))[k+1,k+1] = ops.CJ1[k+1, k+1] * ops.J_set2[k+1][n-k+1, n+j-k+1]
                    end
                end
            end
            for j in -θ+1:1   
                if 1<=k+2<=n+j+1<=N+1 
                    view(HY, Block(n+j+1, n+1))[k+2,k+1] = ops.CJ1[k+1, k+2] * ops.T_set1[k+1][n-k+j, n-k+1]
                end
                if 1<=k<=n-j+1<=N+1 
                    view(HY, Block(n-j+1, n+1))[k,k+1] = ops.CJ1[k, k+1] * ops.T_set1[k][n-k+1, n-k-j+2]
                end
            end
        end
    end
    return HY
end


@views function Koornwinder_Conversion_NW(ops::SemiclassicalOperatorMatrixSets, N)    # T_{(a,b,c+1,d+1)}^{(a+1,b+1,c+1,d+1)}
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)

    HNW = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (0, 2), (0, 0))
    @inbounds for n in 0:N
        for k in 0:n
            if n >= 2 && n-k >= 2    
                view(HNW, Block(n-1,n+1))[k+1,k+1] = ops.T_set2[k+1][n-k-1, n-k+1]
            end
            if n >= 1 && n-k >= 1  
                view(HNW, Block(n,n+1))[k+1,k+1] = ops.T_set2[k+1][n-k, n-k+1]
            end
            view(HNW, Block(n+1,n+1))[k+1,k+1] = ops.T_set2[k+1][n-k+1, n-k+1]
        end
    end
    return HNW
end


@views function Koornwinder_Conversion_NW2(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)  # T_{(a,b,c,d)}^{(a+1,b+1,c+1,d+1)}
    θ = gkd.θ
    ϵ = gkd.ϵ
    blocksizes = collect(1:N+1)
    totaldim   = sum(blocksizes)

    HNW2 = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (θ-2, θ+2), (0, 2))
    @inbounds for k in 0:N
        if ϵ == 0 && k >= 1
            A = ops.CR1[k, k+1] .* (ops.J_set4[k+1][1:N+1, 1:N+1] * ops.T_set3[k+1][1:N+1, 1:N+1])
        end
        if k >= 2
            C = ops.CR1[k-1, k+1] .* (ops.J_set5[k+1][1:N+1, 1:N+1] * ( ops.T_set1[k][1:N+1, 1:N+1] / ops.T_set3[k][1:N+1, 1:N+1] )  )
        end

        for n in k:N
            if ϵ == 0 && k >= 1
                for j in (min(-5 + θ÷2, -3 - θ÷2)) : (-1 + θ÷2)
                    if 1 <= k <= n + j + 1 <= N + 1
                        view(HNW2, Block(n + j + 1, n + 1))[k, k + 1] = A[n + j - k + 2, n - k + 1]
                    end
                end
            end

            for j in 0:(2 + θ)
                if k <= n - j   
                    view(HNW2, Block(n - j + 1, n + 1))[k + 1, k + 1] = ops.CR1[k + 1, k + 1] * ops.T_set4[k + 1][n - j - k + 1, n - k + 1]
                end
            end

            if k >= 2
                for j in (min(-4, θ - 6)) : (θ - 2)
                    if 1 <= k - 1 <= n + j + 1 <= N + 1
                        view(HNW2, Block(n + j + 1, n + 1))[k - 1, k + 1] = C[n - k + 1, n + j - k + 3]
                    end
                end
            end
        end
    end
    return HNW2
end


@views function Koornwinder_Conversion_W(ops::SemiclassicalOperatorMatrixSets, N)    # T_{W, (a+1,b+1,c,d)}^{(a,b,c,d)} 
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)

    HW = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (2, 0), (0, 0))
    @inbounds for n in 0:N
        for k in 0:n 
            view(HW, Block(n+1,n+1))[k+1,k+1] = ops.T_set3[k+1][n-k+1, n-k+1]
            if n+2<=N+1
                view(HW, Block(n+2,n+1))[k+1,k+1] = ops.T_set3[k+1][n-k+1, n-k+2]
            end
            if n+3<=N+1
                view(HW, Block(n+3,n+1))[k+1,k+1] = ops.T_set3[k+1][n-k+1, n-k+3]
            end
        end
    end
    return HW
end


@views function Koornwinder_Conversion_W2(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)  # T_{W,(a+1,b+1,c+1,d+1)}^{(a,b,c,d)}
    θ = gkd.θ
    ϵ = gkd.ϵ
    blocksizes = collect(1:N+1)
    totaldim   = sum(blocksizes)

    HW2 = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (θ+2, θ-2), (2, 0) )

    @inbounds for k in 0:N   
        if ϵ == 0 && k+2<=N+1
            A = ops.CR1[k+1, k+2] .* (ops.J_set7[k+1][1:N+1, 1:N+1] * ops.T_set3[k+2][1:N+1, 1:N+1])
        end
        
        if k+3<=N+1
            C = ops.CR1[k+1, k+3] .* (ops.J_set5[k+3][1:N+1, 1:N+1] * ( ops.T_set1[k+2][1:N+1, 1:N+1] / ops.T_set3[k+2][1:N+1, 1:N+1] ) )
        end

        for n in k:N
            if ϵ == 0 
                for j in (min(-5 + θ÷2, -3 - θ÷2)) : (-1 + θ÷2)
                    if 1 <= k + 2 <= n - j + 1 <= N + 1
                        view(HW2, Block(n - j + 1, n + 1))[k + 2, k + 1] = A[n - k + 1, n - j - k]
                    end
                end
            end

            for j in 0:(2 + θ)
                if k <= n + j <= N
                    view(HW2, Block(n + j + 1, n + 1))[k + 1, k + 1] = ops.CR1[k + 1, k + 1] * ops.T_set4[k + 1][n - k + 1, n + j - k + 1]
                end
            end

            for j in (min(-4, θ - 6)) : (θ - 2)
                if 1 <= k + 3 <= n - j + 1 <= N + 1
                    view(HW2, Block(n - j + 1, n + 1))[k + 3, k + 1] = C[n - j - k - 1, n - k + 1]
                end
            end
        end
    end
    return HW2
end


@views function Koornwinder_Differentiation_Y_NW(ops::SemiclassicalOperatorMatrixSets, N)   # D_y^{(a,b,c,d)}
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)

    DYNW = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (-1, 1), (-1, 1))
    @inbounds for n in 0:N
        for k in 0:n
            if n >= 1 && k >= 1 
                view(DYNW, Block(n,n+1))[k,k+1] = ops.CD1[k,k+1]
            end
        end
    end
    return DYNW
end


@views function Koornwinder_Differentiation_Y_W(ops::SemiclassicalOperatorMatrixSets, N)   # W_y^{(a+1,b+1,c+1,d+1)}
    blocksizes = collect(1:N+1)
    totaldim = sum(blocksizes)

    DYW = BandedBlockBandedMatrix(Zeros(totaldim, totaldim), blocksizes, blocksizes, (1, -1), (1, -1))
    @inbounds for n in 0:N
        for k in 0:n
            if n+2 <= N+1
                view(DYW, Block(n+2,n+1))[k+2,k+1] = -ops.CD1[k+1,k+2]
            end
        end
    end
    return DYW
end


@views function Koornwinder_Differentiation_X_NW(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)   # D_x^{(a,b,c,d)}
    θ = gkd.θ
    ϵ = gkd.ϵ
    blocksizes = collect(1:N+1)
    DXNW = BandedBlockBandedMatrix(Zeros(sum(blocksizes), sum(blocksizes)),
                                   blocksizes, blocksizes, (θ-3, θ+1), (0, 2))

    @inbounds for k in 0:N
        if ϵ == 0 && k >= 1
            A =  ops.CR1[k, k+1] .* ( ops.J_set4[k+1][1:N+1, 1:N+1] *
                 ( ops.D_set1[k][1:N+1, 1:N+1] / ops.T_set1[k][1:N+1, 1:N+1] ) ) .-
                 (-k* ops.CR1[k, k+1] + ops.CJ2[k,k]*ops.CD1[k, k+1]) .* 
                 ( ops.J_set8[k+1][1:N+1, 1:N+1] * ops.T_set3[k+1][1:N+1, 1:N+1] )
        end
        if k >= 2
            C =  ops.CR1[k-1, k+1] .* ( ops.J_set9[k][1:N+1, 1:N+1] *
                 ( ops.D_set1[k-1][1:N+1, 1:N+1] / ops.T_set1[k-1][1:N+1, 1:N+1] / ops.T_set1[k][1:N+1, 1:N+1] ) ) .-
                 (-k* ops.CR1[k-1, k+1] + ops.CJ2[k-1,k]*ops.CD1[k, k+1]) .* 
                 ( ops.J_set10[k][1:N+1, 1:N+1] * ( ops.T_set3[k][1:N+1, 1:N+1] / ops.T_set1[k][1:N+1, 1:N+1] ) )
        end

        for n in k:N
            if ϵ == 0 && k >= 1
                for j in min(-2-θ÷2, -4+θ÷2) : (-2+θ÷2)
                    if 1 <= n + j + 1 <= N + 1 && k <= n + j + 1
                        view(DXNW, Block(n + j + 1, n + 1))[k, k + 1] =
                            A[n + j - k + 2, n - k + 1]
                    end
                end
            end

            for j in 1:(θ + 1)
                if 1 <= n - j + 1 <= N + 1 && k + 1 <= n - j + 1
                    view(DXNW, Block(n - j + 1, n + 1))[k + 1, k + 1] =
                        ops.CR1[k + 1, k + 1] * ops.D_set1[k + 1][n - j - k + 1, n - k + 1]
                end
            end

            if k >= 2
                for j in min(-3, -5 + θ) : (θ - 3)
                    if 1 <= n + j + 1 <= N + 1 && k - 1 <= n + j + 1
                        view(DXNW, Block(n + j + 1, n + 1))[k - 1, k + 1] =
                            C[n + j - k + 3, n - k + 1]
                    end
                end
            end
        end
    end

    return DXNW
end


@views function Koornwinder_Differentiation_X_W(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)   # W_x^{(a+1,b+1,c+1,d+1)}
    θ = gkd.θ
    ϵ = gkd.ϵ
    blocksizes = collect(1:N+1)
    DXW = BandedBlockBandedMatrix(Zeros(sum(blocksizes), sum(blocksizes)),
                                  blocksizes, blocksizes, (θ+1, θ-3), (2, 0))

    @inbounds for k in 0:N
        if ϵ == 0 && k <= N-1
            A = (-ops.CR1[k+1, k+2]) .* ( ops.J_set7[k+1][1:N+1, 1:N+1] *
                 ( ops.D_set1[k+1][1:N+1, 1:N+1] / ops.T_set1[k+1][1:N+1, 1:N+1] ) ) .+
                 ( -(k+1) * ops.CR1[k+1, k+2] + ops.CJ2[k+1, k+1] * ops.CD1[k+1, k+2] ) .* 
                 ( ops.J_set11[k+1][1:N+1, 1:N+1] * ops.T_set3[k+2][1:N+1, 1:N+1] )
        end
        if k <= N-2
            C = (-ops.CR1[k+1, k+3]) .* ( ops.J_set12[k+1][1:N+1, 1:N+1] *
                 ( ops.D_set1[k+1][1:N+1, 1:N+1] / ops.T_set1[k+1][1:N+1, 1:N+1] / ops.T_set1[k+2][1:N+1, 1:N+1] ) ) .+
                 ( -(k+2) * ops.CR1[k+1, k+3] + ops.CJ2[k+1, k+2] * ops.CD1[k+2, k+3] ) .* 
                 ( ops.J_set13[k+1][1:N+1, 1:N+1] * ( ops.T_set3[k+2][1:N+1, 1:N+1] / ops.T_set1[k+2][1:N+1, 1:N+1] ) )
        end

        for n in k:N
            if ϵ == 0 && k <= N-1
                for j in min(-2-θ÷2, -4+θ÷2) : (-2+θ÷2)
                    if 1 <= k + 2 <= n - j + 1 <= N + 1
                        view(DXW, Block(n - j + 1, n + 1))[k + 2, k + 1] =
                            A[n - k + 1, n - j - k]
                    end
                end
            end

            for j in 1:(θ + 1)
                if 1 <= k + 1 <= n + j + 1 <= N + 1
                    view(DXW, Block(n + j + 1, n + 1))[k + 1, k + 1] =
                        -ops.CR1[k + 1, k + 1] * ops.D_set1[k + 1][n - k + 1, n + j - k + 1]
                end
            end

            if k <= N - 2
                for j in min(-3, -5 + θ) : (θ - 3)
                    if 1 <= k + 3 <= n - j + 1 <= N + 1
                        view(DXW, Block(n - j + 1, n + 1))[k + 3, k + 1] =
                            C[n - k + 1, n - j - k - 1]
                    end
                end
            end
        end
    end

    return DXW
end


"""
    Compute the Laplacian/Biharmonic operator matrices of the Generalised 
    Koornwinder Polynomials orthonormal w.r.t. (β-x)ᵃ (x-α)ᵇ (y-γ*ρ(x))ᶜ (δ*ρ(x)-y)ᵈ on Ω.
"""
@views function Koornwinder_Laplacian(gkd::GeneralisedKoornwinderDomain, ops::SemiclassicalOperatorMatrixSets, N)  # Δ_{W,(1,1,1,1)}^{(1,1,1,1)}

    DXNW = Koornwinder_Differentiation_X_NW(gkd, ops, N)
    DXW = Koornwinder_Differentiation_X_W(gkd, ops, N)
    HNW = Koornwinder_Conversion_NW(ops, N)
    DYNW = Koornwinder_Differentiation_Y_NW(ops, N)
    HW = Koornwinder_Conversion_W(ops, N)
    DYW = Koornwinder_Differentiation_Y_W(ops, N)
    Laplacian_W = DXNW * DXW + HNW * DYNW * HW * DYW

    return Laplacian_W
end

@views function Koornwinder_Biharmonic(gkd::GeneralisedKoornwinderDomain, ops1::SemiclassicalOperatorMatrixSets, ops2::SemiclassicalOperatorMatrixSets, N)   # ₂Δ_{W, (2,2,2,2)}^{(2,2,2,2)}

    DXNW = Koornwinder_Differentiation_X_NW(gkd, ops1, N)
    DXW = Koornwinder_Differentiation_X_W(gkd, ops1, N)
    HNW = Koornwinder_Conversion_NW(ops1, N)
    DYNW = Koornwinder_Differentiation_Y_NW(ops1, N)
    HW = Koornwinder_Conversion_W(ops1, N)
    DYW = Koornwinder_Differentiation_Y_W(ops1, N)
    
    DXNW1 = Koornwinder_Differentiation_X_NW(gkd, ops2, N)
    DXW1 = Koornwinder_Differentiation_X_W(gkd, ops2, N)
    HNW1 = Koornwinder_Conversion_NW(ops2, N)
    DYNW1 = Koornwinder_Differentiation_Y_NW(ops2, N)
    HW1 = Koornwinder_Conversion_W(ops2, N)
    DYW1 = Koornwinder_Differentiation_Y_W(ops2, N)

    Biharmonic_W = (DXNW1 * DXNW + HNW1 * DYNW1 * HNW * DYNW) * (DXW * DXW1 + HW * DYW * HW1 * DYW1)
    
    return Biharmonic_W
end


"""
    Fast transforms for the Generalised Koornwinder Polynomials
    orthonormal w.r.t. (β-x)ᵃ (x-α)ᵇ (y-γ*ρ(x))ᶜ (δ*ρ(x)-y)ᵈ on Ω.
"""
function kron_Iw_mul(w, z)
    @assert size(w, 1) == 1 "w must be a 1×N row vector"
    N = size(w, 2)
    g = zeros(eltype(z), N)
    for i in 1:N
        g[i] = (w * z[(i-1)*N + 1 : i*N])[1]
    end
    return g
end


function kron_Iv_mul(v, y)
    @assert isa(v, Vector) "v must be a column vector of type Vector"
    N = length(v)
    h = similar(v, N^2)  #Vector{eltype(v)}(undef, N^2)
    for i in 1:N
        h[(i-1)*N+1 : i*N] .= y[i] * v
    end
    return h
end


function apply_stable_Q_action(ops1::SemiclassicalOperatorMatrixSets, k::Int, v_k::AbstractVector, qr_set)
    u_vec = copy(v_k)
    n = length(v_k)

    # Chain multiplications of Q factors
    j_start = iseven(k) ? 0 : 1
    for j in reverse(j_start:2:(k - 2))
        Qj, dj = qr_set[j+1]
        u_vec = Qj.Q * (dj .* u_vec)
    end

    if iseven(k)
        u_vec = ops1.FCM[1:n, 1:n] \ u_vec
    else
        u_vec = ops1.R_set1[2][1:n, 1:n] \ u_vec      
        u_vec = ops1.FCM[1:n, 1:n] \ u_vec  
        u_vec = ops1.Gs[1:n, 1:n] * u_vec       
    end

    return u_vec
end


function apply_stable_Q_inverse(ops1::SemiclassicalOperatorMatrixSets, k::Int, v::AbstractVector, qr_set)
    n = length(v)

    if iseven(k)
        v = ops1.FCM[1:n, 1:n] * v
    else
        v = ops1.Gs[1:n, 1:n] \ v
        v = ops1.FCM[1:n, 1:n] * v
        v = ops1.R_set1[2][1:n, 1:n] * v
    end

    # Chain multiplications of Qᵀ factors 
    j_start = iseven(k) ? 0 : 1
    for j in j_start:2:(k - 2)
        Qj, dj = qr_set[j+1]
        v = dj .* (Qj.Q' * v)  
    end

    return v
end


# Directly using `T² \ splat(ft).(axes(T², 1))` may cause an OOM error for some complicated `ft`.
function Cheb²_safe_coefficients(T²::RectPolynomial, ft, max_nblocks)

    T²ₙ = T²[:, Block.(Base.OneTo(max_nblocks))]
    T²ₙ_coef_f = T²ₙ \ splat(ft).(axes(T²ₙ,1))

    if norm(T²ₙ_coef_f[Block(max_nblocks)]) > 1e-16
        T²_coef_f = T²ₙ_coef_f
    else
        T²_coef_f = T² \ splat(ft).(axes(T²,1))
    end

    return T²_coef_f

end

"""
    Analysis transform: function `f_RHS` --> coefficient vector `koornwinder_coef_f`
    Synthesis transform: coefficient vector `koornwinder_coef_u` --> function `u_approx`
"""


function Koornwinder_analysis_transform(gkd::GeneralisedKoornwinderDomain, ops1::SemiclassicalOperatorMatrixSets, f_RHS, a, b, c, d, N)

    α, β, γ, δ, ρ, θ, Fρ1 = gkd.α, gkd.β, gkd.γ, gkd.δ, gkd.ρ, gkd.θ, gkd.Fρ1

    ft(x, t) = f_RHS(x, t * sqrt(Fρ1(x)))

    T₁ = chebyshevt(γ..δ)
    T₂ = chebyshevt(α..β)
    T² = RectPolynomial(T₂, T₁)
    
    T²_coef_f = Cheb²_safe_coefficients(T², ft, 2000)  

    Kronecker_coef_f = zeros((N+1)^2)
    for n = 0:N
        for k = 0:N 
            Kronecker_coef_f[k*(N+1) + n + 1] = T²_coef_f[Block(n + k + 1)][n + 1]
        end
    end

    Id = Matrix{Float64}(I, N+1, N+1)
    P1 = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ)) 
    P2 = sqrt((2 / (β - α))^(a + b)) * Normalized(jacobi(a, b, α..β))
    R_T1 = jac2cheb(Id, d, c) * (jacobi(d, c, γ..δ) \ P1)[1:N+1, 1:N+1]
    R_T2 = jac2cheb(Id, a, b) * (jacobi(a, b, α..β) \ P2)[1:N+1, 1:N+1]
    inv_R_T1 = inv(R_T1)
    inv_R_T2 = inv(R_T2)
    J_set = [J[1:N+1, 1:N+1] for J in ops1.J_set6]

    # Correct the direction to ensure positive-phase QR 
    qr_raw = qr.(MATPOLY.(Fρ1, J_set))
    qr_set = [let
        F = qr_raw[k]
        d = sign.(diag(F.R))
        d[d .== 0] .= 1.0
        (F, d)
    end for k in eachindex(qr_raw)]

    A_coef_f = Vector{eltype(Kronecker_coef_f)}(undef, (N + 1)^2)
    for k in 0:N
        Bk = inv_R_T2 * kron_Iw_mul((inv_R_T1[k+1, :])', Kronecker_coef_f)
        A_coef_f[k*(N + 1)+1 : ((k+1)*(N+1))] .= apply_stable_Q_inverse(ops1, k, Bk, qr_set)
    end

    koornwinder_coef_f = zeros((N+1)*(N+2)÷2)
    for n = 0:N
        for k = 0:n 
            koornwinder_coef_f[n*(n+1)÷2 + k + 1] = A_coef_f[k*(N+1)+n-k+1]
        end
    end

    return BlockedArray(koornwinder_coef_f, collect(1:N+1))
end




function Koornwinder_synthesis_transform(gkd::GeneralisedKoornwinderDomain, ops1::SemiclassicalOperatorMatrixSets, koornwinder_coef_u, a, b, c, d, N)

    α, β, γ, δ, ρ, θ, Fρ1 = gkd.α, gkd.β, gkd.γ, gkd.δ, gkd.ρ, gkd.θ, gkd.Fρ1

    T₁ = chebyshevt(γ..δ)
    T₂ = chebyshevt(α..β)
    T² = RectPolynomial(T₂, T₁)
    t = axes(T₂, 1)
    x = axes(T₁, 1)
    w₁(x) = (δ - x)^d * (x - γ)^c
    w₂(t) = (β - t)^a * (t - α)^b * (ρ(t))^(c + d)
    u₁ = T₁ * (T₁ \ w₁.(x))
    u₂ = T₂ * (T₂ \ w₂.(t))
    Q1 = Normalized(jacobi(d, c, γ..δ))
    Q2 = Normalized(jacobi(a, b, α..β))
    WJp = real.((Q1 \ (u₁ .* Q1))[1:N+1, 1:N+1])
    WJs = real.((Q2 \ (u₂ .* Q2))[1:N+1, 1:N+1])

    Id = Matrix{Float64}(I, N+1, N+1)
    P1 = sqrt((2 / (δ - γ))^(d + c)) * Q1 
    P2 = sqrt((2 / (β - α))^(a + b)) * Q2
    R_T1 = jac2cheb(Id, d, c) * (jacobi(d, c, γ..δ) \ P1)[1:N+1, 1:N+1]
    R_T2 = jac2cheb(Id, a, b) * (jacobi(a, b, α..β) \ P2)[1:N+1, 1:N+1]

    A_coef_u = zeros((N+1)^2)
    for n = 0:N
        for k = 0:(N - n)
            A_coef_u[n*(N+1) + k + 1] = koornwinder_coef_u[(k + n)*(k + n + 1) ÷ 2 + n + 1]
        end
    end

    J_set = [J[1:N+1, 1:N+1] for J in ops1.J_set6]

    # Correct the direction to ensure positive-phase QR 
    qr_raw = qr.(MATPOLY.(Fρ1, J_set))
    qr_set = [let
        F = qr_raw[k]
        R = F.R
        d = sign.(diag(R)) 
        d[d .== 0] .= 1.0
        (F, d)  
    end for k in eachindex(qr_raw)]

    Kronecker_coef_u = zeros(eltype(A_coef_u), (N + 1)^2)
    for k = 0:N
        A_k = A_coef_u[(k*(N+1)+1):((k+1)*(N+1))]

        QAk = apply_stable_Q_action(ops1, k, A_k, qr_set)

        Kronecker_coef_u .+= kron_Iv_mul(R_T1 * WJp[:, k+1], R_T2 * WJs * QAk)
    end

    T²_coef_u = zeros((N+1)*(N+2) ÷ 2)
    for n = 0:N
        for k = 0:n 
            T²_coef_u[n*(n+1) ÷ 2 + k + 1] = Kronecker_coef_u[(n - k)*(N+1) + k + 1]
        end
    end

    Tₙ² = T²[:, Block.(1:N+1)]
    u = Tₙ² * T²_coef_u

    u_approx(x,y) = u[SVector(x, y/sqrt(Fρ1(x)))] 

    return u_approx
end


"""
    Solving the linear system:  converting BandedBlockBandedMatrix{Float64} to 
    SparseMatrixCSC{Float64, Int64} can improve the efficiency and resolve memory issues.
"""

function \(A::BandedBlockBandedMatrix{T}, b::BlockedArray{T, 1, <:AbstractVector{T}, <:Tuple}) where T
    x = sparse(A) \ parent(b)  
    return BlockedArray(x, 1:blocksize(b,1) )  
end


# include the smooth domain case
include("DegenerateKoornwinderPolynomials.jl")


end # module


