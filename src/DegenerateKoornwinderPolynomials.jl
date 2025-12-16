export DegenerateKoornwinderDomain, DegenerateSemiclassical_Operatormatrices

"""
    Smooth generalised Koornwinder domains are defined as
    Ω = {(x, y)∈ ℝ² | −δ·ρ(x) ≤ y ≤ δ·ρ(x), α ≤ x ≤ β}, 
    with ρ(x) being the square root of a polynomial and θ = deg(ρ²) ≥ 2.
"""
struct DegenerateKoornwinderDomain
    α::Float64    
    β::Float64
    δ::Float64
    ρ::Function        # Please use the form sqrt(Complex(...)) instead of sqrt(...)

    # factor out the endpoint singularities ρ(x) = sqrt((β-x)^a1 * (x-α)^b1) * ρ0(x)
    a1::Float64
    b1::Float64
    ρ0::Function

    θ::Int             # deg(ρ²)
    Fρ1::Function      # ρ²
    Fρ2::Function      # ρ * ρ'
end
# Do not use anonymous functions; they might lead to errors when using `Koornwinder_analysis_transform`.


"""
    Hierachically compute the operator matrices of the semiclassical OPs 
    orthonormal w.r.t. ρ^{2c+2k+1} on [α,β], for 0≤k≤N.
"""

function first_connection_matrix(gkd::DegenerateKoornwinderDomain, c, N)  

    α, β, ρ, θ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.ρ, gkd.θ, gkd.Fρ1, gkd.Fρ2

    deg = θ + 1                    
    μ₀_initial = zeros(deg)                 

    S  = Normalized(jacobi(0, 0, α..β))
    S1 = Normalized(jacobi(1, 1, α..β))
    x  = axes(S, 1)
    Ls = (S1 \ S)' * 0.5 * (β - α)
    Ds = (S1 \ (Derivative(x) * S)) * 0.5 * (β - α) 
    Js = jacobimatrix(S)
    Fρ1Js = S \ (Fρ1.(x) .* S)                  # ρ²[J(w_S^{(0,0)})]
    Gρ = z -> Fρ2(z) * (z - α) * (β - z)
    GρJs = S \ (Gρ.(x) .* S)                    # (ρρ'σₛ) [J(w_S^{(0,0)})] 

    μ₀_initial, _ = quadgk(y ->   ρ(y)^(2*c+1) .* S[y, 1:deg], α, β; rtol=1e-14, atol=1e-15)

    MAT = -(2*c+1) * GρJs[1:N, 1:N] + Fρ1Js[1:N, 1:N] * Ls[1:N, 1:N] * Ds[1:N, 1:N]
    MAT_aug = [MAT; Matrix(I, deg, N)]
    b_aug = vcat(zeros(N), μ₀_initial)
    μ₀ = (MAT_aug \ b_aug)[1:N]
    p₀ = S[(α+β)/2, 1]

    W₀ = GramMatrix(real.(μ₀), Js[1:N,1:N], p₀)
    R₀ = (cholesky(W₀).L)'

    return W₀, R₀
end


function semiclassical_jacobimatrices(gkd::DegenerateKoornwinderDomain, c, N)

    α, β, ρ, θ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.ρ, gkd.θ, gkd.Fρ1, gkd.Fρ2
    a1, b1, ρ0 = gkd.a1, gkd.b1, gkd.ρ0
    
    jacobi_matrices = Vector{BandedMatrix{Float64}}(undef, N+1)   
    connection_matrices = Vector{BandedMatrix{Float64}}(undef, N+1)    

    T = chebyshevt(α..β)
    x = axes(T, 1)
    u = T * (T \ (ρ0.(x) .^(2*c+1))) 
    Q = Normalized(jacobi(a1*(2*c+1)/2, b1*(2*c+1)/2, α..β))
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


function semiclassical_raisingmatrices(gkd::DegenerateKoornwinderDomain, J_set, λ3, N)  # λ3 is a positive odd number!

    α, β, Fρ1 = gkd.α, gkd.β, gkd.Fρ1

    T_set = Vector{BandedMatrix{Float64}}(undef, N+1) 
    f(z) = Fρ1(z)^((λ3 - 1) ÷ 2)

    for k in 0:N
        Jr = MATPOLY(f, J_set[k+1])  
        Rr = cholesky(Symmetric(Jr)).U
        T_set[k+1] = Rr      
    end

    return T_set
end


function semiclassical_derivativematrices(gkd::DegenerateKoornwinderDomain, R_set1, R_set2, c, N)

    α, β, ρ, θ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.ρ, gkd.θ, gkd.Fρ1, gkd.Fρ2
    a1, b1, ρ0 = gkd.a1, gkd.b1, gkd.ρ0
    D_set = Vector{BandedMatrix{Float64}}(undef, N+1)

    T = chebyshevt(α..β)
    x = axes(T, 1)
    H = Normalized(jacobi(a1*(2*c+1)/2, b1*(2*c+1)/2, α..β))
    y = axes(H,1)
    S = Normalized(jacobi(1 + a1*(2*c+1)/2, 1 + b1*(2*c+1)/2, α..β))
    Ds = (S \ (Derivative(y) * H) )* 0.5*(β-α)                   
    u = T * (T \ (ρ0.(x) .^(2*c+1)))                             
    uH = H \ (u .* H)                                            
    R = cholesky(Symmetric(uH[1:N, 1:N])).U
    v = T * (T \ ((β .- x).^(a1-1) .* (x .- α).^(b1-1) .* ρ0.(x) .^(2*c+3)))      
    vS = S \ (v .* S)                                             
    Rd = cholesky(Symmetric(vS[1:N, 1:N])).U 

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
    orthonormal w.r.t. (δ²-x²)ᶜ on [-δ,δ].
"""
function classical_jacobimatrix(gkd::DegenerateKoornwinderDomain, c)
    δ = gkd.δ
    P = Normalized(jacobi(c, c, -δ..δ))
    return jacobimatrix(P)
end

function classical_raisingmatrix(gkd::DegenerateKoornwinderDomain, c)
    δ = gkd.δ
    P = Normalized(jacobi(c, c, -δ..δ))
    P1 = Normalized(jacobi(c+1, c+1, -δ..δ))
    return (P1 \ P) * 0.5 * (2*δ)
end

function classical_derivativematrix(gkd::DegenerateKoornwinderDomain, c)
    δ = gkd.δ
    P = Normalized(jacobi(c, c, -δ..δ))
    P1 = Normalized(jacobi(c+1, c+1, -δ..δ))
    x = axes(P,1)
    return (P1 \  (Derivative(x) * P)) * 0.5 * (2*δ)
end

"""
    Wrap all required operator matrices for univariate OPs into a structure.
"""

struct DegenerateSemiclassicalOperatorMatrixSets
    J_set3::BMVec; R_set0::BMVec; J_set6::BMVec; R_set1::BMVec; J_set9::BMVec; J_set12::BMVec; 
    J_set10::BMVec; J_set13::BMVec; D_set1::BMVec; T_set1::BMVec;
    CJ2::AbstractMatrix{Float64}; CR1::AbstractMatrix{Float64}; CD1::AbstractMatrix{Float64};
    FCM::AbstractMatrix{Float64}; Gs::AbstractMatrix{Float64}
end

function DegenerateSemiclassical_Operatormatrices(gkd::DegenerateKoornwinderDomain, c, N)

    α, β, δ, ρ, θ, Fρ1, Fρ2 = gkd.α, gkd.β, gkd.δ, gkd.ρ, gkd.θ, gkd.Fρ1, gkd.Fρ2

    J_set3, R_set0 = semiclassical_jacobimatrices(gkd, c, N+1)
   
    J_set6, R_set1 = semiclassical_jacobimatrices(gkd, c+1, N+1)
  
    J_set9 = MATPOLY.(Fρ1, J_set3)
    J_set12 = MATPOLY.(Fρ1, J_set6)
    J_set10 = MATPOLY.(Fρ2, J_set3)
    J_set13 = MATPOLY.(Fρ2, J_set6)
    
    D_set1 = semiclassical_derivativematrices(gkd, R_set0, R_set1, c, N+1)

    T_set1 = semiclassical_raisingmatrices(gkd, J_set3, 3, N) 
  

    CJ2 = classical_jacobimatrix(gkd, c+1)
    CR1 = classical_raisingmatrix(gkd, c)
    CD1 = classical_derivativematrix(gkd, c)

    FCM = first_connection_matrix(gkd, c+1, 2*N+2)[2]
    Gs = first_connection_matrix(gkd, 0, 2*N+2)[1]

    return DegenerateSemiclassicalOperatorMatrixSets(
        J_set3, R_set0, J_set6, R_set1, J_set9, J_set12, J_set10, J_set13,
        D_set1, T_set1, CJ2, CR1, CD1, FCM, Gs
    )
end

"""
    Compute the differentiation matrices of the Generalised 
    Koornwinder Polynomials orthonormal w.r.t. (δ²*ρ²(x)-y²)ᶜ on Ω.
"""
@views function Koornwinder_Differentiation_Y_NW(ops::DegenerateSemiclassicalOperatorMatrixSets, N)   # D_y^{(c)}
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

@views function Koornwinder_Differentiation_Y_W(ops::DegenerateSemiclassicalOperatorMatrixSets, N)   # W_y^{(c+1)}
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

@views function Koornwinder_Differentiation_X_NW(gkd::DegenerateKoornwinderDomain, ops::DegenerateSemiclassicalOperatorMatrixSets, N)   # D_x^{(c)}
    θ = gkd.θ
    blocksizes = collect(1:N+1)
    DXNW = BandedBlockBandedMatrix(Zeros(sum(blocksizes), sum(blocksizes)),
                                   blocksizes, blocksizes, (θ-3, θ-1), (0, 2))

    @inbounds for k in 0:N
        if k >= 2
            C =  ops.CR1[k-1, k+1] .* ( ops.J_set9[k][1:N+1, 1:N+1] *
                 ( ops.D_set1[k-1][1:N+1, 1:N+1] / ops.T_set1[k-1][1:N+1, 1:N+1] / ops.T_set1[k][1:N+1, 1:N+1] ) ) .-
                 (-k* ops.CR1[k-1, k+1] + ops.CJ2[k-1,k]*ops.CD1[k, k+1]) .* 
                 ( ops.J_set10[k][1:N+1, 1:N+1]  / ops.T_set1[k][1:N+1, 1:N+1] ) 
        end

        for n in k:N
            for j in 1:(θ - 1)
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

@views function Koornwinder_Differentiation_X_W(gkd::DegenerateKoornwinderDomain, ops::DegenerateSemiclassicalOperatorMatrixSets, N)   # W_x^{(c+1)}
    θ = gkd.θ
    blocksizes = collect(1:N+1)
    DXW = BandedBlockBandedMatrix(Zeros(sum(blocksizes), sum(blocksizes)),
                                  blocksizes, blocksizes, (θ-1, θ-3), (2, 0))

    @inbounds for k in 0:N
        if k <= N-2
            C = (-ops.CR1[k+1, k+3]) .* ( ops.J_set12[k+1][1:N+1, 1:N+1] *
                 ( ops.D_set1[k+1][1:N+1, 1:N+1] / ops.T_set1[k+1][1:N+1, 1:N+1] / ops.T_set1[k+2][1:N+1, 1:N+1] ) ) .+
                 ( -(k+2) * ops.CR1[k+1, k+3] + ops.CJ2[k+1, k+2] * ops.CD1[k+2, k+3] ) .* 
                 ( ops.J_set13[k+1][1:N+1, 1:N+1]  / ops.T_set1[k+2][1:N+1, 1:N+1] ) 
        end

        for n in k:N
            for j in 1:(θ - 1)
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

@views function Koornwinder_Laplacian(gkd::DegenerateKoornwinderDomain, ops::DegenerateSemiclassicalOperatorMatrixSets, N)  # Δ_{W,(c+1)}^{(c+1)}

    DXNW = Koornwinder_Differentiation_X_NW(gkd, ops, N)
    DXW = Koornwinder_Differentiation_X_W(gkd, ops, N)
    DYNW = Koornwinder_Differentiation_Y_NW(ops, N)
    DYW = Koornwinder_Differentiation_Y_W(ops, N)
    Laplacian_W = DXNW * DXW + DYNW * DYW

    return Laplacian_W
end

"""
    Fast transforms for the Generalised Koornwinder Polynomials
    orthonormal w.r.t. (δ²*ρ²(x)-y²)ᶜ on Ω.
"""
function apply_stable_Q_action(ops::DegenerateSemiclassicalOperatorMatrixSets, k::Int, v_k::AbstractVector, qr_set)
    u_vec = copy(v_k)
    n = length(v_k)

    # Chain multiplications of Q factors
    j_start = iseven(k) ? 0 : 1
    for j in reverse(j_start:2:(k - 2))
        Qj, dj = qr_set[j+1]
        u_vec = Qj.Q * (dj .* u_vec)
    end

    if iseven(k)
        u_vec = ops.FCM[1:n, 1:n] \ u_vec
    else
        u_vec = ops.R_set1[2][1:n, 1:n] \ u_vec      
        u_vec = ops.FCM[1:n, 1:n] \ u_vec  
        u_vec = ops.Gs[1:n, 1:n] * u_vec       
    end

    return u_vec
end


function apply_stable_Q_inverse(ops::DegenerateSemiclassicalOperatorMatrixSets, k::Int, v::AbstractVector, qr_set)
    n = length(v)

    if iseven(k)
        v = ops.FCM[1:n, 1:n] * v
    else
        v = ops.Gs[1:n, 1:n] \ v
        v = ops.FCM[1:n, 1:n] * v
        v = ops.R_set1[2][1:n, 1:n] * v
    end

    # Chain multiplications of Qᵀ factors 
    j_start = iseven(k) ? 0 : 1
    for j in j_start:2:(k - 2)
        Qj, dj = qr_set[j+1]
        v = dj .* (Qj.Q' * v)  
    end

    return v
end

"""
    Analysis transform: function `f_RHS` --> coefficient vector `koornwinder_coef_f`
    Synthesis transform: coefficient vector `koornwinder_coef_u` --> function `u_approx`
"""

function Koornwinder_analysis_transform(gkd::DegenerateKoornwinderDomain, ops::DegenerateSemiclassicalOperatorMatrixSets, f_RHS, c, N)

    α, β, δ, ρ, θ, Fρ1 = gkd.α, gkd.β, gkd.δ, gkd.ρ, gkd.θ, gkd.Fρ1

    ft(x, t) = f_RHS(x, t * sqrt(Fρ1(x)))

    T₁ = chebyshevt(-δ..δ)
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
    P1 = (1 / δ)^c * Normalized(jacobi(c, c, -δ..δ)) 
    P2 = Normalized(jacobi(0, 0, α..β))
    R_T1 = jac2cheb(Id, c, c) * (jacobi(c, c, -δ..δ) \ P1)[1:N+1, 1:N+1]
    R_T2 = jac2cheb(Id, 0, 0) * (jacobi(0, 0, α..β) \ P2)[1:N+1, 1:N+1]
    inv_R_T1 = inv(R_T1)
    inv_R_T2 = inv(R_T2)
    J_set = [J[1:N+1, 1:N+1] for J in ops.J_set6]

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
        A_coef_f[k*(N + 1)+1 : ((k+1)*(N+1))] .= apply_stable_Q_inverse(ops, k, Bk, qr_set)
    end

    koornwinder_coef_f = zeros((N+1)*(N+2)÷2)
    for n = 0:N
        for k = 0:n 
            koornwinder_coef_f[n*(n+1)÷2 + k + 1] = A_coef_f[k*(N+1)+n-k+1]
        end
    end

    return BlockedArray(koornwinder_coef_f, collect(1:N+1))
end


function Koornwinder_synthesis_transform(gkd::DegenerateKoornwinderDomain, ops::DegenerateSemiclassicalOperatorMatrixSets, koornwinder_coef_u, c, N)

    α, β, δ, ρ, θ, Fρ1 = gkd.α, gkd.β, gkd.δ, gkd.ρ, gkd.θ, gkd.Fρ1
   
    T₁ = chebyshevt(-δ..δ)
    T₂ = chebyshevt(α..β)
    T² = RectPolynomial(T₂, T₁)
    t = axes(T₂, 1)
    x = axes(T₁, 1)
    w₁(x) = (δ - x)^c * (x + δ)^c
    w₂(t) = (ρ(t))^(2*c)
    u₁ = T₁ * (T₁ \ w₁.(x))
    u₂ = T₂ * (T₂ \ w₂.(t))
    Q1 = Normalized(jacobi(c, c, -δ..δ))
    Q2 = Normalized(jacobi(0, 0, α..β))
    WJp = real.((Q1 \ (u₁ .* Q1))[1:N+1, 1:N+1])
    WJs = real.((Q2 \ (u₂ .* Q2))[1:N+1, 1:N+1])

    Id = Matrix{Float64}(I, N+1, N+1)
    P1 = (1 / δ)^c * Q1 
    P2 = Q2
    R_T1 = jac2cheb(Id, c, c) * (jacobi(c, c, -δ..δ) \ P1)[1:N+1, 1:N+1]
    R_T2 = jac2cheb(Id, 0, 0) * (jacobi(0, 0, α..β) \ P2)[1:N+1, 1:N+1]

    A_coef_u = zeros((N+1)^2)
    for n = 0:N
        for k = 0:(N - n)
            A_coef_u[n*(N+1) + k + 1] = koornwinder_coef_u[(k + n)*(k + n + 1) ÷ 2 + n + 1]
        end
    end

    J_set = [J[1:N+1, 1:N+1] for J in ops.J_set6]

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

        QAk = apply_stable_Q_action(ops, k, A_k, qr_set)

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




