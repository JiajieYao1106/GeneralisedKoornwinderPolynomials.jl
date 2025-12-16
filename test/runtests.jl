using GeneralisedKoornwinderPolynomials, LinearAlgebra, Plots, Test
using SparseArrays, BlockBandedMatrices, BlockArrays, StaticArrays
using ClassicalOrthogonalPolynomials
import GeneralisedKoornwinderPolynomials: classical_raisingmatrix, classical_jacobimatrix, classical_derivativematrix
import GeneralisedKoornwinderPolynomials: Koornwinder_Differentiation_X_NW, Koornwinder_Differentiation_X_W, Koornwinder_Conversion_NW
import GeneralisedKoornwinderPolynomials: Koornwinder_Conversion_W, Koornwinder_Differentiation_Y_NW, Koornwinder_Differentiation_Y_W

α, β, γ, δ, θ, ϵ, N = 0.2, 0.8, -0.3, 0.3, 3, 1, 30

ρ(x)  = sqrt(Complex(1+x+3x^3) )
dρ(x) = nothing
Fρ1(x) = 1+x+3x^3
Fρ2(x) = 0.5 * (1 + 9x^2)

gkd1 = GeneralisedKoornwinderDomain(α, β, γ, δ, ρ, θ, ϵ, dρ, Fρ1, Fρ2)

ops1 = Semiclassical_Operatormatrices(gkd1, 0, 0, 0, 0, N)

JX = Koornwinder_Multiplication_X(ops1, N)                # J_x^{(0,0,0,0)} 
JY = Koornwinder_Multiplication_Y(gkd1, ops1, N)          # J_y^{(0,0,0,0)} 
DXNW = Koornwinder_Differentiation_X_NW(gkd1, ops1, N)    # D_x^{(0,0,0,0)}
DXW  = Koornwinder_Differentiation_X_W(gkd1, ops1, N)     # W_x^{(1,1,1,1)}
HNW = Koornwinder_Conversion_NW(ops1, N)                  # T_{(0,0,1,1)}^{(1,1,1,1)}
DYNW = Koornwinder_Differentiation_Y_NW(ops1, N)          # D_y^{(0,0,0,0)}
HW = Koornwinder_Conversion_W(ops1, N)                    # T_{W, (1,1,0,0)}^{(0,0,0,0)} 
DYW = Koornwinder_Differentiation_Y_W(ops1, N)            # W_y^{(1,1,1,1)}

Laplacian_W = DXNW * DXW + HNW * DYNW * HW * DYW          # Δ_{W,(1,1,1,1)}^{(1,1,1,1)}


@testset "Koornwinder_multiplication_by_x" begin
    a, b, c, d = 0, 0, 0, 0
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))
    
    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1) )                                                              
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ))  
                                                                                
    @test (x * R[x, n-k+1] * (rho)^k * P[y/rho, k+1] ≈ JX[Block(n,n+1)][k+1,k+1] * R[x, n-k] * rho^k * P[y/rho, k+1] + 
    JX[Block(n+1,n+1)][k+1,k+1] * R[x, n-k+1] * rho^k * P[y/rho, k+1] + 
    JX[Block(n+2,n+1)][k+1,k+1] * R[x, n-k+2] * rho^k * P[y/rho, k+1])
end


@testset "Koornwinder_multiplication_by_y" begin
    a, b, c, d = 0, 0, 0, 0
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))
    
    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1))         
    R1 = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+3))      
    R2 = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k-1))     
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ))      

    @test (y * R[x, n-k+1] * rho^k * P[y/rho, k+1] ≈ sum(JY[Block(n+j+1, n+1)][k+2,k+1] * R1[x, n-k+j] * rho^(k+1) * P[y/rho, k+2] + 
    JY[Block(n-j+1, n+1)][k,k+1] * R2[x, n-k-j+2] * rho^(k-1) * P[y/rho, k] for j in -θ+1:1)  )
end


@testset "Koornwinder_nonweighted_conversion" begin
    a, b, c, d = 0, 0, 1, 1
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1))
    R1 = LanczosPolynomial(@. (β-z)^(a+1) * (z-α)^(b+1) * (sqrt(Fρ1(z)))^(c+d+2*k+1))
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ)) 
    
    @test (R[x, n-k+1] * rho^k * P[y/rho, k+1] ≈ HNW[Block(n-1,n+1)][k+1,k+1] * R1[x, n-k-1] * rho^k * P[y/rho, k+1] + 
    HNW[Block(n,n+1)][k+1,k+1]  * R1[x, n-k] * rho^k * P[y/rho, k+1] + HNW[Block(n+1,n+1)][k+1,k+1]  * R1[x, n-k+1] * rho^k * P[y/rho, k+1])
end


@testset "Koornwinder_weighted_conversion" begin
    a, b, c, d = 1, 1, 0, 0
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))

    W1 = (β-x)^a * (x-α)^b * (y-γ*rho)^c * (δ*rho-y)^d
    W2 = (β-x)^(a-1) * (x-α)^(b-1) * (y-γ*rho)^c * (δ*rho-y)^d

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1))
    R1 = LanczosPolynomial(@. (β-z)^(a-1) * (z-α)^(b-1) * (sqrt(Fρ1(z)))^(c+d+2*k+1))
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ))     

    @test (W1 * R[x, n-k+1] * rho^k * P[y/rho, k+1] ≈ W2 * HW[Block(n+1,n+1)][k+1,k+1] * R1[x, n-k+1] * rho^k * P[y/rho, k+1] + 
    W2 * HW[Block(n+2,n+1)][k+1,k+1] * R1[x, n-k+2] * rho^k * P[y/rho, k+1] + 
    W2 * HW[Block(n+3,n+1)][k+1,k+1] * R1[x, n-k+3] * rho^k * P[y/rho, k+1])
end


@testset "Koornwinder_nonweighted_partial_x_differentiation" begin
    a, b, c, d = 0, 0, 0, 0
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))
    drho = Fρ2(x) / sqrt(Fρ1(x))

    CR = classical_raisingmatrix(gkd1, c, d)
    CJ = classical_jacobimatrix(gkd1, c+1, d+1)
    CD = classical_derivativematrix(gkd1, c, d)

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1)) 
    R1 = LanczosPolynomial(@. (β-z)^(a+1) * (z-α)^(b+1) * (sqrt(Fρ1(z)))^(c+d+2*k+3)) 
    R2 = LanczosPolynomial(@. (β-z)^(a+1) * (z-α)^(b+1) * (sqrt(Fρ1(z)))^(c+d+2*k-1))

    P0 = Normalized(jacobi(d, c, γ..δ)) 
    P1 = sqrt((2 / (δ - γ))^(d + c + 2)) * Normalized(jacobi(d+1, c+1, γ..δ))
    dP = sqrt((2 / (δ - γ))^(d + c)) * (Derivative(axes(P0, 1)) * P0)     
    dR = Derivative(z) * R
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ))

    @test k * CR[k+1,k+1] - CJ[k,k+1] * CD[k,k+1] < 1e-13


    @test (dR[x, n-k+1] * rho^(k) * P[y/rho, k+1] + R[x, n-k+1] * k * rho^(k-1) * drho * P[y/rho, k+1] +  
    R[x, n-k+1] * rho^k * dP[y/rho, k+1] * (-y * drho / rho^2) ≈ sum(DXNW[Block(n-j+1,n+1)][k+1,k+1] * 
    R1[x, n-k-j+1] * rho^k * P1[y/rho, k+1] for j in 1:max(θ+1, 3)) + sum(DXNW[Block(n+j+1,n+1)][k-1,k+1] * 
    R2[x, n-k+j+3] * rho^(k-2) * P1[y/rho, k-1] for j in (min(-3, -5+θ)):(θ-3)) ) 
end


@testset "Koornwinder_weighted_partial_x_differentiation" begin
    a, b, c, d = 1, 1, 1, 1
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))
    drho = Fρ2(x) / sqrt(Fρ1(x))
    
    W1 = (β-x)^a * (x-α)^b * (y-γ*rho)^c * (δ*rho-y)^d
    W2 = (β-x)^(a-1) * (x-α)^(b-1) * (y-γ*rho)^(c-1) * (δ*rho-y)^(d-1)
    dW1 = - a * (β-x)^(a-1) * (x-α)^b * (y-γ*rho)^c * (δ*rho-y)^d + 
            (β-x)^a * b * (x-α)^(b-1) * (y-γ*rho)^c * (δ*rho-y)^d + 
            (β-x)^a * (x-α)^b * c * (y-γ*rho)^(c-1) * (-γ * drho) * (δ*rho-y)^d +
            (β-x)^a * (x-α)^b * (y-γ*rho)^c * d * (δ*rho-y)^(d-1) * (δ * drho)

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1)) 
    R1 = LanczosPolynomial(@. (β-z)^(a-1) * (z-α)^(b-1) * (sqrt(Fρ1(z)))^(c+d+2*k-1)) 
    R2 = LanczosPolynomial(@. (β-z)^(a-1) * (z-α)^(b-1) * (sqrt(Fρ1(z)))^(c+d+2*k+3)) 
 
    P0 = Normalized(jacobi(d, c, γ..δ)) 
    P1 = sqrt((2 / (δ - γ))^(d + c - 2)) * Normalized(jacobi(d-1, c-1, γ..δ))
    dP = sqrt((2 / (δ - γ))^(d + c)) * (Derivative(axes(P0, 1)) * P0)     
    dR = Derivative(z) * R
    P = sqrt((2 / (δ - γ))^(d + c)) * Normalized(jacobi(d, c, γ..δ))

    @test (dW1 * R[x, n-k+1] * rho^(k) * P[y/rho, k+1] + W1 * dR[x, n-k+1] * rho^(k) * P[y/rho, k+1] + 
    W1 * R[x, n-k+1] * k * rho^(k-1) * drho * P[y/rho, k+1] + W1 * R[x, n-k+1] * rho^k * dP[y/rho, k+1] * (-y * drho / rho^2) ≈ 
    W2 * sum(DXW[Block(n+j+1,n+1)][k+1,k+1] * R1[x, n-k+j+1] * rho^k * P1[y/rho, k+1] for j in 1:max(θ+1, 3)) + 
    W2 * sum(DXW[Block(n-j+1,n+1)][k+3,k+1] * R2[x, n-k-j-1] * rho^(k+2) * P1[y/rho, k+3] for j in (min(-3, -5+θ)):(θ-3)) )
end


@testset "Koornwinder_nonweighted_partial_y_differentiation" begin
    a, b, c, d = 0, 0, 0, 0
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1)) 

    P = Normalized(jacobi(d, c, γ..δ)) 
    P1 = sqrt((2 / (δ - γ))^(d + c + 2)) * Normalized(jacobi(d+1, c+1, γ..δ))
    dP = sqrt((2 / (δ - γ))^(d + c)) * (Derivative(axes(P, 1)) * P)     

    @test R[x, n-k+1] * rho^(k-1) * dP[y/rho, k+1] ≈ DYNW[Block(n,n+1)][k,k+1] * R[x, n-k+1] * rho^(k-1) * P1[y/rho, k]
end


@testset "Koornwinder_weighted_partial_y_differentiation" begin
    a, b, c, d = 1, 1, 1, 1
    n, k = 8, 3
    x, y = 0.6, 0.2
    rho = sqrt(Fρ1(x))

    W1 = (β-x)^a * (x-α)^b * (y-γ*rho)^c * (δ*rho-y)^d
    dW1 = (β-x)^a * (x-α)^b * (c * (y-γ*rho)^(c-1) * (δ*rho-y)^d - d * (y-γ*rho)^c * (δ*rho-y)^(d-1) )
    W2 = (β-x)^a * (x-α)^b * (y-γ*rho)^(c-1) * (δ*rho-y)^(d-1)

    z = Inclusion(α..β)
    R = LanczosPolynomial(@. (β-z)^a * (z-α)^b * (sqrt(Fρ1(z)))^(c+d+2*k+1)) 

    P = Normalized(jacobi(d, c, γ..δ)) 
    P1 = sqrt((2 / (δ - γ))^(d + c - 2)) * Normalized(jacobi(d-1, c-1, γ..δ))
    dP = sqrt((2 / (δ - γ))^(d + c)) * (Derivative(axes(P, 1)) * P)     
    
    @test (W1 * R[x, n-k+1] * rho^(k-1) * dP[y/rho, k+1] + sqrt((2 / (δ - γ))^(d + c)) * dW1 * R[x, n-k+1] * rho^k * P[y/rho, k+1] ≈ 
    W2 * DYW[Block(n+2,n+1)][k+2,k+1] * R[x, n-k+1] * rho^(k+1) * P1[y/rho, k+2])
end


@testset "Koornwinder_fast_transforms" begin
    x, y = 0.6, 0.2

    u_exact(x,y) = (0.8 - x) * (x - 0.2) * (0.09 * Fρ1(x) - y^2) * exp(-100 * (x^2 + y^2))
    f_RHS(x,y) = Δ(u_exact)(x,y) 

    koornwinder_coef_f = Koornwinder_analysis_transform(gkd1, ops1, f_RHS, 1, 1, 1, 1, 30)

    koornwinder_coef_u =  Laplacian_W[Block.(1:26), Block.(1:26)] \ koornwinder_coef_f[Block.(1:26)]

    u_approx = Koornwinder_synthesis_transform(gkd1, ops1, koornwinder_coef_u, 1, 1, 1, 1, 25)

    Error = abs(u_approx(x,y) - u_exact(x,y))

    @test Error < 1e-8
end