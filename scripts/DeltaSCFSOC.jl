module DeltaSCFSOC

using DrWatson
quickactivate("$(pwd())../")
using LinearAlgebra, Plots

const ħ = 1.0545718e-34

@doc raw"""
    Wavefunction(n::Int64)

Define a wavefunction for a given n in a basis of $l$, $m_l$, and $m_s$

The length of the wavefunction vector is given by
```math
\sum_{n=1}^{n} \sum_{l=0}^{n-1} 2(2l + 1)
```

# Examples
```jldoctest
julia> Ψ = Wavefunction(2)
```
"""
mutable struct Wavefunction
    Ψ::Vector{Tuple{Int64,Complex{Int64},ComplexF64}}
    l::Vector{Int16}
    m_l::Vector{Complex{Int64}}
    m_s::Vector{ComplexF64}

    function Wavefunction(n::Int64)
        # Ensure n is greater than 0
        if n < 0
            throw(ArgumentError("n must be ∈ ℕ₀"))
        end

        # Calculate the total number of combinations
        len_Ψ = sum((2 * l + 1) * 2 for l = 0:n-1)

        # Create a matrix for all possible values of quantum numbers
        Ψ = Vector{Tuple{Int64,Complex{Int64},ComplexF64}}(undef, len_Ψ)

        idx = 1
        for l = 0:n-1
            for m_l = -l:l
                for m_s = -0.5:0.5
                    Ψ[idx] = (l, m_l, m_s)
                    idx += 1
                end
            end
        end

        # Save vectors of the quantum numbers
        l = [Ψ[i][1] for i = 1:len_Ψ]
        m_l = [Ψ[i][2] for i = 1:len_Ψ]
        m_s = [Ψ[i][3] for i = 1:len_Ψ]

        new(Ψ, l, m_l, m_s)
    end
end

Ψ = Wavefunction(2)

@doc raw"""
    Lz_Sz_prefactor(Ψ::Wavefunction, λ::Float64)::Vector{ComplexF64}

Calculate the z angular momentum spin operator prefactor

```math
\frac{\lambda \hbar}{2} (m_l \cdot m_s) | \psi_{\ell, m_{\ell}, m_s} \rangle
```
"""
function Lz_Sz_prefactor(Ψ::Wavefunction, λ::Float64)::Vector{ComplexF64}
    ((λ * ħ) / 2) .* (Ψ.m_l .* Ψ.m_s)
end

@doc raw"""
    l_up_s_down_prefactor(ψ::Wavefunction, λ::float64)

Calculate the L_+S_- operator prefactor.

```math
\frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - 3m_{\ell})(s^2 + s - 3m_s) \right]^{\frac{1}{2}} | \psi_{\ell, m_{\ell} + 1, m_s - 1} \rangle
```
"""
# L_up_S_down_prefactor(Ψ::Wavefunction, λ::Float64) = [(λ * ħ^2) / 2 * ((ψ.l^2 + l - (3 * ψ.m_l)) * (0.5^2 + 0.5 - (3 * ψ.m_s)))^0.5 for ψ in Ψ]

function L_up_S_down_prefactor(Ψ::Wavefunction, λ::Float64)
    ((λ * ħ^2) / 2) .*
    ((Ψ.l .^ 2 .+ Ψ.l .- (3 .* Ψ.m_l)) .* (Ψ.m_s .^ 2 .+ Ψ.m_s .- (3 .* Ψ.m_s))) .^ 0.5
end

@doc raw"""
    L_down_S_up_prefactor(Ψ::Wavefunction, λ::Float64)

Calculate the L_-S_+ operator prefactor.

```math
\frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - m_{\ell})(s^2 + s - m_s) \right]^{\frac{1}{2}} | \psi_{\ell, m_{\ell} - 1, m_s + 1} \rangle
```
"""
function L_down_S_up_prefactor(Ψ::Wavefunction, λ::Float64)
    ((λ * ħ^2) / 2) .* ((Ψ.l .^ 2 .+ Ψ.l .- Ψ.m_l) .* (Ψ.m_s .^ 2 .+ Ψ.m_s .- Ψ.m_s)) .^ 0.5
end

@doc raw"""
    Lz_Sz_kron(Ψ::Wavefunction)

Use logical indexing to apply prefactors based on kroenecker delta for the Lz_Sz operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_zS_z | \Psi_{\ell, m_{\ell}, m_s} = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
"""
function Lz_Sz_kron(Ψ::Wavefunction)
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .&& Ψ.m_s .== Ψ.m_s'
end

@doc raw"""
    L_up_S_down_kron(Ψ::Wavefunction)

Use logical indexing to apply prefactors based on kroenecker delta for the L+_S- operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_+S_- | \Psi_{\ell, m_{\ell}+1, m_s-1} = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
"""
function L_up_S_down_kron(Ψ::Wavefunction)
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .+ 1 .&& Ψ.m_s .== Ψ.m_s' .- 1
end

@doc raw"""
    L_up_S_down_kron(Ψ::Wavefunction)

Use logical indexing to apply prefactors based on kroenecker delta for the L-_S+ operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_+S_- | \Psi_{\ell, m_{\ell}-1, m_s+1} = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
"""
function L_down_S_up_kron(Ψ::Wavefunction)
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .- 1 .&& Ψ.m_s .== Ψ.m_s' .+ 1
end

function construct_full_H(n::Int, λ::Float64)
    # Setup Ψ as a vector of all possible wavefunctions
    Ψ = Wavefunction(n)

    # Construct the Hamiltonian matrix
    H = Array{ComplexF64,2}(undef, length(Ψ.Ψ), length(Ψ.Ψ))

    # Compute the Hamiltonian matrix elements based on the prefactors and kroenecker deltas
    t1 = Lz_Sz_prefactor(Ψ, λ) .* Lz_Sz_kron(Ψ)
    t2 = L_up_S_down_prefactor(Ψ, λ) .* L_up_S_down_kron(Ψ)
    t3 = L_down_S_up_prefactor(Ψ, λ) .* L_down_S_up_kron(Ψ)

    # Assign the array elements
    H .= t1 .+ t2 .+ t3

    return H
end

H = construct_full_H(2, 1.0)

end  # module
