### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 8d8dc664-aa90-4c96-9c0a-901dca016350
using DrWatson

# ╔═╡ 654e561f-7396-4c1b-8239-a3b6450bd145
quickactivate("$(pwd())../")

# ╔═╡ 04c8bebb-b4d2-4d16-8b43-594428bb4c18
using Org

# ╔═╡ fac1f703-6198-4f5c-a8b3-ae10d2a76065
using PlutoUI, LinearAlgebra, Plots, LaTeXStrings

# ╔═╡ 55d85840-180a-461f-b39c-8ba43ce9223f
org"""
#+title: Model System
#+author: Dylan Morgan
#+startup: latexpreview inlineimages
#+latex_header: \usepackage{braket}
#+auto_tangle: t
#+property: header-args:julia :session jl :results silent
#+property: header-args :tangle ../scripts/DeltaSCFSOC.jl
"""

# ╔═╡ d1404e3f-77ec-4ec3-9e6f-dd406c95272b
org"""
* Setup
** Imports
"""

# ╔═╡ 0385debe-6f66-4f69-b1f3-a560456e3240
org"""
** Constants
"""

# ╔═╡ 93199e3a-9593-40c2-99b3-5bb3d77b8505
const ħ = 1.0545718e-34

# ╔═╡ b8414a41-9c50-48e9-a651-647f018b3def
org"""
* Model System
We can set up a spin-orbit coupling Hamiltonian for a model system, where the wavefunction is represented in a basis of the \(\ell\), \(m_{\ell}\), and \(m_s\) quantum numbers for a given \(n\).

#+name: model-H
\begin{equation}
    \Braket{ \Psi_{\ell, m_{\ell}, m_s} | \lambda \hat{L} \cdot \hat{S} | \Psi_{\ell, m_{\ell}, m_s} }
\end{equation}

Where \(\hat{L}\) is the orbital angular momentum operator, \(\hat{S}\) is the spin operator, and \(\lambda\) is the spin-orbit coupling strength, which is dependent on the atomic number of the elements in the solid.

** Wavefunction
Setup the wavefunction as a ~mutable struct~ using the quantum numbers as a basis. This will create a vector where each element is a state defined by the quantum numbers

"""

# ╔═╡ cbd36b1a-4794-43f0-b0dc-21838fa3aa70
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

# ╔═╡ da902439-c07b-425e-8f23-b3daf9f44d81
org"""
Now, initialise the wavefunction when $$n=2$$

"""

# ╔═╡ cc41e84b-c401-4f78-8021-3b6482d2314c
Ψ = Wavefunction(2)

# ╔═╡ a107ebfd-d6df-452f-8640-3f44283bf4ba
org"""
* Define the Operator
** Uncertainty Principle
The uncertainty principle states that when two observable operators do not commute, they cannot be measured simultaneously, and the more accurately that one is known, the less accurately the other can be known. For angular momentum, this is given by the Robertson-Schrödinger relation

\begin{equation}
    \sigma_{J_x} \sigma_{J_y} \geq \frac{\hbar}{2} | \langle J_z \rangle |
\end{equation}

where \(\sigma_J\) is the standard deviation in the measured values of \(J\). \(J\) can also be replaced by \(L\) or \(S\), and \(x, y, z\) can be rearranged in any order. However it is still possible to measure \(J^2\) and any one component of \(J\). These values are characterised by \(\ell\) and \(m\).

** Derivation
In order to calculate [[model]], we need to apply the operators to the ket and work out the prefactors.

#+name: angular-spin-relation
\begin{equation}
    \begin{split}
        \hat{J}^2 &= \left( \hat{L} + \hat{S} \right)^2 \\
        &= \hat{L}^2 + \hat{S}^2 + 2\hat{L} \cdot \hat{S} \\
        &= \hat{L}^2 + \hat{S}^2 + 2\hat{L}_z\hat{S}_z + \hat{L}_+\hat{S}_- + \hat{L}_-\hat{S}_+ \\
    \end{split}
\end{equation}

However, we can neglect the \(\hat{L}^2 + \hat{S}^2\) terms as they are not included in our Hamiltonian in [[model]]. Now, to define how each operator acts on the ket

\begin{equation}
    \begin{split}
        \hat{L}_z \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \hbar m_{\ell} \Ket{ \psi_{\ell, m_{\ell}, m_s} } \\
        \hat{S}_z \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \hbar m_S \Ket{ \psi_{\ell, m_{\ell}, m_s} }
    \end{split}
\end{equation}

\begin{equation}
    \begin{split}
        L_+ \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \left[ (\ell + m_{\ell} + 1)(l - m_{\ell}) \right]^{\frac{1}{2}} \hbar \Ket{ \psi_{\ell, m_{\ell} + 1, m_s} } \\
        L_- \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \left[ (\ell - m_{\ell} + 1)(l + m_{\ell}) \right]^{\frac{1}{2}} \hbar \Ket{ \psi_{\ell, m_{\ell} - 1, m_s} }
    \end{split}
\end{equation}

\begin{equation}
    \begin{split}
        S_+ \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \left[ (s + m_s + 1)(s - m_s) \right]^{\frac{1}{2}} \hbar \Ket{ \psi_{\ell, m_{\ell}, m_s + 1} } \\
        S_- \Ket{ \psi_{\ell, m_{\ell}, m_s} } &= \left[ (s - m_s + 1)(s + m_s) \right]^{\frac{1}{2}} \hbar \Ket{ \psi_{\ell, m_{\ell}, m_s - 1} } \\
    \end{split}
\end{equation}

Then, substituting [[angular-spin-relation]]  into [[model-H]] , and applying \(\lambda (\hat{L} \cdot \hat{S})\) to \(\Ket{ \psi_{\ell, m_{\ell}, m_s} }\):

\begin{equation}
    \implies \lambda (\hat{L} \cdot \hat{S}) \Ket{ \Psi_{\ell, m_{\ell}, m_s} } = \frac{\lambda \hbar}{2}(m_{\ell} \cdot m_s) \Ket{ \psi_{\ell, m_{\ell}, m_s} } + \frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - 3m_{\ell})(s^2 + s - 3m_s) \right]^{\frac{1}{2}} \Ket{ \psi_{\ell, m_{\ell} + 1, m_s - 1} } + \frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - m_{\ell})(s^2 + s - m_s) \right]^{\frac{1}{2}} \Ket{ \psi_{\ell, m_{\ell} - 1, m_s + 1} }
\end{equation}

\begin{equation}
    \implies \Braket{ \Psi_{\ell', m_{\ell}', m_s'} | \lambda (\hat{L} \cdot \hat{S}) | \Psi_{\ell, m_{\ell}, m_s} } = \lambda \Braket{ \psi_{\ell', m_{\ell}', m_s'} | \hat{L}_z \hat{S}_z | \psi_{\ell, m_{\ell}, m_s} } + \lambda \Braket{ \psi_{\ell', m_{\ell}', m_s'} | \hat{L}_+ \hat{S}_- | \psi_{\ell, m_{\ell} + 1, m_s - 1} } + \lambda \Braket{ \psi_{\ell', m_{\ell}', m_s'} | \hat{L}_- \hat{S}_- | \psi_{\ell, m_{\ell} - 1, m_s + 1} }
\end{equation}

where

\begin{equation}
    \Braket{ \psi_{\ell', m_{\ell}', m_s'} | \psi_{\ell, m_{\ell}, m_s} } = \delta_{\ell' \ell} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
\end{equation}

* Setup the Eigenvalue Problem
** Operator(s) on Ket
Define how the Hamiltonian acts on the wavefunction in the ket

"""

# ╔═╡ 80705724-5b4c-4f67-8b10-03c716042036
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

# ╔═╡ 2053655a-1777-4e83-9502-7d00c17de612
@doc raw"""
    l_up_s_down_prefactor(ψ::Wavefunction, λ::float64)::Vector{ComplexF64}

Calculate the L_+S_- operator prefactor.

```math
\frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - 3m_{\ell})(s^2 + s - 3m_s) \right]^{\frac{1}{2}} | \psi_{\ell, m_{\ell} + 1, m_s - 1} \rangle
```
"""
function L_up_S_down_prefactor(Ψ::Wavefunction, λ::Float64)::Vector{ComplexF64}
    ((λ * ħ^2) / 2) .*
    ((Ψ.l .^ 2 .+ Ψ.l .- (3 .* Ψ.m_l)) .* (Ψ.m_s .^ 2 .+ Ψ.m_s .- (3 .* Ψ.m_s))) .^ 0.5
end

# ╔═╡ 2e44cd9b-7ad9-4a24-92e1-48ddf35abc56
@doc raw"""
    L_down_S_up_prefactor(Ψ::Wavefunction, λ::Float64)::Vector{ComplexF64}

Calculate the L_-S_+ operator prefactor.

```math
\frac{\lambda \hbar^2}{2} \left[ (\ell^2 + \ell - m_{\ell})(s^2 + s - m_s) \right]^{\frac{1}{2}} | \psi_{\ell, m_{\ell} - 1, m_s + 1} \rangle
```
"""
function L_down_S_up_prefactor(Ψ::Wavefunction, λ::Float64)::Vector{ComplexF64}
    ((λ * ħ^2) / 2) .* ((Ψ.l .^ 2 .+ Ψ.l .- Ψ.m_l) .* (Ψ.m_s .^ 2 .+ Ψ.m_s .- Ψ.m_s)) .^ 0.5
end

# ╔═╡ 765755cf-76ea-4da4-ace4-c223ef2f23b2
org"""
** Bra on ket
Additionally create \(\Bra{\psi'}\) and enact the operators on \(\Ket{\psi}\) to calculate the inner product

"""

# ╔═╡ 09b3b979-078b-4e67-b6fe-68f5a4d07987
@doc raw"""
    Lz_Sz_kron(Ψ::Wavefunction)::BitMatrix

Use logical indexing to apply prefactors based on the Kronecker delta for the Lz_Sz operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_zS_z | \Psi_{\ell, m_{\ell}, m_s} \rangle = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
```
"""
function Lz_Sz_kron(Ψ::Wavefunction)::BitMatrix
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .&& Ψ.m_s .== Ψ.m_s'
end

# ╔═╡ 6e83fb4b-1830-4f33-b3e7-1587528fe196
@doc raw"""
    L_up_S_down_kron(Ψ::Wavefunction)::BitMatrix

Use logical indexing to apply prefactors based on the Kronecker delta for the L+_S- operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_+S_- | \Psi_{\ell, m_{\ell}+1, m_s-1} \rangle = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
```
"""
function L_up_S_down_kron(Ψ::Wavefunction)::BitMatrix
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .+ 1 .&& Ψ.m_s .== Ψ.m_s' .- 1
end

# ╔═╡ dd132e66-1759-466a-bde8-d534cdccc091
@doc raw"""
    L_up_S_down_kron(Ψ::Wavefunction)::BitMatrix

Use logical indexing to apply prefactors based on the Kronecker delta for the L-_S+ operator.

```math
\langle \Psi_{\ell', m_{\ell}', m_s'} | L_+S_- | \Psi_{\ell, m_{\ell}-1, m_s+1} \rangle = \delta_{l' l} \delta_{m_{\ell}' m_{\ell}} \delta_{m_s' m_s}
```
"""
function L_down_S_up_kron(Ψ::Wavefunction)::BitMatrix
    Ψ.l .== Ψ.l' .&& Ψ.m_l .== Ψ.m_l' .- 1 .&& Ψ.m_s .== Ψ.m_s' .+ 1
end

# ╔═╡ f5787c0f-bbe0-4326-ac4b-a268e92c1c25
org"""
* Solve the Eigenvalue Problem
** Setup the Hamiltonian
Iterate over all quantum numbers to create \(\hat{H}\) for \(n=2\)

"""

# ╔═╡ 7252392f-d6ac-4af9-86a0-4032b0cf4b69
"""
    construct_full_H(Ψ::Wavefunction, λ::Float64)::Matrix{ComplexF64}

Construct the full Hamiltonian matrix for the given wavefunction and λ
"""
function construct_full_H(Ψ::Wavefunction, λ::Float64)::Matrix{ComplexF64}
    # Construct the Hamiltonian matrix
    H = Matrix{ComplexF64}(undef, length(Ψ.Ψ), length(Ψ.Ψ))

    # Compute the Hamiltonian matrix elements based on the prefactors and kroenecker deltas
    t1 = Lz_Sz_prefactor(Ψ, λ) .* Lz_Sz_kron(Ψ)
    t2 = L_up_S_down_prefactor(Ψ, λ) .* L_up_S_down_kron(Ψ)
    t3 = L_down_S_up_prefactor(Ψ, λ) .* L_down_S_up_kron(Ψ)

    # Assign the array elements
    H .= t1 .+ t2 .+ t3
end

# ╔═╡ 87f10aaa-1bf4-4955-88a8-508f11538ecd
H = construct_full_H(Ψ, 1.0)

# ╔═╡ 03c62d7f-6e3a-463d-afbe-03765480b1a0
org"""
** Diagonalise the Hamiltonian
"""

# ╔═╡ cb435008-00ae-4de7-9dfb-a0a8498e52fb
function diagonalise(M::Matrix{ComplexF64})::Diagonal{ComplexF64, Vector{ComplexF64}}
    # Find the eigenvalues and eigenvectors of the Hamiltonian
    E = eigen(M)

    # Get the diagonal matrix of eigenvalues
    Diagonal(E.values)

    # Check that the diagonal Hamiltonian is within numerical error of D
    # P = eigen.vectors
    # @assert norm(H - (P * D * inv(P))) < 1e-10
end

# ╔═╡ 71603122-6a65-444b-a2dd-2aa4d8e9098d
diagonalise(H)

# ╔═╡ 9ce5e80a-9ed5-4e1e-984d-d8ac8d29d771
org"""
* Varying the SOC Constant
Calculate and diagonalise the Hamilton whilst varying \(\lambda\)

"""

# ╔═╡ 2f8f0adf-17b2-4aad-863d-b24ce9067ee2
@bind SOC_factor Slider(1:10, default=5, show_value=true)

# ╔═╡ 4ed7375a-d8a9-4755-bff0-fa692e8e4b6f
Λ = collect(0:(SOC_factor/10)/ħ:SOC_factor/ħ)

# ╔═╡ 01162634-6dc7-45c4-aca7-a6c675a89b86
e_vals = Matrix{ComplexF64}(undef, length(Ψ.Ψ), length(Λ))

# ╔═╡ 021db029-5b31-4097-bfb0-6756d929a34c
for λ in eachindex(Λ)
    H = construct_full_H(Ψ, Λ[λ])
    H_d = diagonalise(H)
    e_vals[:, λ] .= H_d.diag
end

# ╔═╡ ca872726-8556-4140-a6e7-8772ab575cf3
e_vals

# ╔═╡ 324873f2-dad3-438f-87e6-ce560b02aacc
org"""
* Plotting
Plot the eigenvalues for various values of \lambda

"""

# ╔═╡ 699c1e43-c19b-4011-b8cb-8a36608017d8
org"""
** Real Eigenvalues
"""

# ╔═╡ be6615fe-28db-4f81-b9ee-3ea1878793f8
begin
	#scatter3d(real(e_vals[1, :]), imag(e_vals[1,:]), Λ)
	
	# Start by plotting just the first eigenvalues (ie. for 1s orbitals)
	real_e_vals = plot(Λ, real(e_vals[1, :]), label=L"$\ell=0$ $m_{\ell}=0$, $m_s=\downarrow$", xlabel=L"λ", markershape=:auto)

	# Then plot the others
	for i in 2:length(e_vals[:, 1])
		Ψ_l_i = real(Ψ.l[i])
		Ψ_m_l_i = real(Ψ.m_l[i])
		Ψ_m_s_i = real(Ψ.m_s[i]) == 0.5 ? "\\uparrow" : "\\downarrow"
		lab = "\$\\ell=$Ψ_l_i\$, \$m_{\\ell}=$Ψ_m_l_i\$, \$m_s=$Ψ_m_s_i\$"
		plot!(real_e_vals, Λ, real(e_vals[i, :]), label=lab, markershape=:auto)
	end
	real_e_vals
end

# ╔═╡ 650959a9-06d5-4673-bdac-65a15e31c2d9
org"""
** Imaginary Eigenvalues
"""

# ╔═╡ 4bc840f2-d95e-4ee3-9c37-d95ebaee444c
begin	
	imag_e_vals = plot(Λ, imag(e_vals[1, :]), label=L"$\ell=0$ $m_{\ell}=0$, $m_s=\downarrow$", xlabel=L"λ", markershape=:auto)

	for i in 2:length(e_vals[:, 1])
		Ψ_l_i = real(Ψ.l[i])
		Ψ_m_l_i = real(Ψ.m_l[i])
		Ψ_m_s_i = real(Ψ.m_s[i]) == 0.5 ? "\\uparrow" : "\\downarrow"
		lab = "\$\\ell=$Ψ_l_i\$, \$m_{\\ell}=$Ψ_m_l_i\$, \$m_s=$Ψ_m_s_i\$"
		plot!(imag_e_vals, Λ, imag(e_vals[i, :]), label=lab, markershape=:auto)
	end
	imag_e_vals
end

# ╔═╡ 4fa9d932-ed70-4bbc-8214-584f935a113b
org"""
* Real and Imaginary
"""

# ╔═╡ 3f8a9c38-878e-4706-8c7e-f733feb72475
begin
	all_e_vals = plot3d(imag(e_vals[1, :]), real(e_vals[1, :]), Λ, label=L"$\ell=0$ $m_{\ell}=0$, $m_s=\downarrow$", xlabel="imaginary", ylabel="real", zlabel=L"λ", markershape=:auto)

	for i in 2:length(e_vals[:, 1])
		Ψ_l_i = real(Ψ.l[i])
		Ψ_m_l_i = real(Ψ.m_l[i])
		Ψ_m_s_i = real(Ψ.m_s[i]) == 0.5 ? "\\uparrow" : "\\downarrow"
		lab = "\$\\ell=$Ψ_l_i\$, \$m_{\\ell}=$Ψ_m_l_i\$, \$m_s=$Ψ_m_s_i\$"
		plot3d!(all_e_vals, imag(e_vals[i, :]), real(e_vals[i, :]), Λ, label=lab, markershape=:auto)
	end 
	all_e_vals
end

# ╔═╡ Cell order:
# ╟─04c8bebb-b4d2-4d16-8b43-594428bb4c18
# ╟─55d85840-180a-461f-b39c-8ba43ce9223f
# ╟─d1404e3f-77ec-4ec3-9e6f-dd406c95272b
# ╠═8d8dc664-aa90-4c96-9c0a-901dca016350
# ╠═654e561f-7396-4c1b-8239-a3b6450bd145
# ╠═fac1f703-6198-4f5c-a8b3-ae10d2a76065
# ╟─0385debe-6f66-4f69-b1f3-a560456e3240
# ╠═93199e3a-9593-40c2-99b3-5bb3d77b8505
# ╟─b8414a41-9c50-48e9-a651-647f018b3def
# ╠═cbd36b1a-4794-43f0-b0dc-21838fa3aa70
# ╟─da902439-c07b-425e-8f23-b3daf9f44d81
# ╠═cc41e84b-c401-4f78-8021-3b6482d2314c
# ╟─a107ebfd-d6df-452f-8640-3f44283bf4ba
# ╠═80705724-5b4c-4f67-8b10-03c716042036
# ╠═2053655a-1777-4e83-9502-7d00c17de612
# ╠═2e44cd9b-7ad9-4a24-92e1-48ddf35abc56
# ╟─765755cf-76ea-4da4-ace4-c223ef2f23b2
# ╠═09b3b979-078b-4e67-b6fe-68f5a4d07987
# ╠═6e83fb4b-1830-4f33-b3e7-1587528fe196
# ╠═dd132e66-1759-466a-bde8-d534cdccc091
# ╟─f5787c0f-bbe0-4326-ac4b-a268e92c1c25
# ╠═7252392f-d6ac-4af9-86a0-4032b0cf4b69
# ╠═87f10aaa-1bf4-4955-88a8-508f11538ecd
# ╟─03c62d7f-6e3a-463d-afbe-03765480b1a0
# ╠═cb435008-00ae-4de7-9dfb-a0a8498e52fb
# ╠═71603122-6a65-444b-a2dd-2aa4d8e9098d
# ╟─9ce5e80a-9ed5-4e1e-984d-d8ac8d29d771
# ╟─2f8f0adf-17b2-4aad-863d-b24ce9067ee2
# ╠═4ed7375a-d8a9-4755-bff0-fa692e8e4b6f
# ╠═01162634-6dc7-45c4-aca7-a6c675a89b86
# ╠═021db029-5b31-4097-bfb0-6756d929a34c
# ╠═ca872726-8556-4140-a6e7-8772ab575cf3
# ╟─324873f2-dad3-438f-87e6-ce560b02aacc
# ╟─699c1e43-c19b-4011-b8cb-8a36608017d8
# ╠═be6615fe-28db-4f81-b9ee-3ea1878793f8
# ╟─650959a9-06d5-4673-bdac-65a15e31c2d9
# ╠═4bc840f2-d95e-4ee3-9c37-d95ebaee444c
# ╟─4fa9d932-ed70-4bbc-8214-584f935a113b
# ╠═3f8a9c38-878e-4706-8c7e-f733feb72475
