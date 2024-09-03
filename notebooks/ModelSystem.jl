### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ e310f764-2645-46f4-a078-a54ea614111b
using DrWatson

# ╔═╡ 807203ea-830b-4d34-a63e-da2f6ad74342
@quickactivate "DeltaSCFSOC"

# ╔═╡ 530505dc-27cc-4152-9403-cd72413efea5
using Org

# ╔═╡ d007d79f-9ebc-49a6-9614-0c3bf3655100
using LinearAlgebra, Plots

# ╔═╡ e24673a0-abb4-4a57-b7e4-e3899be90f9c
org"""
#+title: Model System
#+author: Dylan Morgan
#+email: dylan.morgan@warwick.ac.uk
#+startup: latexpreview inlineimages
#+latex_header: \usepackage{braket}

* Setup
** Imports
"""

# ╔═╡ 3c0078cb-d5f3-49ae-a174-5631d6f2bd73
org"""
** Constants
"""

# ╔═╡ e50ce456-5270-45ef-bc0e-73ed525291f6
const ħ = 1.0545718e-34

# ╔═╡ f37d625a-a5c7-4ddf-bf3d-99cefb01cd82
org"""
** Types
"""

# ╔═╡ 33902afc-2276-4871-9f96-cc9c1d3e619a
struct GreaterThan{T<:Real}
    x::T
    min_value::T

    function greater_than(x::T, min_value::T) where {T<:Real}
        if x < min_value
            throw(DomainError("$x must be >= $min_value"))
        end
        new{T}(x, min_value)
    end
end

# ╔═╡ 3f14d14f-1683-4399-9948-d5ab1b1d630f
struct Between{T<:Real}
    x::T
    min::T
    max::T

    function between(x::T, min::T, max::T) where {T<:Real}
        if x < min || x > max
            throw(DomainError("$x must be in [$min, $max]"))
        end
        new{T}(x, min, max)
    end
end

# ╔═╡ ad8bbce9-d0f6-4f49-9b98-916ad45d62f7
struct BetweenPlusMinus{T<:Real}
    x::T
    origin::T

    function between_plus_minus(x::T, origin::T) where {T<:Real}
        if x < -origin || x > origin
            throw(DomainError("$x must be in [-$origin, $origin]"))
        end
        new{T}(x, origin)
    end
end

# ╔═╡ 99aa7fce-4b42-472d-8e59-d923623433f0
org"""
** Model System
We can set up a spin-orbit coupling Hamiltonian for a model system, where the wavefunction is represented in a basis of the \(\ell\), \(m_{\ell}\), and \(m_s\) quantum numbers for a given \(n\).

#+name: model-H
\begin{equation}
    \Braket{ \Psi_{\ell, m_{\ell}, m_s} | \lambda \hat{L} \cdot \hat{S} | \Psi_{\ell, m_{\ell}, m_s} }
\end{equation}

Where \(\hat{L}\) is the orbital angular momentum operator, \(\hat{S}\) is the spin operator, and \(\lambda\) is the spin-orbit coupling strength, which is dependent on the atomic number of the elements in the solid.
"""

# ╔═╡ 00409db9-3b70-424e-a038-930242e44624
org"""
** Wavefunction
Setup the wavefunction as a ~mutable struct~ so we can enact the ladder operators on the wavefunction object later
"""

# ╔═╡ 71693a2b-e587-4e22-a981-c8756f8d0dbb
mutable struct Wavefunction
    λ::Float64
    n::GreaterThan{Int}
    l::Between{Int}
    m_l::BetweenPlusMinus{Int}
    m_s::Float16

    function wavefunction(
        lambda::Float64,
        principal::Int,
        azimuth::Int,
        magnetic::Int,
        spin::Float16,
    )

        if spin != 0.5 || spin != -0.5
            throw(ArgumentError("Spin must be either 0.5 or -0.5"))
        end

        λ = lambda
        n = greater_than(principal, 0)
        l = between(azimuth, 0, n - 1)
        m_l = between_plus_minus(magnetic_val, azimuth_val)
        m_s = spin
        new(λ, n, l, m_l, m_s)
    end
end

# ╔═╡ e1576863-acbf-463c-8a85-0fdee54a272d
org"""
* Define the Operator
** Uncertainty Principle
The uncertainty principle states that when two observable operators do not commute, they cannot be measured simultaneously, and the more accurately that one is known, the less accurately the other can be known. For angular momentum, this is given by the Robertson-Schrödinger relation

\begin{equation}
    \sigma_{J_x} \sigma_{J_y} \geq \frac{\hbar}{2} | \langle J_z \rangle |
\end{equation}

where \(\sigma_J\) is the standard deviation in the measured values of \(J\). \(J\) can also be replaced by \(L\) or \(S\), and \(x, y, z\) can be rearranged in any order. However it is still possible to measure \(J^2\) and any one component of \(J\). These values are characterised by \(\ell\) and \(m\).
"""

# ╔═╡ dbd615d6-5774-465e-93d0-d90dcec8e482
org"""
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
"""

# ╔═╡ a36712c6-7d85-411e-bf14-8d2f119783c1
org"""
* Solve the Eigenvalue Problem
** Operator(s) on Ket
Define how the Hamiltonian acts on the wavefunction in the ket
"""

# ╔═╡ 360d3912-102c-44ff-9fb9-3b7720192201
# Calculate the z component of the angular momentum and spin operators
L_z_S_z_prefactor(ψ::Wavefunction) = ((ψ.λ * ħ) / 2) * (ψ.m_l * ψ.m_s)

# ╔═╡ 7efcc248-19e5-4783-a232-0a837386264d
org"""
Additionally create \(\Bra{\psi'}\) and enact the operators on \(\Ket{\psi}\). These only need to be defined for \(\hat{L}_+\hat{S}_-\) and \(\hat{L}_-\hat{S}_+\) as \(\hat{L}_z\hat{S}_z\) does not raise or lower the quantum numbers and only serves as a prefactor.

"""

# ╔═╡ ed493dd6-8c19-4816-9b0e-a166d4d286dd
deepcopy_ψ(ψ::Wavefunction) = deepcopy(ψ)

# ╔═╡ 93f5a7a2-2a54-40b3-a91d-5476b025954b
org"""
** Bra on ket
Set \(\hat{H}_{ij}\) in accordance with \(\Braket{ \psi_{\ell', m_{\ell}', m_s'} | \psi_{\ell, m_{\ell}, m_s} } = \delta_{\ell'\ell} \delta_{m_{\ell}'m_{\ell}} \delta_{m_s'm_s}\)

"""

# ╔═╡ 8195168d-fae0-4b0c-b4b0-e3826ed9b10d
org"""
* Setup Full Hamiltonian Matrix
Iterate over all quantum numbers to create \(\hat{H}\) for \(n=2\)
"""

# ╔═╡ 5ffff0bd-66d0-47be-bd1e-7f6f193d07eb
begin
    n = 2
    l = collect(Int8, 0:n-1)
    m_l = collect(Int8, -maximum(l):maximum(l))
    m_s = [-0.5, 0.5]
end

# ╔═╡ a41c7acd-8a8d-4f9c-867f-981bfe5bccbd
# Calculate the L_+S_- ladder operator
L_up_S_down_prefactor(ψ::Wavefunction) =
    (ψ.λ * ħ^2) / 2 * ((ψ.l^2 + l - (3 * ψ.m_l)) * (0.5^2 + 0.5 - (3 * ψ.m_s)))^0.5

# ╔═╡ 8156fb85-6227-4e31-94c9-eb64c99503c4
# Calculate the L_-S_+ ladder operator
L_down_S_up_prefactor(ψ::Wavefunction) =
    (ψ.λ * ħ^2) / 2 * ((ψ.l^2 + l - ψ.m_l) * (0.5^2 + 0.5 - ψ.m_s))^0.5

# ╔═╡ b2f66939-04fb-4856-960b-240708b72a16
function ψOψ(ψ_prime::Wavefunction, ψ::Wavefunction, operator::String)
    if operator == "L_z_S_z"
        if ψ_prime.l == ψ.l && ψ_prime.m_l == ψ.m_l && ψ_prime.s == ψ.s
            return L_z_S_z_prefactor(ψ)
        else
            return 0
        end

    elseif operator == "L_up_S_down"
        if ψ_prime.l == ψ.l && ψ_prime.m_l == ψ.m_l + 1 && ψ_prime.s == ψ.s - 1
            return L_up_S_down_prefactor(ψ)
        else
            return 0
        end

    elseif operator == "L_down_S_up"
        if ψ_prime.l == ψ.l && ψ_prime.m_l == ψ.m_l - 1 && ψ_prime.s == ψ.s + 1
            return 0
        else
            return L_down_S_up_prefactor(ψ)
        end
    else
        throw(
            ArgumentError(
                "Operator must be one of 'L_z_S_z', 'L_up_S_down', 'L_down_S_up'",
            ),
        )
    end
end

# ╔═╡ 73229298-a8b7-495b-9504-d8293ec6b51e
function L_up_S_down_ladder(ψ::Wavefunction)
    # L_+S_- raises m_l and lowers m_s both by 1
    if ψ.m_l > l - 1 && m_s == 0.5
        ψ.m_l += 1
        ψ.m_s -= 1
    end
end

# ╔═╡ db767803-21ca-4e8b-bc6f-21971302d97c
function L_down_S_up_ladder(ψ::Wavefunction)
    # L_-S_+ lowers m_l and raises m_s both by 1
    if ψ.m_l < -l + 1 && m_s == -0.5
        ψ.m_l -= 1
        ψ.m_s += 1
    end
end

# ╔═╡ dc6c288d-3701-4c28-8b29-4cdaddeab1c4
m_l

# ╔═╡ Cell order:
# ╟─530505dc-27cc-4152-9403-cd72413efea5
# ╟─e24673a0-abb4-4a57-b7e4-e3899be90f9c
# ╠═e310f764-2645-46f4-a078-a54ea614111b
# ╠═807203ea-830b-4d34-a63e-da2f6ad74342
# ╠═d007d79f-9ebc-49a6-9614-0c3bf3655100
# ╟─3c0078cb-d5f3-49ae-a174-5631d6f2bd73
# ╠═e50ce456-5270-45ef-bc0e-73ed525291f6
# ╟─f37d625a-a5c7-4ddf-bf3d-99cefb01cd82
# ╠═33902afc-2276-4871-9f96-cc9c1d3e619a
# ╠═3f14d14f-1683-4399-9948-d5ab1b1d630f
# ╠═ad8bbce9-d0f6-4f49-9b98-916ad45d62f7
# ╟─99aa7fce-4b42-472d-8e59-d923623433f0
# ╟─00409db9-3b70-424e-a038-930242e44624
# ╠═71693a2b-e587-4e22-a981-c8756f8d0dbb
# ╟─e1576863-acbf-463c-8a85-0fdee54a272d
# ╟─dbd615d6-5774-465e-93d0-d90dcec8e482
# ╟─a36712c6-7d85-411e-bf14-8d2f119783c1
# ╠═360d3912-102c-44ff-9fb9-3b7720192201
# ╠═a41c7acd-8a8d-4f9c-867f-981bfe5bccbd
# ╠═8156fb85-6227-4e31-94c9-eb64c99503c4
# ╟─7efcc248-19e5-4783-a232-0a837386264d
# ╠═ed493dd6-8c19-4816-9b0e-a166d4d286dd
# ╠═73229298-a8b7-495b-9504-d8293ec6b51e
# ╠═db767803-21ca-4e8b-bc6f-21971302d97c
# ╟─93f5a7a2-2a54-40b3-a91d-5476b025954b
# ╠═b2f66939-04fb-4856-960b-240708b72a16
# ╟─8195168d-fae0-4b0c-b4b0-e3826ed9b10d
# ╠═5ffff0bd-66d0-47be-bd1e-7f6f193d07eb
# ╠═dc6c288d-3701-4c28-8b29-4cdaddeab1c4
