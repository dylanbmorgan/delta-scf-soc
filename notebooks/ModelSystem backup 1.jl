### A Pluto.jl notebook ###
# Generated by ox-pluto for Pluto v0.18+
# Created 2024-09-03 Tue 15:33

using InteractiveUtils

# ╔═╡ e117469c-f481-406f-9bed-50f7edd35fd1
using Org

# ╔═╡ e24673a0-abb4-4a57-b7e4-e3899be90f9c
org"""
#+title: Model System
#+author: Dylan Morgan
#+startup: latexpreview inlineimages
#+latex_header: \usepackage{braket}
#+property: header-args :async yes
#+property: header-args:bash :session sh
#+property: header-args:julia :session jl :results silent

## Setup
### Imports
"""

# ╔═╡ e310f764-2645-46f4-a078-a54ea614111b
using DrWatson

# ╔═╡ 807203ea-830b-4d34-a63e-da2f6ad74342
@quickactivate "DeltaSCFSOC"

# ╔═╡ d007d79f-9ebc-49a6-9614-0c3bf3655100
using LinearAlgebra, Plots

# ╔═╡ 3c0078cb-d5f3-49ae-a174-5631d6f2bd73
org"""
### Constants
"""

# ╔═╡ e50ce456-5270-45ef-bc0e-73ed525291f6
const ħ = 1.0545718e-34

# ╔═╡ f37d625a-a5c7-4ddf-bf3d-99cefb01cd82
org"""
### Types
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

# ╔═╡ 57aa7e54-99bf-49b7-8560-36190dadfcc6
org"""
## Model System
We can set up a spin-orbit coupling Hamiltonian for a model system, where the wavefunction is represented in a basis of the \(\ell\), \(m_{\ell}\), and \(m_s\) quantum numbers for a given \(n\).

#+name: model-H
\begin{equation}
    \Braket{ \Psi_{\ell, m_{\ell}, m_s} | \lambda \hat{L} \cdot \hat{S} | \Psi_{\ell, m_{\ell}, m_s} }
\end{equation}

Where \(\hat{L}\) is the orbital angular momentum operator, \(\hat{S}\) is the spin operator, and \(\lambda\) is the spin-orbit coupling strength, which is dependent on the atomic number of the elements in the solid.

### Wavefunction
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

# ╔═╡ 916d8c9f-9c63-4d6d-910c-c6f3939b81c4
org"""
## Define the Operator
### Uncertainty Principle
The uncertainty principle states that when two observable operators do not commute, they cannot be measured simultaneously, and the more accurately that one is known, the less accurately the other can be known. For angular momentum, this is given by the Robertson-Schrödinger relation

\begin{equation}
    \sigma_{J_x} \sigma_{J_y} \geq \frac{\hbar}{2} | \langle J_z \rangle |
\end{equation}

where \(\sigma_J\) is the standard deviation in the measured values of \(J\). \(J\) can also be replaced by \(L\) or \(S\), and \(x, y, z\) can be rearranged in any order. However it is still possible to measure \(J^2\) and any one component of \(J\). These values are characterised by \(\ell\) and \(m\).

### Derivation
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

Then, substituting [[angular-spin-relation]] into [[model-H]], and applying \(\lambda (\hat{L} \cdot \hat{S})\) to \(\Ket{ \psi_{\ell, m_{\ell}, m_s} }\):

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

## Solve the Eigenvalue Problem
### Operator(s) on Ket
Define how the Hamiltonian acts on the wavefunction in the ket

"""

# ╔═╡ 360d3912-102c-44ff-9fb9-3b7720192201
# Calculate the z component of the angular momentum and spin operators
L_z_S_z_prefactor(ψ::Wavefunction) = ((ψ.λ * ħ) / 2) * (ψ.m_l * ψ.m_s)

# ╔═╡ a41c7acd-8a8d-4f9c-867f-981bfe5bccbd
# Calculate the L_+S_- ladder operator
L_up_S_down_prefactor(ψ:Wavefunction) =
    (ψ.λ * ħ^2) / 2 * ((ψ.l^2 + l - (3 * ψ.m_l)) * (0.5^2 + 0.5 - (3 * ψ.m_s)))^0.5

# ╔═╡ 8156fb85-6227-4e31-94c9-eb64c99503c4
# Calculate the L_-S_+ ladder operator
L_down_S_up_prefactor(ψ::Wavefunction) =
    (ψ.λ * ħ^2) / 2 * ((ψ.l^2 + l - ψ.m_l) * (0.5^2 + 0.5 - ψ.m_s))^0.5

# ╔═╡ 7efcc248-19e5-4783-a232-0a837386264d
org"""
Additionally create \(\Bra{\psi'}\) and enact the operators on \(\Ket{\psi}\). These only need to be defined for \(\hat{L}_+\hat{S}_-\) and \(\hat{L}_-\hat{S}_+\) as \(\hat{L}_z\hat{S}_z\) does not raise or lower the quantum numbers and only serves as a prefactor.

"""

# ╔═╡ ed493dd6-8c19-4816-9b0e-a166d4d286dd
deepcopy_ψ(ψ::Wavefunction) = deepcopy(ψ)

# ╔═╡ 73229298-a8b7-495b-9504-d8293ec6b51e
function L_up_S_down_ladder(ψ:Wavefunction)
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

# ╔═╡ 93f5a7a2-2a54-40b3-a91d-5476b025954b
org"""
### Bra on ket
Set \(\hat{H}_{ij}\) in accordance with \(\Braket{ \psi_{\ell', m_{\ell}', m_s'} | \psi_{\ell, m_{\ell}, m_s} } = \delta_{\ell'\ell} \delta_{m_{\ell}'m_{\ell}} \delta_{m_s'm_s}\)

"""

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

# ╔═╡ 8195168d-fae0-4b0c-b4b0-e3826ed9b10d
org"""
## Setup Full Hamiltonian Matrix
Iterate over all quantum numbers to create \(\hat{H}\) for \(n=2\)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """

"""[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"


# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """

"""# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.4"
manifest_format = "2.0"
project_hash = "ac1187e548c6ab173ac57d4e72da1620216bce54"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"


# ╔═╡ Cell order:
# ╟─e117469c-f481-406f-9bed-50f7edd35fd1
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
# ╟─57aa7e54-99bf-49b7-8560-36190dadfcc6
# ╠═71693a2b-e587-4e22-a981-c8756f8d0dbb
# ╟─916d8c9f-9c63-4d6d-910c-c6f3939b81c4
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
# ╠═00000000-0000-0000-0000-000000000001
# ╠═00000000-0000-0000-0000-000000000002