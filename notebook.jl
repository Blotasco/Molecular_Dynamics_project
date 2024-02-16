### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 4cf0ab18-6647-11eb-2806-87ebde056fd0
begin
	using LinearAlgebra
	using Statistics: mean
	using ForwardDiff: derivative
	using Combinatorics: combinations
	using Formatting: format
	using Plots
	using PlutoUI
	using LaTeXStrings
	
	plotlyjs() # Use the plotlyjs backend
	
	default(framestyle=:zerolines)
end;

# ╔═╡ 9d5f2ed2-6179-11eb-3069-a98bc0c0e21d
md"""
# Molecular Dynamics Project
"""

# ╔═╡ c2abace0-6179-11eb-3d59-1faf66ea4e87
md"""
## Part 1
"""

# ╔═╡ 6eeff9d0-6179-11eb-06a2-91e862d14160
md"""
##### 1a) Understanding the potential
"""

# ╔═╡ 005bc24e-617a-11eb-0fec-1b98ef100475
md"""
 **i)** We plot
"""

# ╔═╡ 7186dbe0-617a-11eb-0a28-83e623ecc625
md"""
 **ii)**
"""

# ╔═╡ 7b8091e0-617a-11eb-0a66-63d1ccd63b7e
md"""

When r is smaller than σ, the first fraction, $\left(\frac{σ}{r}\right)^{12}$, dominates and we get a positive value. When r is larger than σ, both of the fractions become smaller than 1, But the second fraction dominates in this case as it has larger abosolute value. This means we get a negative value, and as r approaches infinity we approach 0 (from below), because both fractions approach 0, although the first one reaches 0 much faster.
"""

# ╔═╡ 169e1570-617c-11eb-3de8-1560387550ee
md"""
 **iii)**
"""

# ╔═╡ 28615b00-617c-11eb-00cc-65e571674f20
md"""
Clearly there is an equilibrium point at the local minimum of $$U$$ at $$r\approx 1.12$$, we also see that the force indeed is zero at this point. There is also a second equilibrium point infinitely far away.
"""

# ╔═╡ 05bf6e10-617d-11eb-1c47-11ecb389b8f1
md"""
 **iv)**
"""

# ╔═╡ 0f1a9a20-617d-11eb-3555-1111efddf3f4
md"""
We can look at the force-graph to see that when the distance is small, the atoms repel each other; the turning point is around 1.12. Therefore, at r = 0.95σ the atoms will repulse eachother because they are unfavorably close. Meanwhile, at r = 1.5σ the atoms will attract eachother. It is important to note that at larger distances the force between them approaches 0.
"""
#What does it mean that potential is negative or positive in relation to kinetic energy?

# ╔═╡ 9650adb2-6180-11eb-3b35-45cfcc098604
md"""
 **v)** Something like electromagnetic force.
"""

# ╔═╡ 6e23014e-6183-11eb-2b24-b72dd880d0d6
md"""
### 1b)
"""

# ╔═╡ d87ec9d0-6183-11eb-39f1-eb1bff077ad9
md"""
**i)** 
The derivative of potential energy with respect to distance is the force. Deriving 
"""

#? Do we need to show all utregning or the main idea?

# ╔═╡ a565d0f0-6185-11eb-1bfd-b14fb9eaf444
md"""
**ii)** 
A lot of math
"""

# ╔═╡ 34b3ac60-618f-11eb-0f14-a1558af74e0e
md"""
### 1c)
"""

# ╔═╡ 80c6c560-618f-11eb-11bb-2de11b2f4fd4
md"""
**i)** 
A LOT of math
"""

# ╔═╡ 8b110c62-618f-11eb-24c4-ef8c7745ab8d
md"""
**ii)** 
"""

# ╔═╡ 0c9fb470-6190-11eb-1f50-eb62baa85055
τ = 3.405 * sqrt(39.95 / (1.0318e-2)) # [sqrt(Å² u/eV) = s]

# ╔═╡ 00e759aa-6300-11eb-331a-ff43a6ab61a2
md"""
## Part 2
"""

# ╔═╡ 7cd3e800-7b72-11eb-3698-274962748383
U(r) = 4(r^(-12) - r^(-6))

# ╔═╡ 682a9e60-617a-11eb-2d0d-4108c271adde
let
	σ = 1
	ϵ = 1
	
	U(r) = 4ϵ*((σ/r)^12 - (σ/r)^6)
	
	r = range(0.9, 3.0, length=1000)
	pU = plot(r, U, xlims=(0.8, 3), lw=2,
		color = "red", title = "LJ Potential", 
		label = "U(r)", xlabel = "r",  ylabel ="J"
	)
	
	F(r) = -derivative(U, r)
	
	pF = plot(r, F, xlims=(0.8, 3), lw=2,
		color = "red", title = "Force in r", 
		label = "F(r)", xlabel = "r",  ylabel ="N", 
	)
	
	zp = r[argmin(U.(r))] # F is zero at this point
	vline!(pF, [zp], color=:black, linestyle=:dot, lw=2)
	
	plot(pU, pF, layout=(2,1))
end

# ╔═╡ 11b5b4a2-6300-11eb-2771-3de463ce5388
md"""
### 2a) Implementation
"""

# ╔═╡ 4147c3fe-6300-11eb-3c06-2bc5c32ccbf5
md"""
**i)** Bla bla
"""

# ╔═╡ 97f5b360-6be2-11eb-3d29-bd8de9b3421d
"force: Akselerasjon (som i dimensjonsløse likninger er det samme som kraften) som  en funksjon av posisjon"
force(r) = 24 * (2*abs(r)^(-12) - abs(r)^(-6)) / abs(r)  

# ╔═╡ e4488500-6302-11eb-2c7b-93a0a0eb0d55
function euler1d(R₁, tmax, n)
	R = zeros(n, 2)
	V = zeros(n, 2)
	t = range(0, tmax, length=n)
	
	Δt = tmax/n
	
	R[1,:] = R₁
	
	for k in 1:(n-1)
		r = abs(R[k,1]-R[k,2])
		
		V[k+1,:] = V[k,:] + Δt * [-1, 1]force(r)
		R[k+1,:] = R[k,:] + Δt * V[k,:]
	end
	
	return (t, R, V)
end

# ╔═╡ 19c52996-6307-11eb-2a43-41270dbf5f00
function eulerCromer1d(R₁, tmax, n)
	R = zeros(n, 2)
	V = zeros(n, 2)
	
	t = range(0, tmax, length=n)
	
	Δt = tmax/n
	
	R[1,:] = R₁
	
	for k in 1:(n-1)
		r = abs(R[k,1]-R[k,2])
		V[k+1,:] = V[k,:] + Δt * [-1, 1]force(r)
		R[k+1,:] = R[k,:] + Δt * V[k+1,:]
	end
	
	return (t, R, V)
end

# ╔═╡ 12e77d24-6314-11eb-32c9-31675f6abc60
function velocityVerlet1d(R₁, tmax, n)
	R = zeros(n, 2)
	V = zeros(n, 2)
	t = range(0, tmax, length=n)
	
	Δt = tmax/n
	
	R[1,:] = R₁
	r = abs(R₁[1] - R₁[2])
	fp = force(r)
	
	for k in 1:(n-1)
		fk = fp
		R[k+1,:] = R[k,:] + V[k,:]*Δt + 1/2*[-1,1]fk*Δt^2
		r = abs(R[k+1, 1] - R[k+1, 2])
		fp = force(r)
		V[k+1,:] = V[k,:] + ((fk + fp)*[-1, 1])*Δt/2
	end
	
	return (t, R, V)
end

# ╔═╡ 67c89960-6579-11eb-390d-b9c51f16786a
function solve1d(R₁, Δt, tmax, integrator)
	t, R, V = integrator(R₁, tmax, Int(floor(tmax/Δt)))
	A = [-force.(R[:, 1]), force.(R[:, 2])]

	return (t, R, V, A)
end

# ╔═╡ 6f74f634-6be0-11eb-39e2-8bc1e64acff7
md"""
### 2b) Motion
"""

# ╔═╡ 8bd16432-6be0-11eb-0526-05fa4130cf8f
md"""
**i** & **ii**) Bla bla
"""

# ╔═╡ 89908eb4-6bd6-11eb-16b0-eb8889e905ab
function plot_1d_solution_movement(t, R, V, A)
	pr = plot(t, [eachcol(R)...], ylabel = "r(t)", color = "green")
	
	pv = plot(t,
		[eachcol(V)...], ylabel = "v(t)", 
		color = ["black" "blue"], linestyle = [:solid :solid]
	)
	
	pa = plot(t,
		[eachcol(A)...], ylabel = "a(t)", xlabel="t",
		color = ["black" "red"], linestyle = [:solid :solid]
	)

	p = plot(pr, pv, pa, layout=(3,1), legend=false)
end

# ╔═╡ 5a9bbf56-6be1-11eb-1cd0-f7657e42c215
function plot_distance_1d(t, R)
	r = abs.(R[:, 1] - R[:, 2])
	plot(t, r, xlabel="t", ylabel="distance", legend=false)
end

# ╔═╡ e60d0a02-6be8-11eb-19f2-7b4cd81d11d3
plot_1d_solution_movement(solve1d([0 1.5], 0.001, 5, velocityVerlet1d)...)

# ╔═╡ 9c8128ac-6be1-11eb-12f0-0b809f394278
let
	t, R, _, _ = solve1d([0 1.5], 0.01, 5, velocityVerlet1d)
	p = plot_distance_1d(t, R)
	plot(p, title="r₀ = 1.5, Euler-Cromer")
end

# ╔═╡ 09a4454a-6be7-11eb-2c70-3d0b2e443bf6
md"""
**iii**) Fits perfectly fine, though we had not predicted the periods.
"""

# ╔═╡ 5115cd0a-6bd7-11eb-2b8b-113019761ec5
function plot_case(R₁, Δt, tmax, integrator)
	t, R, V, A = solve1d(R₁, Δt, tmax, integrator)
	
	plot_distance_1d(t, R)
end

# ╔═╡ 021a639a-6bd8-11eb-3efc-d9349695af85
plot_case([0 0.95], 0.01, 5, eulerCromer1d)

# ╔═╡ ee22725e-6bd9-11eb-2334-23e1cdec1ea1
md"""
Energy is always conserved in a closed system. As we see in plot, the energy is indeed  constant.
"""

# ╔═╡ 2d7e7604-6313-11eb-04c5-b15da7afef17
md"""
## Part 3
"""

# ╔═╡ 992bc6cc-63fc-11eb-2f97-c93fea5210b3
md"""
$$\frac{\mathrm d^2\vec r_i^\prime}{\mathrm dt^{\prime 2}} = 24 \sum_{j\neq i} \left(2||\vec r_i^\prime -\vec r_j^\prime||^{-12} - ||\vec r_i^\prime -\vec r_j^\prime||^{-6}\right) \frac{\vec r_i^\prime -\vec r_j^\prime}{||\vec r_i^\prime -\vec r_j^\prime||^{2}}$$
"""

# ╔═╡ 592b8212-833a-11eb-3685-a11fe43f9d6f
md"""
### 3a) Implementation
"""

# ╔═╡ be8c8b24-833a-11eb-1638-73dd6c5bc19f
md"""
**i**) Skriver en naiv Velocity-Verlet-funksjon
"""

# ╔═╡ e66a8a6c-831f-11eb-32ee-e12706005c1f
"""
`R₁` er en (N,3)-array med startposisjonene til N atomer
`V₁` er en (N,3)-array med startshastighetene til N atomer
"""
function velocityVerletNaive(R₁, V₁, Δt, tmax)
	N = size(R₁, 1) # nr. of atoms
	n = Int(floor(tmax/Δt))
	
	R = zeros(n, N, 3)
	V = zeros(n, N, 3)
	
	t = range(0, tmax-Δt, step=Δt)
	
	R[1,:,:] = R₁
	V[1,:,:] = V₁
	
	function force(k, i)
		ΣF = [0, 0, 0]
		
		for j=1:N
			(i == j) && continue
			
			r⃗ = R[k, i, :] - R[k, j, :]
			r = norm(r⃗)
			
			ΣF += (2r^-14 - r^-8)*r⃗
		end
		
		24*ΣF
	end
	
	ΣFₚ  = [force(1, i)[d] for i=1:N, d=1:3] # shape (N,3)
	
	for k=1:n-1 # timestep
		R[k+1, :, :] = R[k, :, :] + V[k, :, :]Δt + 1/2*ΣFₚ*Δt^2
	
		ΣFₙ = [force(k+1, i)[d] for i=1:N, d=1:3]
		
		V[k+1, :, :] = V[k, :, :] + 1/2*(ΣFₚ + ΣFₙ)Δt
		
		ΣFₚ = ΣFₙ
	end
	
	return t, R, V
end

# ╔═╡ c8c963fa-833a-11eb-0d2b-3b60b56ffaa6
md"""
**ii**) Writing a better Velocity-Verlet function using Newton's third law.
"""

# ╔═╡ fbc231a4-85c1-11eb-0af5-6987101d71bc
"R is a N×3 array containg all current positions"
function calc_forces(R)
	N = size(R,1)
	ΣF = zeros(N,3)
	T = zeros(N,N,3) # table

	for (i,j) ∈ combinations(1:N,2)
		r⃗ = R[i, :] - R[j, :]
		r = norm(r⃗)

		F = if r < 3 						 # 3c)
			24*(2r^-14 - r^-8)*r⃗
		else
			[0, 0, 0]
		end

		# to calculate in each direction only once

		T[i, j, :] = F
		T[j, i, :] = -F
	end

	for i=1:N
		ΣF[i, :] = sum(T[i, :, :], dims=1)
	end

	ΣF
end

# ╔═╡ 69672a0a-8673-11eb-1537-c79abb78c5df
let
	r = range(0.9, 5, length=100)
	
	F(r) = ifelse(r<3, 24*(2r^-14 - r^-8), 0)
	
	plot(r, F)
end

# ╔═╡ db18f48a-8321-11eb-306d-8db667b77a9d
function velocityVerlet(R₁, V₁, Δt, tmax)
	N = size(R₁, 1) # nr. of atoms
	n = Int(floor(tmax/Δt))
	
	R = zeros(n, N, 3)
	V = zeros(n, N, 3)
	
	t = range(0, tmax-Δt, step=Δt)
	
	R[1,:,:] = R₁
	V[1,:,:] = V₁
	
	ΣFₚ = calc_forces(R₁)
	
	for k=1:n-1 # timestep
		R[k+1,:,:] = R[k,:,:] + V[k, :, :]Δt + 1/2*ΣFₚ*Δt^2
	
		ΣF = calc_forces(R[k+1,:,:])
		
		V[k+1,:,:] = V[k,:,:] + 1/2*(ΣFₚ + ΣF)Δt

		ΣFₚ = ΣF
	end
	
	return t, R, V
end

# ╔═╡ fd37ead6-6414-11eb-0196-c19d1a71a2ed
function writeOvito(R, fn)
	# R is of shape (n, N, 3), N: #atoms, n: #time-steps
	n, N, _ = size(R)
	
	open(fn, "w") do io
    	for i=1:n
			write(io, string(N), "\n")
			write(io, "timestep: $i\n")
			
			for j=1:N
				write(io, "Ar ")
				join(io, string.(R[i, j, :]), " ")
				write(io, "\n")
			end
		end
	end;
end

# ╔═╡ 977353f4-8350-11eb-1b77-53e68418f307
md"""
**TODO**: The text between iii and iv makes no sense. (??)
"""

# ╔═╡ b984168e-8352-11eb-0a3c-d7f1ed00a4aa
Uₐ(r) = ifelse(r < 3, U(r) - U(3), 0)

# ╔═╡ 5531984e-8674-11eb-177a-1921f5a7cc92
plot(2.5:0.001:3.5, [Uₐ, U], label=["Uₐ" "U"], framestyle=:auto)

# ╔═╡ a82a8900-85c2-11eb-0d21-1f18db14b68e
"R is a N×3 array containg all current positions.
 Returns an N-vector contain"
function calc_potentials(R)
	N = size(R,1)
	U = zeros(N)
	T = zeros(N,N) # table

	for (i,j) ∈ combinations(1:N, 2)
		r = norm(R[i, :] - R[j, :])

		T[i, j] = T[j, i] = Uₐ(r)
	end

	for i=1:N
		U[i] = sum(T[i, :])
	end

	U
end

# ╔═╡ f07ed7c8-8352-11eb-3498-3b6e0d0dc700
md"""
**v**) Yes, but very slightly.
"""

# ╔═╡ ee4e444a-8350-11eb-2af2-8d80fb265f2b
md"""
### 3b) Verification
"""

# ╔═╡ 44b992b4-8351-11eb-1c37-1776cd088551
md"""
**i**) 
"""

# ╔═╡ 3ee7809a-8351-11eb-1571-150953d80e06
let 
	t, R, V = velocityVerlet([1.5 0 0; 0 0 0], zeros(2,3), 0.01, 5)
	x1, x2 = eachcol(R[:, :, 1])
	plot(t, [x1, x2], label=["x1" "x2"])
end

# ╔═╡ c81f8814-8351-11eb-2b31-f3d64a06edbb
let 
	t, R, V = velocityVerlet([0.95 0 0; 0 0 0], zeros(2,3), 0.01, 5)
	x1, x2 = eachcol(R[:, :, 1])
	plot(t, [x1, x2], label=["x1" "x2"])
end

# ╔═╡ f95d6e5a-8351-11eb-0b50-d7c5040098b4
md"""
**ii**)
"""

# ╔═╡ 06b9ab88-8352-11eb-19bb-3726a315ee54
let
	R₁ = [
		1  0 0
		0  1 0
	   -1  0 0
		0 -1 0
	]
	
	t, R, V = velocityVerlet(R₁, zeros(4, 3), 0.01, 5)
	
	writeOvito(R, "3biii.txt")
end

# ╔═╡ 813e5028-8352-11eb-3384-1398b14cf967
md"""
**iii**) We see the motion is periodic, similar to earlier results.
"""

# ╔═╡ 2f1da08a-85bb-11eb-0c7b-912f7ef45cb0
md"""
**iv**) TODO: how to best plot the potential and the force (??) (??) (??)
"""

# ╔═╡ 2515a9d4-85bb-11eb-2a51-b7c418cc0105
md"""
**v**) 
"""

# ╔═╡ ae64d270-8352-11eb-21ca-ffe8d5735353
let
	R₁ = [
		1 0.1 0
		0  1  0
	   -1  0  0
		0 -1  0
	]
	
	r, R, V = velocityVerletNaive(R₁, zeros(4, 3), 0.01, 5)
	
	writeOvito(R, "3bv.txt")
end

# ╔═╡ 9e8d7c9c-85bb-11eb-3a8f-0f502986dd44
md"""
### 3c) Initialisation
"""

# ╔═╡ 41f75302-85bd-11eb-0069-0398b242697d
md"""
**i**) 
"""

# ╔═╡ ceae3114-85bb-11eb-330c-19450635bf0d
"returns a 4n³×3 array with initial positions"
function lattice(n, L)
	R = zeros(4n^3, 3)
	d = L/n
	
	c = 0
	for i=1:n, j=1:n, k=1:n
		R[(1+4c):(4+4c), :] = [ i     j      k
					   		    i 	  0.5+j  0.5+k
					   		    0.5+i j      0.5+k
					   		    0.5+i 0.5+j  k ] * d

		c += 1
	end
	
	R
end

# ╔═╡ fea82f40-85bc-11eb-17cc-1de9da208b73
md"""
**ii**) Density is $$\rho = M / V = 4m / d^3$$ so $$d = \sqrt[3]{4m/\rho}$$ 
"""

# ╔═╡ 22e9174a-8679-11eb-26d0-a1ea19b9bfcd
σ = 3.405 * 10^-10;

# ╔═╡ 3ea6f6f4-85c1-11eb-1751-b979842b3efd
m = 39.95 * 1.660_539_066 * 10^-27; # kg

# ╔═╡ 04f06e0a-8677-11eb-235e-5bc200560750
ρ = 1.374 * 10^-3 * 10^6; # kg/m^3

# ╔═╡ 8ff1a094-8676-11eb-3d25-ed1441348d79
d = ∛(4m/ρ) / σ

# ╔═╡ 3ea6ac0c-85bd-11eb-160f-47dde08fa9af
let
	R₁ = lattice(3, d)
	n = size(R₁, 1)
	
	t, R, V = velocityVerlet(R₁, zeros(n, 3), 0.01, 5)
	
	writeOvito(R, "3cii.txt")
end

# ╔═╡ 5f8837a0-85be-11eb-2eae-b74d32ef778d
md"""
**iii**) 
"""

# ╔═╡ 0e5f929e-85c1-11eb-2038-738cc54e6cad
md"""
### 3d) Many atoms, open boundary
"""

# ╔═╡ 17ad2606-85c1-11eb-063d-39405078e41e
md"""
**i**) 
"""

# ╔═╡ 20a8c466-85c1-11eb-2afd-79a9373d2cbc
let
	n = 4 # ∛256/4
	R₁ = lattice(n, 5d)
	
	r, R, V = velocityVerlet(R₁, zeros(4n^3, 3), 0.01, 10) # tmax (??) 
	
	writeOvito(R, "3di.txt")
end

# ╔═╡ c298e560-85c3-11eb-2d43-299bee6c2f90
md"""
**ii**) (??) [See questions on 3a]
"""

# ╔═╡ d22bda3c-85c3-11eb-0f9e-f7c35830340b
md"""
### 3e) Boundary conditions
"""

# ╔═╡ 0c3376d6-85c4-11eb-0d12-796aebd716f9
md"""
**i**)
"""

# ╔═╡ 334a7a1c-85c4-11eb-3092-9dfc80376287
"R is a N×3 array containg all current positions"
function calc_forces_bounded(R, L)
	N = size(R,1)
	ΣF = zeros(N,3)
	T = zeros(N,N,3) # table

	for (i,j) ∈ combinations(1:N, 2) # each unordered particle pair
		r⃗ = R[i, :] - R[j, :]
		r⃗ = r⃗ - round.(r⃗/L)*L
		
		r = norm(r⃗)
		
		F = if r < 3
			24*(2r^-14 - r^-8)*r⃗
		else
			[0, 0, 0]
		end

		# to calculate in each direction only once

		T[i, j, :] = F
		T[j, i, :] = -F
	end

	for i=1:N
		ΣF[i, :] = sum(T[i, :, :], dims=1)
	end

	ΣF
end

# ╔═╡ 55aeb292-85c6-11eb-0c13-8bc44cd6cf48
function bound(x, L)
	if abs(x) < L
		x
	else
		sign(L - x)*L + (x % L)
	end
end

# ╔═╡ 08b5e140-85c4-11eb-34c4-e9f5b0d01152
function velocityVerletBounded(R₁, V₁, Δt, tmax; L) # (??) eeek
	N = size(R₁, 1) # nr. of atoms
	n = Int(floor(tmax/Δt))
	
	R = zeros(n, N, 3)
	V = zeros(n, N, 3)
	
	t = range(0, tmax-Δt, step=Δt)
	
	R[1,:,:] = R₁
	V[1,:,:] = V₁
	
	ΣFₚ = calc_forces_bounded(R₁, L)
	
	for k=1:n-1 # timestep 
		R[k+1,:,:] = bound.(R[k, :, :] + V[k, :, :]Δt + 1/2*ΣFₚ*Δt^2, L)
	
		ΣF = calc_forces_bounded(R[k+1,:,:], L)
		
		V[k+1,:,:] = V[k, :, :] + 1/2*(ΣFₚ + ΣF)Δt

		ΣFₚ = ΣF
	end
	
	return t, R, V
end

# ╔═╡ 09a7c642-8745-11eb-3f7b-0d65a462cbb5
L(n) = d*n/ 2 # per side around the origin

# ╔═╡ 75cb14a0-85c4-11eb-316c-65c5ad373814
let
	n = 4*3^3 # 108
	R₁ = lattice(3, 5d)
	V₁ = randn(n, 3) * √30
	
	r, R, V = velocityVerletBounded(R₁, V₁, 0.01, 100, L=L(n)) # tmax (??) 
	
	writeOvito(R, "3eii.txt")
end

# ╔═╡ d27cf722-8835-11eb-31ec-6f2e7ad04364
2L(4*3^3)

# ╔═╡ 7bfa1904-85cc-11eb-0144-f7a24ce8846e
md"""
## Part 4
"""

# ╔═╡ f2787170-85cc-11eb-330f-05215d805051
md"""
### a) Temperature
"""

# ╔═╡ d30043a8-85cd-11eb-2645-154032e65730
md"""
**i**)
"""

# ╔═╡ 19a2230e-85cd-11eb-1c6c-bb55169fd8bd
"V is a N×3 array containing the velocities"
function temperature(V)
	N = size(V, 1)
	
	1/(3N) * sum(V .^ 2)
end

# ╔═╡ d1c5caa0-85cd-11eb-0a1d-0d6908ba7f28
md"""
**ii**)
"""

# ╔═╡ 1e9f4668-85ce-11eb-0c76-8d488de07e26
let
	R₁ = lattice(3, 5d)
	n = size(R₁, 1) # 108
	V₁ = randn(n, 3) * √300
	
	t, R, V = velocityVerletBounded(R₁, V₁, 0.001, 0.5, L=L(n))
	
	writeOvito(R, "4aii.txt")
	
	T = [temperature(V[k,:,:]) for k=1:size(V,1)]

	m = mean(T)
	
	plot(t, [T, t->m], label=["T" "mean"])
end

# ╔═╡ 5d543b0c-874c-11eb-1990-05036b728c1b
md"""
We test 30 different initial temperatures with the standard deviation in the range from 90 to 130, and plot the mean temperature. We see some gaps in between the different mean temperatures, but by running this repeatly we see that these appear randomly and is not a feature of the particles' movement. 
"""

# ╔═╡ 4d7894ec-8749-11eb-036e-4d008f0db465
results4aiii = Dict{Float64, Float64}()

# ╔═╡ bf0a9446-8741-11eb-0453-37912fe8751f
let
	empty!(results4aiii)
	
	vs = range(90, 130, length=30)

	tmax = 0.5
	Δt = 0.001
	
	for v in vs
		R₁ = lattice(3, 5d)
		n = size(R₁, 1) # 108
		V₁ = randn(n, 3) * √v
		_, _, V = velocityVerletBounded(R₁, V₁, Δt, tmax, L=L(n))
		
		T = [temperature(V[k,:,:]) for k=1:size(V,1)]
		
		push!(results4aiii, v => mean(T))
	end
end

# ╔═╡ ec629756-8749-11eb-0cf9-ab1cdcaaf045
results4aiii |> values |> collect |> sort

# ╔═╡ 057e5900-874a-11eb-07c5-3ba09e4c5f25
let
	t = 0:0.001:(0.5-0.001)
	
	c(m) = t -> m
	
	plot()
	
	for v in keys(results4aiii)
		plot!(t, get(results4aiii, v, missing) |> c, label=format(v))
	end
	
	plot!(legend=false)
end

# ╔═╡ 53987f54-874e-11eb-16ac-b56d5b131946
md"""
### c) *Mean squared displacement and diffusion coefficient
"""

# ╔═╡ 77d05494-8750-11eb-3521-01a7281b268b
md"""
**i**)
"""

# ╔═╡ 8e28f392-874e-11eb-026a-858259a5a7b0
"""
R is N×3-array containing positions
Rᵣ is a N×3-array containing reference positions
"""
function msd(R, Rᵣ)
	N = size(R, 1)
	
	1/N * sum(norm(R[n,:] - Rᵣ[n,:])^2 for n=1:N)
end

# ╔═╡ 2f7e7a1a-875d-11eb-26c0-83c053c450bb
md"""
**ii**) 
"""

# ╔═╡ 6ab495ce-875d-11eb-32ab-b165547ac741
let
	R₁ = lattice(6, 5d)
	n = size(R₁, 1)
	V₁ = randn(n, 3) * √100
	
	Δt = 0.01
	tₘ = 0.1
	
	t, R, V = velocityVerletBounded(R₁, V₁, Δt, tₘ, L=L(n))
	
	writeOvito(R, "4cii.txt")
	
	T = [temperature(V[k,:,:]) for k=1:size(V,1)]
		
	Rᵣ = R[size(R,1)-1,:,:] # assuming R is at equilibrium at the end
	
	m = [msd(R[k,:,:], Rᵣ) for k=1:size(R,1)]
	
	with(:gr) do
		pT = plot(t, T, label=L"T")
		pm = plot(t, m, label=L"\langle r^2(t)\rangle")

		plot(pT, pm)
	end
end

# ╔═╡ Cell order:
# ╠═4cf0ab18-6647-11eb-2806-87ebde056fd0
# ╟─9d5f2ed2-6179-11eb-3069-a98bc0c0e21d
# ╟─c2abace0-6179-11eb-3d59-1faf66ea4e87
# ╟─6eeff9d0-6179-11eb-06a2-91e862d14160
# ╟─005bc24e-617a-11eb-0fec-1b98ef100475
# ╠═682a9e60-617a-11eb-2d0d-4108c271adde
# ╟─7186dbe0-617a-11eb-0a28-83e623ecc625
# ╟─7b8091e0-617a-11eb-0a66-63d1ccd63b7e
# ╟─169e1570-617c-11eb-3de8-1560387550ee
# ╟─28615b00-617c-11eb-00cc-65e571674f20
# ╟─05bf6e10-617d-11eb-1c47-11ecb389b8f1
# ╟─0f1a9a20-617d-11eb-3555-1111efddf3f4
# ╟─9650adb2-6180-11eb-3b35-45cfcc098604
# ╠═6e23014e-6183-11eb-2b24-b72dd880d0d6
# ╟─d87ec9d0-6183-11eb-39f1-eb1bff077ad9
# ╠═a565d0f0-6185-11eb-1bfd-b14fb9eaf444
# ╟─34b3ac60-618f-11eb-0f14-a1558af74e0e
# ╟─80c6c560-618f-11eb-11bb-2de11b2f4fd4
# ╟─8b110c62-618f-11eb-24c4-ef8c7745ab8d
# ╠═0c9fb470-6190-11eb-1f50-eb62baa85055
# ╟─00e759aa-6300-11eb-331a-ff43a6ab61a2
# ╠═7cd3e800-7b72-11eb-3698-274962748383
# ╟─11b5b4a2-6300-11eb-2771-3de463ce5388
# ╟─4147c3fe-6300-11eb-3c06-2bc5c32ccbf5
# ╠═97f5b360-6be2-11eb-3d29-bd8de9b3421d
# ╠═e4488500-6302-11eb-2c7b-93a0a0eb0d55
# ╠═19c52996-6307-11eb-2a43-41270dbf5f00
# ╠═12e77d24-6314-11eb-32c9-31675f6abc60
# ╠═67c89960-6579-11eb-390d-b9c51f16786a
# ╠═6f74f634-6be0-11eb-39e2-8bc1e64acff7
# ╠═8bd16432-6be0-11eb-0526-05fa4130cf8f
# ╠═89908eb4-6bd6-11eb-16b0-eb8889e905ab
# ╠═5a9bbf56-6be1-11eb-1cd0-f7657e42c215
# ╠═e60d0a02-6be8-11eb-19f2-7b4cd81d11d3
# ╠═9c8128ac-6be1-11eb-12f0-0b809f394278
# ╟─09a4454a-6be7-11eb-2c70-3d0b2e443bf6
# ╠═5115cd0a-6bd7-11eb-2b8b-113019761ec5
# ╠═021a639a-6bd8-11eb-3efc-d9349695af85
# ╠═ee22725e-6bd9-11eb-2334-23e1cdec1ea1
# ╠═2d7e7604-6313-11eb-04c5-b15da7afef17
# ╟─992bc6cc-63fc-11eb-2f97-c93fea5210b3
# ╟─592b8212-833a-11eb-3685-a11fe43f9d6f
# ╟─be8c8b24-833a-11eb-1638-73dd6c5bc19f
# ╠═e66a8a6c-831f-11eb-32ee-e12706005c1f
# ╟─c8c963fa-833a-11eb-0d2b-3b60b56ffaa6
# ╠═fbc231a4-85c1-11eb-0af5-6987101d71bc
# ╠═69672a0a-8673-11eb-1537-c79abb78c5df
# ╠═db18f48a-8321-11eb-306d-8db667b77a9d
# ╟─fd37ead6-6414-11eb-0196-c19d1a71a2ed
# ╠═977353f4-8350-11eb-1b77-53e68418f307
# ╠═b984168e-8352-11eb-0a3c-d7f1ed00a4aa
# ╠═5531984e-8674-11eb-177a-1921f5a7cc92
# ╠═a82a8900-85c2-11eb-0d21-1f18db14b68e
# ╠═f07ed7c8-8352-11eb-3498-3b6e0d0dc700
# ╠═ee4e444a-8350-11eb-2af2-8d80fb265f2b
# ╠═44b992b4-8351-11eb-1c37-1776cd088551
# ╠═3ee7809a-8351-11eb-1571-150953d80e06
# ╠═c81f8814-8351-11eb-2b31-f3d64a06edbb
# ╟─f95d6e5a-8351-11eb-0b50-d7c5040098b4
# ╠═06b9ab88-8352-11eb-19bb-3726a315ee54
# ╠═813e5028-8352-11eb-3384-1398b14cf967
# ╠═2f1da08a-85bb-11eb-0c7b-912f7ef45cb0
# ╠═2515a9d4-85bb-11eb-2a51-b7c418cc0105
# ╠═ae64d270-8352-11eb-21ca-ffe8d5735353
# ╠═9e8d7c9c-85bb-11eb-3a8f-0f502986dd44
# ╠═41f75302-85bd-11eb-0069-0398b242697d
# ╠═ceae3114-85bb-11eb-330c-19450635bf0d
# ╠═fea82f40-85bc-11eb-17cc-1de9da208b73
# ╠═22e9174a-8679-11eb-26d0-a1ea19b9bfcd
# ╠═3ea6f6f4-85c1-11eb-1751-b979842b3efd
# ╠═04f06e0a-8677-11eb-235e-5bc200560750
# ╠═8ff1a094-8676-11eb-3d25-ed1441348d79
# ╠═3ea6ac0c-85bd-11eb-160f-47dde08fa9af
# ╠═5f8837a0-85be-11eb-2eae-b74d32ef778d
# ╠═0e5f929e-85c1-11eb-2038-738cc54e6cad
# ╠═17ad2606-85c1-11eb-063d-39405078e41e
# ╠═20a8c466-85c1-11eb-2afd-79a9373d2cbc
# ╠═c298e560-85c3-11eb-2d43-299bee6c2f90
# ╠═d22bda3c-85c3-11eb-0f9e-f7c35830340b
# ╠═0c3376d6-85c4-11eb-0d12-796aebd716f9
# ╠═334a7a1c-85c4-11eb-3092-9dfc80376287
# ╠═55aeb292-85c6-11eb-0c13-8bc44cd6cf48
# ╠═08b5e140-85c4-11eb-34c4-e9f5b0d01152
# ╠═09a7c642-8745-11eb-3f7b-0d65a462cbb5
# ╠═75cb14a0-85c4-11eb-316c-65c5ad373814
# ╠═d27cf722-8835-11eb-31ec-6f2e7ad04364
# ╠═7bfa1904-85cc-11eb-0144-f7a24ce8846e
# ╠═f2787170-85cc-11eb-330f-05215d805051
# ╠═d30043a8-85cd-11eb-2645-154032e65730
# ╠═19a2230e-85cd-11eb-1c6c-bb55169fd8bd
# ╠═d1c5caa0-85cd-11eb-0a1d-0d6908ba7f28
# ╠═1e9f4668-85ce-11eb-0c76-8d488de07e26
# ╟─5d543b0c-874c-11eb-1990-05036b728c1b
# ╠═4d7894ec-8749-11eb-036e-4d008f0db465
# ╠═bf0a9446-8741-11eb-0453-37912fe8751f
# ╠═ec629756-8749-11eb-0cf9-ab1cdcaaf045
# ╠═057e5900-874a-11eb-07c5-3ba09e4c5f25
# ╠═53987f54-874e-11eb-16ac-b56d5b131946
# ╠═77d05494-8750-11eb-3521-01a7281b268b
# ╠═8e28f392-874e-11eb-026a-858259a5a7b0
# ╠═2f7e7a1a-875d-11eb-26c0-83c053c450bb
# ╠═6ab495ce-875d-11eb-32ab-b165547ac741
