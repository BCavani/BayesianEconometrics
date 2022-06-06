### SIR, Sampling-Importance Resampling

### Author: Bruno Cavani

using Plots
using Distributions
using StatsBase
using PrettyTables

# Plotting posterior pi(theta) (up to a normalizing constant)

t = collect(0:.0001:1)
plot(t,(2*ones(length(t))+t).^(125).*(ones(length(t)).-t).^(38).*t.^(34),label="Posterior",c=:black)

# Let's compute the normalizing constant (also known as prior predictive) via Monte Carlo integration

M = 100000
a = 0
b = 1
theta = rand(Uniform(a,b),M)
pred = mean((2*ones(length(theta)).+theta).^(125).*(ones(length(theta)).-theta).^(38).*theta.^(34))
post = (2*ones(length(t))+t).^(125).*(ones(length(t)).-t).^(38).*t.^(34)/pred
plot(t,post,ylabel="Density",xlabel="theta",label="Prior-predictive")

# computing E[theta]

I2 = pred
I1 = mean((2*ones(length(theta)).+theta).^(125).*(ones(length(theta)).-theta).^(38).*theta.^(35))
E  = I1/I2


# Computing Var(theta) = E[theta^2]-{E[theta]}^2
E2 = mean((2*ones(length(theta)).+theta).^(125).*(ones(length(theta)).-theta).^(38).*theta.^(36))/I2
E2


V = E2-E^2
SD = sqrt(V)

M = 100000

# SIR, Step 1: sampling from q(.) = Uniform(0,1)

theta = rand(Uniform(0,1),M)
histogram(theta,title="Draws from the proposal distribution",legend=false)


# SIR, Step 2: cumputing weights

numerator = (2*ones(length(theta)).+theta).^(125).*(ones(length(theta)).-theta).^(38).*theta.^(34)
denominator = 1
w = numerator/denominator
w = w/sum(w)

# SIR, Step 3: resampling the set of thetas with the weights

theta1 = sample(theta, Weights(w),M,replace=true)
histogram(theta1,title="Draws from target-posterior",normalize=true,c=:white,label="SIR",size=(600,325))
plot!(t,post,c=:red,label="Posterior")

# Comparing MC integration with SIR

MCI = round.([E,V,SD],digits=6)
SIR = round.([mean(theta1),var(theta1),std(theta1)],digits=6)
data = hcat(["Mean","Variance","Standard Deviation"],MCI,SIR)
pretty_table(data,header=["","MCI","SIR"],header_crayon=crayon"blue bold")
