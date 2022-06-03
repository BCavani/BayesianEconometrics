### Monte Carlo Integration and MCI via Importance Function

### Author: Bruno Cavani

using Distributions
using SpecialFunctions
using Plots

# Monte Carlo Integration

M = 10000
n = 20
sim = zeros(n,M)

for i in 1:n
    theta = rand(Beta(8,4),M)
    gtheta = theta
    sim[i,:] = cumsum(gtheta)./collect(1:M)
end

mean(sim)

# Monte Carlo Integration via Importance Function

# Let Uniform(a,b) be the Importance Function

a = 0.3
b = 1.0
sim1 = zeros(20,M)

for i in 1:n
    theta = rand(Uniform(a,b),M)
    gtheta = theta.*pdf.(Beta(8,4),theta)./(1/(b-a))
    sim1[i,:] = cumsum(gtheta)./collect(1:M)
end

mean(sim1)

plot(transpose(sim1),c=:grey,ylim=(0.6,0.75),xlabel="Monte Carlo sample size",ylabel="Expectation",legend=false,
     fmt = png)
for i in 1:n
    plot!(sim[i,:],c=:black)
end
hline!([2/3],c=:red)
current()
