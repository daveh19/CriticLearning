###########
# optimize is good and can use multiple methods
using Optim

function my_null(x::Vector)
	return pdf(Normal(0,1),x[1]) + (cdf(Normal(0,1),x[1]) - cdf(Normal(0,1),x[2]) ) * 0.5 * x[1]
end

f = my_null;

res1 = optimize(f, [0.0, 0.0], method = :nelder_mead, store_trace = true)

#optimize(f, [0.0, 0.0], method = :l_bfgs)
# a slight offset from origin is required for bfgs to detect slope correctly
res2 = optimize(f, [0.0, 0.06], method = :l_bfgs, store_trace = true)



##########
# curve_fit is less reliable and uses larger tolerance value
#=using Distributions
using LsqFit

model(x, p) = pdf(Normal(0,1),p[1]) + (cdf(Normal(0,1),p[1]) - cdf(Normal(0,1),p[2]) ) * 0.5 * p[1]

#model([],[0,0])

xdata = [0]
ydata = [0]

# a slight offset from origin is required for curve_fit to detect slope correctly
fit = curve_fit(model, xdata, ydata, [0.5, 0.5])

fit.param
model([],fit.param)=#


##########
# fminbox requires definition of jacobian and hessian apparently
#=d = DifferentiableFunction(f)
l = [-1, -1]
u = [Inf, Inf]
x0 = [0.0, -0.5]
results = fminbox(d, x0, l, u) =#


##########
# NLopt provides the same functionality as Optim but via the more reliable GSL-NLopt libraries
## consider implementing

