###########
# optimize is good and can use multiple methods
using Optim

function my_null(x::Vector)
	return pdf(Normal(0,1),x[1]) + (cdf(Normal(0,1),x[1]) - cdf(Normal(0,1),x[2]) ) * 0.5 * x[1]
end

function my_null2(x::Vector)
	temp = (sigma^2 * 2 * pdf(Normal(0,sigma),x[1]) + sigma^2 * 2 * pdf(Normal(0,sigma),x[2]));
	temp_1 = x[1] * (0.75 * (2 * cdf(Normal(0,sigma),x[1]) - 1) - 0.25 * (2 * cdf(Normal(0,sigma),x[2]) - 1) )
	temp_2 =  x[2] * (-0.25 * (2 * cdf(Normal(0,sigma),x[1]) - 1) + 0.75 * (2 * cdf(Normal(0,sigma),x[2]) - 1) )
	temp_3 = (x[1] * (-0.5 * R_ext) + x[2] * (-0.5 * R_ext) )
	ret_val = (temp + temp_1 + temp_2 + temp_3);
	if (abs(x[1]) > 100 || abs(x[2]) > 100)
		ret_val = Inf;
	end
	return  ret_val
end

function my_null3(x::Vector)
	if ((abs(x[1]) > 1.) || (abs(x[2]) > 1.))
		ret_val = Inf
	else
		ret_val = 2* pdf(Normal(0,1),invphi(x[1])) + invphi(x[1]) * ((1 - 0.25) * x[1] - (0.25 * x[2]) ) ;
	end
	return  ret_val
end

f = my_null;

res1 = optimize(f, [0.0010, 0.0010], method = :nelder_mead, store_trace = true)

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

