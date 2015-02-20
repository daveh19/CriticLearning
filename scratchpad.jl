if false
print_single_block_performance(exp_results[1].subjects_task1[9].blocks[end])


local_pre = pre(x, is_problem_1);
# Note: local_post returns a tuple where one value is 0. All comparisons to find the non zero value should use absolute comparison.
local_post = post(x, is_problem_1);

local_reward = -1
local_average_reward = 1

dw = zeros(no_pre_neurons, no_post_neurons);
dw[:,1] = learning_rate * local_pre[:] * local_post[1] * (local_reward - local_average_reward);
dw[:,2] = learning_rate * local_pre[:] * local_post[2] * (local_reward - local_average_reward);

end


## check RND statistics
using Distributions
using PyPlot
include("parameters_critic_simulations.jl")
srand(random_seed+1);

global a = rand(Normal(input_baseline,input_baseline_variance), no_pre_neurons);

beta = 0.375; # easier problem
pop1 = rand(Normal(0,beta), ((int)(no_pre_neurons/2)));
beta = 0.25; # harder problem
pop2 = rand(Normal(0,beta), (int)(no_pre_neurons/2));
global b = [pop1; pop2];

global tuning_pos = a + b;
global tuning_neg = a - b;

global w_initial = rand(Uniform(0,1), (no_pre_neurons, no_post_neurons));
w = deepcopy(w_initial);
w[:,1] += -initial_weight_bias*b;
w[:,2] += initial_weight_bias*b;

global ksi = rand(Normal(0,output_noise), no_post_neurons);


figure()
x = linspace(1,length(a),length(a));
scatter(x,a, marker="o", c="g")
title("a")
mean(a)
std(a)

figure()
title("histogram of a")
PyPlot.plt.hist(a,20);

figure()
x = linspace(1,length(pop1),length(pop2));
scatter(x,pop1, marker="o", c="g")
x = linspace(length(pop1) + 1, length(pop1) + length(pop2),length(pop2));
scatter(x,pop2, marker="o", c="r")
title("b")
mean(pop1)
std(pop1)
mean(pop2)
std(pop2)

figure()
title("histogram of pop1")
PyPlot.plt.hist(pop1,20);
figure()
title("histogram of pop2")
PyPlot.plt.hist(pop2,20);

figure()
title("tuning function")
x = linspace(1,length(tuning),length(tuning));
scatter(x, tuning_pos, marker="o", c="g", label="a+b")
scatter(x, tuning_neg, marker="o", c="r", label="a-b")
legend()
mean(tuning_pos)
std(tuning_pos)
mean(tuning_neg)
std(tuning_neg)


figure()
x = linspace(1,size(w_initial,1),size(w_initial,1));
scatter(x,w_initial[:,1], marker="o", c="g", label="left") # left
scatter(x,w_initial[:,2], marker="o", c="r", label="right") # right
title("initial weights")
legend()

figure()
x = linspace(1,size(w,1),size(w,1));
title("initial weights incorporating left-right biases")
scatter(x[1:(int)(no_pre_neurons/2)],w[1:(int)(no_pre_neurons/2),1], marker="o", c="g", label="left, easy") #left, easy
scatter(x[1:(int)(no_pre_neurons/2)],w[1:(int)(no_pre_neurons/2),2], marker="o", c="r", label="right, easy") # right, easy
scatter(x[(int)(no_pre_neurons/2)+1:end],w[(int)(no_pre_neurons/2)+1:end,1], marker="o", c="y", label="left, hard") # left, hard
scatter(x[(int)(no_pre_neurons/2)+1:end],w[(int)(no_pre_neurons/2)+1:end,2], marker="o", c="k", label="right, hard") # right, hard
legend()

mean(w_initial[:,1]) # left
std(w_initial[:,1]) # left
mean(w_initial[:,2]) # right
std(w_initial[:,2]) # right

mean(w[1:(int)(no_pre_neurons/2),1]) # left, easy
std(w[1:(int)(no_pre_neurons/2),1]) # left, easy
mean(w[1:(int)(no_pre_neurons/2),2]) # right, easy
std(w[1:(int)(no_pre_neurons/2),2]) # right, easy

mean(w[(int)(no_pre_neurons/2)+1:end,1]) # left, hard
std(w[(int)(no_pre_neurons/2)+1:end,1]) # left, hard
mean(w[(int)(no_pre_neurons/2)+1:end,2]) # right, hard
std(w[(int)(no_pre_neurons/2)+1:end,2]) # right, hard

figure()
scatter(pop1, w[1:(int)(no_pre_neurons/2),1], marker="o", c="g", label="left, easy")
scatter(pop2, w[(int)(no_pre_neurons/2)+1:end,1], marker="o", c="y", label="left, hard")
scatter(pop1, w[1:(int)(no_pre_neurons/2),2], marker="o", c="r", label="right, easy")
scatter(pop2, w[(int)(no_pre_neurons/2)+1:end,2], marker="o", c="k", label="right, hard")
legend()
xlabel("tuning bias (b)")
ylabel("initial weight")
title("tuning bias (b) vs initial weight")


loop_length = no_trials_in_block * no_blocks_in_experiment;
ksi = Array(Float64,2,loop_length);
for i = 1:loop_length
	ksi[:,i] = rand(Normal(0,output_noise), no_post_neurons);
end
mean(ksi[1,:])
std(ksi[1,:])
mean(ksi[2,:])
std(ksi[2,:])

maximum(ksi[1,:])
minimum(ksi[1,:])
maximum(ksi[2,:])
minimum(ksi[2,:])

median(ksi[1,:])
median(ksi[2,:])

figure()
title("histogram of left noise")
n1,bins1 = PyPlot.plt.hist(ksi[1,:]',100);
figure()
title("histogram of right noise")
n2,bins2 = PyPlot.plt.hist(ksi[2,:]',100);


