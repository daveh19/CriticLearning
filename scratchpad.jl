###########
# handy fig mod code

latest_experiment_results = exp_results[1];

# modify contents of plt[:rcParams] dictionary
rc("axes",labelsize="xx-large")


title("")
f = gcf()
f[:set_dpi](100)
f[:set_size_inches](4.5,5,forward=true);
f[:savefig]("sinergia_flow.pdf",  bbox_inches="tight")



f = gcf()
f[:set_dpi](100)
f[:set_size_inches](5,5,forward=true);
f[:savefig]("sinergia_flow.pdf",  bbox_inches="tight")


rcdefaults() # restore default settings for matplotlib
ion() # turn interactive plots back on after reset
plt[:show]() # manually show a plot if interactive plots is off


D = get_fignums()
for i in D
  plt[:figure](i)
  #ylabel("test")
  f[:savefig](string("figure_title_%i.pdf"),  bbox_inches="tight")
end


f[:savefig]("figure_title.pdf",  bbox_inches="tight")


plt[:rcParams]["axes.labelsize"]
fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
                 'x-large', 'xx-large', 'smaller', 'larger']


###########
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


## LLVM code
:(2 + 2) # abstract-syntax-tree
code_lowered(generic_function, (types_arg_list,))
code_typed(generic_function, (types_arg_list,))
code_llvm(generic_function, (types_arg_list,))
code_native(generic_function, (types_arg_list,))
code_typed(sort, ( Array{Float64,1}, ) )

# developing gaussian tuning curves
type plain_tc_type
           no_curves :: Int;
           mu
           sigma
           height
end
d = plain_tc_type(1, Array(Float64, (3,1)), Array(Float64, (3,1)), Array(Float64, (3,1)) );

type tc_type
	no_curves :: Int;
	mu :: Array{Float64, 1};
	sigma :: Array{Float64, 1};
	height :: Array{Float64, 1};
end

figure()
no_pre_neurons = 50;
a = Array(tc_type, no_pre_neurons);
for i=1:no_pre_neurons;
	no_curves = 1;
	tuning_mu = rand(Uniform(-1,1), no_curves);
	tuning_sigma = ones(no_curves);
	tuning_sigma *= 0.25;
	tuning_height = rand(Normal(2,0.25), no_curves);
	c = tc_type(no_curves, tuning_mu, tuning_sigma, tuning_height);

	scatter(tuning_mu, tuning_height, c="r");
	scatter(tuning_mu, tuning_sigma, c="b");
	a[i] = c;
end

# fixing subjects who don't learn
for i = 1:no_subjects
    restore_subject(exp_results[1].subjects_task[i,1]);
    print("Subject $i, left: ", sum(w[:,1,1]), ", right: ", sum(w[:,2,1]),"\n")
end
Subject restored
Subject 1, left: 29.929715336783776, right: 14.681446176525938
Subject restored
Subject 2, left: 22.07651633118477, right: 31.45468877498815
Subject restored
Subject 3, left: 23.673822228646394, right: 27.05970223473139
Subject restored
Subject 4, left: 33.33160050234709, right: 22.84921345900872
Subject restored
Subject 5, left: 26.546413463743338, right: 26.4222037778666
Subject restored
Subject 6, left: 22.67275763527978, right: 26.929372831385127
Subject restored
Subject 7, left: 28.58124434653703, right: 25.46907869470326
Subject restored
Subject 8, left: 27.163487195103343, right: 24.24119733686835
Subject restored
Subject 9, left: 18.670667687673983, right: 32.057128550437774
Subject restored
Subject 10, left: 28.81075674459314, right: 19.680580385954322

function plot_subjects_initial_weight_distributions(subjects::Array{Subject,2}, task_id::Int=1)
	(no_subjects, no_tasks) = size(subjects);

	inter_subject_gap = 0.1;
	lr_gap = (no_subjects+2) * inter_subject_gap;
	figure()
	x1 = ones(no_pre_neurons_per_task);
	x2 = ones(no_pre_neurons_per_task) * lr_gap;

	for i = 1:no_subjects
		restore_subject(subjects[i,task_id]);
		#=scatter(x1+( (i-1) * inter_subject_gap), w[:,1,1], c="b")
		scatter(x2+( (i-1) * inter_subject_gap), w[:,2,1], c="g")=#
		scatter( (i * x1) , w[:,1,1], c="b")
		scatter( (i * x1) + 0.5, w[:,2,1], c="g")
	end
end

end # end of 'false'


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
