include("plot_D_vector.jl")

using PyCall
@pyimport seaborn as sns
#sns.set(font_scale=1.5)
# sns.set_context("poster")
#sns.set_context("talk")
sns.set_context("paper")
sns.set(font_scale=2)


local_similarity = 0.9;


### First run : two macro tasks, single critic across four subtasks
local_c = 0.25;
local_critic_dims = 4;
setup_plot_D_basic_variables(local_similarity, local_c, local_critic_dims);
R_ext = 1.;
p_scale = 1.;

global use_include_learning_term_in_flow = true :: Bool;
global use_include_internal_bias_term_in_flow = false :: Bool;
global use_include_external_bias_term_in_flow = false :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_4critic_purelearn.pdf", bbox_inches="tight")


## internal bias

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = false :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_4critic_internalbias.pdf", bbox_inches="tight")


## external bias

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = false :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_4critic_externalbias.pdf", bbox_inches="tight")


## both biases

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_4critic_bothbiases.pdf", bbox_inches="tight")



## full system

global use_include_learning_term_in_flow = true :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_4critic_fullsystem.pdf", bbox_inches="tight")




### Second run : omit the hidden task, just have two subtasks on a single macro task
local_c = 0.5;
local_critic_dims = 2;
setup_plot_D_basic_variables(local_similarity, local_c, local_critic_dims);
R_ext = 0.95;

global use_include_learning_term_in_flow = true :: Bool;
global use_include_internal_bias_term_in_flow = false :: Bool;
global use_include_external_bias_term_in_flow = false :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_2critic_purelearn.pdf", bbox_inches="tight")


## internal bias

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = false :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_2critic_internalbias.pdf", bbox_inches="tight")


## external bias

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = false :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_2critic_externalbias.pdf", bbox_inches="tight")


## both biases

global use_include_learning_term_in_flow = false :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_2critic_bothbiases.pdf", bbox_inches="tight")



## full system

global use_include_learning_term_in_flow = true :: Bool;
global use_include_internal_bias_term_in_flow = true :: Bool;
global use_include_external_bias_term_in_flow = true :: Bool;

calculate_linear_model_flow_vectors();
plot_linear_model_flow_vectors();
title("");

savefig("testflowfigs_2critic_fullsystem.pdf", bbox_inches="tight")
