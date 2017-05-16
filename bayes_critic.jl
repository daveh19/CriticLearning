using PyPlot
using Distributions
using StatsBase


function bayes_initialise(sequence_length=100)
  # personalise parameters
  sequence_length = 100;
  task_repn_map_mean = zeros(Float64,2);
  task_repn_map_mean[1] = 0.3;
  task_repn_map_mean[2] = 0.7;
  task_repn_map_stdev = zeros(Float64,2);
  # task_repn_map_stdev[1] = sqrt(0.01);
  # task_repn_map_stdev[2] = sqrt(0.01);
  bernoulli_conversion_theta = 0.5;

  # save parameters to dictionary
  settings_d = Dict();
  settings_d["sequence_length"] = sequence_length;
  settings_d["task_repn_map_mean"] = task_repn_map_mean;
  settings_d["task_repn_map_stdev"] = task_repn_map_stdev;
  settings_d["bernoulli_conversion_theta"] = bernoulli_conversion_theta;

  critic_params = Dict();
  critic_p = zeros(Float64, 2);
  critic_p[1] = 0.5;
  critic_p[2] = 0.5;
  critic_params["p"] = critic_p;
  return (settings_d, critic_params);
end


function bayes_host()
  (settings_dict, critic_dict) = bayes_initialise();

  task_seq = generate_task_sequence(settings_dict);

  input_representations = zeros(Float64,settings_dict["sequence_length"]);
  critic_representations = zeros(2,2,settings_dict["sequence_length"]);
  hypothesis_representations = zeros(2,2);

  # main loop
  for i = 1:settings_dict["sequence_length"]
    input_representations[i] = get_input_representation(task_seq, i, settings_dict)
    critic_representations[:,:,i] = get_pdfs_critic_given_d(input_representations[i], critic_dict, settings_dict)
  end
  print("Done\n")
  simulation_run = Dict{String,Any}()
  simulation_run["task_seq"] = task_seq;
  simulation_run["input_representations_seq"] = input_representations;
  simulation_run["critic_representations_seq"] = critic_representations;
  # @show simulation_run
  return simulation_run;
end


function generate_task_sequence(settings_dict::Dict)
  sequence_length = settings_dict["sequence_length"]
  sequence_id = zeros(Int,sequence_length,1);
  for i = 1:sequence_length
    sequence_id[i] = (rand(Uniform(0,1)) < 0.5 ? 1 : 2);
  end
  return sequence_id;
end


function get_input_representation(task_sequence::Array{Int,2}, trial_number::Int, settings_dict::Dict)
  task_id = task_sequence[trial_number];
  representation_value = 0.0;
  # @show task_id

  if (task_id == 1)
    # model input representation as a gaussian (initially zero variance)
    representation_value = settings_dict["task_repn_map_mean"][1] + rand(Normal(0,1)) * settings_dict["task_repn_map_stdev"][1];
  elseif (task_id == 2)
    representation_value = settings_dict["task_repn_map_mean"][2] + rand(Normal(0,1)) * settings_dict["task_repn_map_stdev"][2];
  else
    print("This should not happen, you have a non-valid task ID\n")
  end
  # @show representation_value

  return representation_value;
end


function convert_input_representation_to_bernoulli(input_representation_d, settings_dict)::Int
  theta = settings_dict["bernoulli_conversion_theta"];

  # We're using a Bernoulli critic representation, so just collapse input_representation_d
  #   into two variables, greater than and less than theta.
  if input_representation_d < theta
    discrete_input_class = 1;
  else
    discrete_input_class = 2;
  end

  return discrete_input_class;
end


# Critic is a Bernoulli(p) process. It can take on only two discrete output values
#   hence the two rows in the output. We currently have two 'critics' in the system.
function get_pdfs_critic(input_representation_d, critic_dict)
  critic_per_column_array = zeros(2,2);
  # Notation: each column of the array represents a different 'critic' in the system
  #   each row is the probability of that critic attaining that (row_id) value
  for i = 1:2
    critic_per_column_array[1,i] = critic_dict["p"][i]
    critic_per_column_array[2,i] = (1 - critic_dict["p"][i])
  end

  return critic_per_column_array;
end


# This is the mid-level function, which works out d_given_c from a Bernoulli
#   distribution.
# The logic of using a pdf() is bogus in the discrete Bernoulli case, but I'll keep
#   it for now as it is much more applicable to the continuous generalisation.
#   should really be called get_P_d_given_critic()
function get_P_d_given_critic(input_representation_d, critic_dict, settings_dict)
  probability_of_d = zeros(1,2); # it's a row as each one is for a different critic

  # We're using a Bernoulli critic representation, so just collapse input_representation_d
  #   into two variables, greater than and less than theta=0.5
  discrete_input_class = convert_input_representation_to_bernoulli(input_representation_d, settings_dict);

  # calculating for both critics!
  if discrete_input_class == 1
    probability_of_d[1] = critic_dict["p"][1];
    probability_of_d[2] = critic_dict["p"][2];
  else
    probability_of_d[1] = (1 - critic_dict["p"][1]);
    probability_of_d[2] = (1 - critic_dict["p"][2]);
  end

  return probability_of_d;
end


# This is the outer function, which calls get_d_given_c and get_c
function get_P_critic_given_d(input_representation_d, critic_dict, settings_dict)
  d_given_c = get_P_d_given_critic(input_representation_d, critic_dict, settings_dict);
  pC = get_pdfs_critic(input_representation_d, critic_dict);

  return ones(2,2);
end


# to plot
#   individual beta distributions implied by critic representations
#   individual hypothesis-space representations
#   combined critic representation using full prior based prediction
#   (Done) histogram/distribution of inputs

function plot_input_representation(input_representations_seq, nbins)
  # xkcd()
  figure();
  # StatsBase fitting
  h = StatsBase.fit(Histogram,input_representations_seq, nbins=nbins)#,-0.6:0.1:1.6) #nbins=5)
  @show h
  # make a pdf
  # h = normalize(h)

  max_val = maximum(input_representations_seq);
  min_val = minimum(input_representations_seq);
  bar_width = ((max_val - min_val) / nbins) - 0.001;

  # data points centered on the value being represented
  x_coords = (h.edges[1][1:end-1] + h.edges[1][2:end]) / 2.0;
  # x_coords = h.edges[1][1:end-1]
  # line
  plot(x_coords, h.weights, "red")
  # bar
  # y_coords = convert(Array{Float64,1}, h.weights)
  # y_coords[y_coords .==0] 1e-7
  y_coords = h.weights
  bar(x_coords, y_coords, align="center", color="g", alpha=0.4, width=bar_width)

  # pyplot histogram
  # h2 = plt[:hist](input_representations_seq,nbins)

  # inbuild hist() function
  # hist(input_representations_seq)
  title("Input representations presented")
  xlabel("Arbitrary input representation (a.u)")
  ylabel("Number of presentations")
  return h
end
