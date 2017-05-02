using PyPlot
using Distributions

function bayes_initialise(sequence_length=100)
  # personalise parameters
  sequence_length = 100;
  task_repn_map_mean = zeros(Float64,2);
  task_repn_map_mean[1] = 0.3;
  task_repn_map_mean[2] = 0.7;
  task_repn_map_stdev = zeros(Float64,2);
  task_repn_map_stdev[1] = 0.3;
  task_repn_map_stdev[2] = 0.3;

  # save parameters to dictionary
  settings_d = Dict();
  settings_d["sequence_length"] = sequence_length;
  settings_d["task_repn_map_mean"] = task_repn_map_mean;
  settings_d["task_repn_map_stdev"] = task_repn_map_stdev;
  return settings_d;
end

function bayes_host()
  settings_dict = bayes_initialise();

  task_seq = generate_task_sequence(settings_dict);

  input_representations = zeros(Float64,settings_dict["sequence_length"]);
  hypothesis_representations = zeros(2,2);
  critic_representations = zeros(2,2);

  # main loop
  for i = 1:settings_dict["sequence_length"]
    input_representations[i] = get_input_representation(task_seq, i, settings_dict)
  end
  print("Done\n")
  simulation_run = Dict{String,Any}()
  simulation_run["task_seq"] = task_seq;
  simulation_run["input_representations_seq"] = input_representations;
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
