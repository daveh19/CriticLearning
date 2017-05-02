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
