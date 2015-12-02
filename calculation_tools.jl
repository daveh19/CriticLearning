
function generate_sim_vars_for_a_given_similarity(target_similarity = 0.6)
  variance_b = (1-target_similarity) ./ (1 + target_similarity);

  return variance_b;
end
