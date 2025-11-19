# ============================================================================
# PROPENSITY SCORE–MATCHED SENSITIVITY ANALYSIS
# ============================================================================

library(MatchIt)

# -----------------------------------------------------------
# 1. Identify model-concordant vs model-discordant patients
# -----------------------------------------------------------

# therapy_cols already defined from earlier code
# optimal_combos matrix and pred_test already exist from counterfactual analysis

test_df <- test_data
test_df$model_concordant <- NA

for(i in 1:nrow(test_df)) {
  observed_vec <- as.numeric(X_test[i, therapy_cols])
  optimal_vec  <- as.numeric(optimal_combos[i, ])
  test_df$model_concordant[i] <- as.integer(all(observed_vec == optimal_vec))
}

table(test_df$model_concordant)

# -----------------------------------------------------------
# 2. Construct baseline matching matrix (intake variables only)
# -----------------------------------------------------------

baseline_vars <- grep(
  "total_score|risk_high_initial|deterrents_month|frequency_month|duration_month|what_sort_of_reasons|when_you_have_the_thoughts_how_long|how_many_times|adolescent|male|dx_|current_and_past_psychiatric_diagnoses_|presenting_symptoms_|family_history_|precipitants_stressors_|internal_protective_factors_|external_protective_factors_|change_in_treatment_|program",
  colnames(test_df),
  value = TRUE
)

match_df <- test_df %>%
  select(model_concordant, improve, all_of(baseline_vars)) %>%
  mutate(model_concordant = as.factor(model_concordant))

# -----------------------------------------------------------
# 3. Mahalanobis 1:1 matching with caliper on suicide severity
# -----------------------------------------------------------

m.out <- matchit(
  model_concordant ~ .,
  data = match_df,
  method = "nearest",
  distance = "mahalanobis",
  ratio = 1,
  caliper = c(total_score = 0.2 * sd(match_df$total_score, na.rm = TRUE)),
  replace = FALSE
)

summary(m.out)


# -----------------------------------------------------------
# 4. Extract matched data and estimate risk difference
# -----------------------------------------------------------

matched_df <- match.data(m.out)

rd <- with(matched_df, mean(improve[model_concordant == 1]) -
             mean(improve[model_concordant == 0]))

se_rd <- sqrt(
  var(matched_df$improve[matched_df$model_concordant == 1]) /
    sum(matched_df$model_concordant == 1) +
    var(matched_df$improve[matched_df$model_concordant == 0]) /
    sum(matched_df$model_concordant == 0)
)

ci_lower <- rd - 1.96 * se_rd
ci_upper <- rd + 1.96 * se_rd

list(
  risk_difference = rd,
  ci_lower = ci_lower,
  ci_upper = ci_upper,
  matched_N = nrow(matched_df)
)



