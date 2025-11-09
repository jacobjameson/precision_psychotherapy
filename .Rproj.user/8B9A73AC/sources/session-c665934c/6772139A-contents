library(ggplot2)
library(dplyr)
library(cowplot)
library(viridis)

# Set publication theme
theme_publication <- function() {
  theme_minimal() +
    theme(
      text = element_text(size = 12),
      plot.title = element_text(size = 12, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 10, color = "black", hjust = 0),
      axis.text = element_text(size = 9, color = "black"),
      axis.line = element_line(color = "black", size = 0.3),
      axis.ticks = element_line(color = "black", size = 0.3),
      panel.grid.major = element_line(color = "#E5E5E5", size = 0.2),
      panel.grid.minor = element_blank(),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 10),
      legend.position = "right",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      strip.text = element_text(size = 12, face = "bold"),
      plot.margin = margin(10, 10, 10, 10)
    )
}

# ============================================================================
# PANEL A: SHAP ANALYSIS (FILTERED)
# ============================================================================

cat("Creating SHAP plot with filtered features...\n")

# Calculate SHAP values if not already done
if(!exists("shap_values_clean")) {
  set.seed(123)
  shap_sample_size <- min(10000, nrow(X_test))  # Increased sample for stability
  shap_idx <- sample(1:nrow(X_test), shap_sample_size)
  shap_data <- xgb.DMatrix(X_test[shap_idx, ])
  shap_values <- predict(xgb_final, shap_data, predcontrib = TRUE, approxcontrib = FALSE)
  shap_values_clean <- shap_values[, -ncol(shap_values)]
  feature_names <- colnames(X_test)
}

# Define features to include (exclude therapist/location/program fixed effects)
include_patterns <- c(
  # Patient clinical features
  "^total_score$", "^risk_high_initial$", "^deterrents_month$", 
  "^what_sort_of_reasons$", "^duration_month$", "^adolescent$", "^male$",
  "^frequency_month$", "^are_there_things$",
  "^how_many_times_have_you_had_these_thoughts$",
  "^when_you_have_the_thoughts_how_long_do_they_last$",
  "^could_can_you_stop_thinking_about_killing_yourself",
  
  # Diagnosis and symptoms
  "^current_and_past_psychiatric_diagnoses_",
  "^presenting_symptoms_",
  "^family_history_",
  "^precipitants_stressors_",
  "^internal_protective_factors_",
  "^external_protective_factors_",
  "^change_in_treatment_",
  "^dx_group_",
  
  # Therapy features
  "^act$", "^cbt$", "^dbt$", "^motivational_interviewing$",
  "^mindfulness$", "^stages_of_change$", "^family_systems$",
  
  # Propensity scores
  "^prop_",
  
  # Treatment context (but not provider IDs)
  "^therapy_duration_category_", "^delivery_method_", "^session_mode_"
)

# Exclude patterns (therapist/location fixed effects)
exclude_patterns <- c(
  "^therapist_name_", "^location_", "^program_", 
  "^pn_month", "^pn_time_block", "^pn_year",   "^intake_to_pn$", "^days_first_srs$"
)

# Filter features
keep_features <- logical(length(feature_names))
for(i in seq_along(feature_names)) {
  # Check if feature matches any include pattern
  include <- any(sapply(include_patterns, function(p) grepl(p, feature_names[i])))
  # Check if feature matches any exclude pattern
  exclude <- any(sapply(exclude_patterns, function(p) grepl(p, feature_names[i])))
  keep_features[i] <- include & !exclude
}

filtered_features <- feature_names[keep_features]
cat(sprintf("  Filtered to %d features from %d total\n", 
            length(filtered_features), length(feature_names)))

# Calculate importance for filtered features only
filtered_shap <- shap_values_clean[, keep_features]
mean_abs_shap <- colMeans(abs(filtered_shap))
names(mean_abs_shap) <- filtered_features

# Get top 20 features
top_features <- names(sort(mean_abs_shap, decreasing = TRUE)[1:20])

# Create data for plot
shap_plot_data <- data.frame()
for(feat in top_features) {
  feat_idx <- which(feature_names == feat)
  shap_plot_data <- rbind(shap_plot_data,
                          data.frame(
                            feature = feat,
                            feature_value = X_test[shap_idx, feat_idx],
                            shap_value = shap_values_clean[, feat_idx]
                          ))
}

# Create clean feature labels
shap_plot_data$feature_clean <- case_when(
  # Core clinical features
  shap_plot_data$feature == "total_score" ~ "Total symptom score",
  shap_plot_data$feature == "risk_high_initial" ~ "High risk at intake",
  shap_plot_data$feature == "days_first_srs" ~ "Days to reassessment",
  shap_plot_data$feature == "intake_to_pn" ~ "Days to first therapy",
  shap_plot_data$feature == "adolescent" ~ "Adolescent",
  shap_plot_data$feature == "male" ~ "Male sex",
  
  # C-SSRS items
  shap_plot_data$feature == "frequency_month" ~ "SI frequency (past month)",
  shap_plot_data$feature == "duration_month" ~ "SI duration (past month)",
  shap_plot_data$feature == "deterrents_month" ~ "Deterrents present",
  shap_plot_data$feature == "are_there_things" ~ "Protective factors",
  
  # Therapies
  shap_plot_data$feature == "cbt" ~ "CBT received",
  shap_plot_data$feature == "dbt" ~ "DBT received",
  shap_plot_data$feature == "act" ~ "ACT received",
  shap_plot_data$feature == "motivational_interviewing" ~ "MI received",
  shap_plot_data$feature == "mindfulness" ~ "Mindfulness received",
  shap_plot_data$feature == "family_systems" ~ "Family therapy received",
  shap_plot_data$feature == "stages_of_change" ~ "Stages of change received",
  
  # Propensity scores
  grepl("^prop_cbt", shap_plot_data$feature) ~ "Propensity: CBT",
  grepl("^prop_dbt", shap_plot_data$feature) ~ "Propensity: DBT",
  grepl("^prop_act", shap_plot_data$feature) ~ "Propensity: ACT",
  grepl("^prop_mi", shap_plot_data$feature) ~ "Propensity: MI",
  grepl("^prop_mindfulness", shap_plot_data$feature) ~ "Propensity: Mindfulness",
  grepl("^prop_family", shap_plot_data$feature) ~ "Propensity: Family",
  
  # Diagnoses
  grepl("depression", shap_plot_data$feature, ignore.case = TRUE) ~ "Dx: Depression",
  grepl("anxiety", shap_plot_data$feature, ignore.case = TRUE) ~ "Dx: Anxiety",
  grepl("bipolar", shap_plot_data$feature, ignore.case = TRUE) ~ "Dx: Bipolar",
  grepl("ptsd|trauma", shap_plot_data$feature, ignore.case = TRUE) ~ "Dx: PTSD/Trauma",
  
  # Symptoms
  grepl("presenting_symptoms_", shap_plot_data$feature) ~ 
    gsub("presenting_symptoms_", "Symptom: ", shap_plot_data$feature),
  
  # Default
  TRUE ~ gsub("_", " ", shap_plot_data$feature)
)


# Normalize feature values for coloring (0-1 scale)
shap_plot_data <- shap_plot_data %>%
  group_by(feature) %>%
  mutate(feature_value_norm = (feature_value - min(feature_value, na.rm = TRUE)) /
           (max(feature_value, na.rm = TRUE) - min(feature_value, na.rm = TRUE) + 1e-10))

# Create clean feature labels for SHAP plot (C-SSRS-accurate)
shap_plot_data$feature_clean <- case_when(
  # Risk and severity measures
  shap_plot_data$feature == "risk_high_initial" ~ "High risk at intake",
  shap_plot_data$feature == "total_score" ~ "C-SSRS total score",
  
  # C-SSRS specific items about suicidal ideation
  shap_plot_data$feature == "frequency_month" ~ "SI frequency (past month)",
  shap_plot_data$feature == "duration_month" ~ "SI duration when present (past month)",
  shap_plot_data$feature == "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to" ~ "SI controllability (past month)",
  shap_plot_data$feature == "deterrents_month" ~ "Deterrents to suicide (past month)",
  shap_plot_data$feature == "how_many_times_have_you_had_these_thoughts" ~ "Number of SI episodes",
  shap_plot_data$feature == "when_you_have_the_thoughts_how_long_do_they_last" ~ "Duration of SI episodes",
  shap_plot_data$feature == "what_sort_of_reasons" ~ "Reasons for SI",
  shap_plot_data$feature == "are_there_things" ~ "Factors preventing suicide",
  
  # Protective factors (internal)
  shap_plot_data$feature == "internal_protective_factors_identifies_reasons_for_living" ~ "Identifies reasons for living",
  shap_plot_data$feature == "internal_protective_factors_ability_to_cope_with_stress" ~ "Ability to cope with stress",
  shap_plot_data$feature == "internal_protective_factors_able_to_access_care_willing_to_reach_out" ~ "Willing to seek help",
  shap_plot_data$feature == "internal_protective_factors_fear_of_death_or_the_actual_act_of_killing_self" ~ "Fear of death/dying",
  
  # Protective factors (external)
  shap_plot_data$feature == "external_protective_factors_supportive_social_network_of_family_or_friends" ~ "Supportive social network",
  
  # Risk factors/stressors
  shap_plot_data$feature == "precipitants_stressors_social_isolation" ~ "Social isolation",
  shap_plot_data$feature == "precipitants_stressors_inadequate_social_supports" ~ "Inadequate social support",
  
  # Treatment received
  shap_plot_data$feature == "stages_of_change" ~ "Stages of Change (received)",
  
  # Propensity scores (likelihood of receiving each therapy)
  shap_plot_data$feature == "prop_act" ~ "Propensity to receive ACT",
  shap_plot_data$feature == "prop_cbt" ~ "Propensity to receive CBT",
  shap_plot_data$feature == "prop_dbt" ~ "Propensity to receive DBT",
  shap_plot_data$feature == "prop_mi" ~ "Propensity to receive MI",
  shap_plot_data$feature == "prop_mindfulness" ~ "Propensity to receive Mindfulness",
  shap_plot_data$feature == "prop_family_systems" ~ "Propensity to receive Family therapy",
  shap_plot_data$feature == "prop_stages_of_change" ~ "Propensity to receive Stages of Change",
  
  # Default
  TRUE ~ shap_plot_data$feature
)

feature_order <- shap_plot_data %>%
  group_by(feature_clean) %>%
  summarise(mean_abs = mean(abs(shap_value))) %>%
  arrange(desc(mean_abs))

shap_plot_data$feature_clean <- factor(shap_plot_data$feature_clean,
                                       levels = rev(feature_order$feature_clean))

# Create SHAP plot
panel_a <- ggplot(shap_plot_data, aes(x = shap_value, y = feature_clean)) +
  geom_jitter(aes(color = feature_value_norm),
              height = 0.15, width = 0, size = 0.8, alpha = 0.7) +
  scale_color_gradient2(low = "#471aa6", mid = "#F4F4F4", high = "#b91313",
                        midpoint = 0.5,
                        name = "Feature\nvalue\n",
                        breaks = c(0, 0.999),
                        labels = c("Low",  "High")) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red", size = 0.3) +
  labs(
    title = "A. Feature importance for improvement prediction",
    subtitle = "SHAP values quantify impact on predicted probability",
    x = "SHAP value (impact on prediction)",
    y = NULL
  ) +
  theme_publication() +
  theme(
    legend.position = c(0.85, 0.4),
    legend.background = element_rect(fill = "white", color = "#CCCCCC", size = 0.3),
    legend.key.size = unit(0.4, "cm"),
    axis.text.y = element_text(size = 8)
  )

# ============================================================================
# PANEL B: DOUBLY ROBUST ESTIMATION
# ============================================================================

if(length(dr_results) > 0) {
  # Create data frame from results
  dr_df <- data.frame(
    therapy = names(dr_results),
    ate = sapply(dr_results, function(x) x$ate),
    ci_lower = sapply(dr_results, function(x) x$ci_lower),
    ci_upper = sapply(dr_results, function(x) x$ci_upper),
    se = sapply(dr_results, function(x) x$se),
    n_treated = sapply(dr_results, function(x) x$n_treated)
  )
  
  # Clean therapy names
  dr_df$therapy_clean <- case_when(
    dr_df$therapy == "cbt" ~ "CBT",
    dr_df$therapy == "dbt" ~ "DBT",
    dr_df$therapy == "act" ~ "ACT",
    dr_df$therapy == "motivational_interviewing" ~ "Motivational Interviewing",
    dr_df$therapy == "mindfulness" ~ "Mindfulness",
    dr_df$therapy == "stages_of_change" ~ "Stages of Change",
    dr_df$therapy == "family_systems" ~ "Family Systems",
    TRUE ~ dr_df$therapy
  )
  
  # Add significance stars
  dr_df$sig <- case_when(
    abs(dr_df$ate) / dr_df$se > 2.58 ~ "***",  # p < 0.01
    abs(dr_df$ate) / dr_df$se > 1.96 ~ "**",   # p < 0.05
    abs(dr_df$ate) / dr_df$se > 1.64 ~ "*",    # p < 0.10
    TRUE ~ ""
  )
  
  # Order by effect size
  dr_df <- dr_df %>% arrange(ate)
  dr_df$therapy_clean <- factor(dr_df$therapy_clean, levels = dr_df$therapy_clean)
  
  # Create forest plot
  panel_b <- ggplot(dr_df, aes(x = therapy_clean, y = ate * 100)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.3) +
    geom_errorbar(aes(ymin = ci_lower * 100, ymax = ci_upper * 100),
                  width = 0, size = 0.4, color = "#333333") +
    geom_point(aes(fill = ate > 0), shape = 21, size = 3.5, 
               color = "#333333", stroke = 0.5) +
    scale_fill_manual(values = c("FALSE" = "black", "TRUE" = "black"),
                      guide = "none") +
    coord_flip() +
    labs(
      title = "B. Therapy-specific associations with improvement",
      subtitle = "Doubly robust estimates accounting for selection bias",
      x = NULL,
      y = "Difference in improvement probability (percentage points)"
    ) +
    theme_publication() +
    theme(
      axis.text.y = element_text(size = 9)
    ) }

# ============================================================================
# COMBINE PANELS
# ============================================================================

# Combine plots using cowplot
figure <- plot_grid(
  panel_a, panel_b,
  ncol = 1,
  rel_widths = c(2, 1),
  align = "h",
  axis = "tb"
)

ggsave("outputs/figures/figure3.png", 
       figure, 
       width = 8 ,
       height = 8, 
       dpi = 300,
       bg = "white")


