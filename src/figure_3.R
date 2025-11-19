library(ggplot2)
library(dplyr)
library(cowplot)
library(viridis)

# Set publication theme
theme_publication <- function() {
  theme_minimal() +
    theme(
      text = element_text(size = 12),
      plot.title = element_text(size = 12, hjust = 0),
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
shap_plot_data$feature_clean <- case_when(
  shap_plot_data$feature == "risk_high_initial" ~ 
    "High suicide risk at intake",
  
  shap_plot_data$feature == "total_score" ~ 
    "C-SSRS total score (0–25)",
  
  shap_plot_data$feature == "how_many_times_have_you_had_these_thoughts" ~ 
    "SI frequency (lifetime)",
  
  shap_plot_data$feature == "deterrents_month" ~ 
    "Suicide deterrents (past month)",
  
  shap_plot_data$feature == "frequency_month" ~ 
    "SI frequency (past month)",
  
  shap_plot_data$feature == "internal_protective_factors_frustration_tolerance" ~ 
    "PF (internal): Frustration tolerance",
  
  shap_plot_data$feature ==
    "precipitants_stressors_chronic_physical_pain_or_other_acute_medical_problem_e_g_cns_disorders" ~
    "Stressor: Chronic pain / acute medical problem",
  
  shap_plot_data$feature == "internal_protective_factors_religious_beliefs" ~
    "PF (internal): Religious beliefs",
  
  shap_plot_data$feature == "when_you_have_the_thoughts_how_long_do_they_last" ~
    "SI duration (episode length)",
  
  shap_plot_data$feature == "presenting_symptoms_anhedonia_lack_of_pleasure" ~
    "Symptom: Anhedonia",
  
  shap_plot_data$feature == "are_there_things" ~ 
    "Protective factors present",
  
  shap_plot_data$feature == "adolescent" ~ 
    "Adolescent patient",
  
  shap_plot_data$feature == "internal_protective_factors_identifies_reasons_for_living" ~ 
    "PF (internal): Reasons for living",
  
  shap_plot_data$feature == "what_sort_of_reasons" ~ 
    "PF (internal): Reasons for living (details)",
  
  shap_plot_data$feature ==
    "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to" ~
    "Controllability of suicidal thoughts",
  
  shap_plot_data$feature == "motivational_interviewing" ~ 
    "MI received",
  
  shap_plot_data$feature == "mindfulness" ~ 
    "Mindfulness received",
  
  shap_plot_data$feature == "dx_group_Trauma_Related_Disorder" ~ 
    "Dx: Trauma-related disorder",
  
  shap_plot_data$feature ==
    "external_protective_factors_cultural_spiritual_and_or_moral_attitudes_against_suicide" ~
    "PF (external): Cultural/moral attitudes against suicide",
  
  shap_plot_data$feature == "family_history_none" ~ 
    "No family psychiatric history",
  
  TRUE ~ gsub("_", " ", shap_plot_data$feature)
)
# Normalize feature values for coloring (0-1 scale)
shap_plot_data <- shap_plot_data %>%
  group_by(feature) %>%
  mutate(feature_value_norm = (feature_value - min(feature_value, na.rm = TRUE)) /
           (max(feature_value, na.rm = TRUE) - min(feature_value, na.rm = TRUE) + 1e-10))

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
    title = "Feature importance for improvement prediction",
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
      title = "Therapy-specific associations with improvement",
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


figure <- cowplot::plot_grid(
  panel_a, panel_b + theme(legend.position = "none"),
  ncol = 1,
  rel_heights = c(1, 1),
  labels = c("A", "B"),
  label_size = 14,
  align = "v"
)


ggsave("outputs/figures/figure3.png", 
       figure, 
       width = 8 ,
       height = 8, 
       dpi = 300,
       bg = "white")

ggsave("outputs/figures/figure3.pdf", 
       figure, 
       width = 8 ,
       height = 8, 
       dpi = 300,
       bg = "white")

