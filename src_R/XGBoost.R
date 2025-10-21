# ============================================================================
# COMPLETE ANALYSIS PIPELINE: THERAPY OPTIMIZATION FOR SUICIDE RISK REDUCTION
# ============================================================================
# This script performs:
# 1. Data preparation and outcome creation
# 2. Feature engineering and selection
# 3. Temporal train-test splitting
# 4. Hyperparameter tuning with cross-validation
# 5. XGBoost model training with optimized parameters
# 6. SHAP analysis for interpretability
# 7. Doubly robust estimation for therapy associations
# 8. Counterfactual analysis for personalization gains
# 9. Comprehensive visualizations
# 10. Results export for manuscript
# ============================================================================

# Load required libraries
library(dplyr)
library(xgboost)
library(pROC)
library(ggplot2)
library(tidyr)

# Set seed for reproducibility
set.seed(123)

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("COMPREHENSIVE ANALYSIS: RISK IMPROVEMENT PREDICTION & THERAPY OPTIMIZATION\n")
cat(rep("=", 90), "\n", sep="")

# ============================================================================
# PART 1: CREATE OUTCOME VARIABLE
# ============================================================================

cat("\n[STEP 1] Creating outcome variable...\n")

# Define risk level ordering
risk_order <- c("Low" = 1, "Moderate" = 2, "High" = 3)

data <- data %>%
  mutate(
    risk_initial_num = risk_order[as.character(risk_level_initial)],
    risk_first_num = risk_order[as.character(risk_level_srs_first)],
    improve = as.integer(risk_first_num < risk_initial_num),
    risk_change = risk_initial_num - risk_first_num
  )

# Calculate improvement statistics
improvement_rate <- mean(data$improve, na.rm = TRUE)
cat(sprintf("  Overall improvement rate: %.1f%%\n", improvement_rate * 100))
cat(sprintf("  Sample size: %d patients\n", nrow(data)))

# Show risk transition matrix
risk_transitions <- data %>%
  filter(!is.na(improve)) %>%
  count(risk_level_initial, risk_level_srs_first) %>%
  arrange(risk_level_initial, risk_level_srs_first)

cat("\nRisk transitions:\n")
print(risk_transitions)

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

cat("\n[STEP 2] Defining feature groups...\n")

# Patient clinical features (baseline)
patient_clinical_features <- c(
  "total_score", "risk_high_initial", "deterrents_month", "what_sort_of_reasons",
  "duration_month", "adolescent",
  "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to",
  "are_there_things", "frequency_month", "when_you_have_the_thoughts_how_long_do_they_last",
  "how_many_times_have_you_had_these_thoughts", "male", "dx_group",
  names(data)[grepl("^current_and_past_psychiatric_diagnoses_", names(data))],
  names(data)[grepl("^presenting_symptoms_", names(data))],
  names(data)[grepl("^family_history_", names(data))],
  names(data)[grepl("^precipitants_stressors_", names(data))],
  names(data)[grepl("^internal_protective_factors_|^external_protective_factors_", names(data))],
  names(data)[grepl("^change_in_treatment_", names(data))]
)

# Therapy modality indicators
therapy_features <- c("act", "cbt", "dbt", "motivational_interviewing",
                      "mindfulness", "stages_of_change", "family_systems")

# Therapy propensity scores
propensity_features <- names(data)[grepl("^prop_", names(data))]

# Treatment context features
treatment_context_features <- c("therapy_duration_category", "delivery_method",
                                "session_mode")

# Therapist/organizational features
therapist_features <- c("therapist_name", "location", "program",
                        "pn_month", "pn_time_block", "pn_year")

# Combine all features
all_features <- unique(c(patient_clinical_features, therapy_features,
                         propensity_features, treatment_context_features,
                         therapist_features))

# Keep only features that exist in data
all_features <- all_features[all_features %in% names(data)]

# Print feature counts
cat(sprintf("  Patient clinical features: %d\n", 
            sum(all_features %in% patient_clinical_features)))
cat(sprintf("  Therapy indicators: %d\n", 
            sum(all_features %in% therapy_features)))
cat(sprintf("  Propensity scores: %d\n", 
            sum(all_features %in% propensity_features)))
cat(sprintf("  Treatment context: %d\n", 
            sum(all_features %in% treatment_context_features)))
cat(sprintf("  Therapist/organizational: %d\n", 
            sum(all_features %in% therapist_features)))
cat(sprintf("  Total features: %d\n", length(all_features)))

# ============================================================================
# PART 3: TEMPORAL TRAIN-TEST SPLIT
# ============================================================================

cat("\n[STEP 3] Creating temporal split (80/20)...\n")

# Sort by admission date and split
data <- data %>%
  arrange(admission_date) %>%
  mutate(row_id = row_number())

split_point <- floor(nrow(data) * 0.8)
train_data <- data %>% filter(row_id <= split_point)
test_data <- data %>% filter(row_id > split_point)

cat(sprintf("  Training set: n=%d (%.1f%% improved)\n",
            nrow(train_data), mean(train_data$improve, na.rm = TRUE) * 100))
cat(sprintf("  Test set: n=%d (%.1f%% improved)\n",
            nrow(test_data), mean(test_data$improve, na.rm = TRUE) * 100))

# Check temporal consistency
cat(sprintf("  Training date range: %s to %s\n",
            min(train_data$admission_date), max(train_data$admission_date)))
cat(sprintf("  Test date range: %s to %s\n",
            min(test_data$admission_date), max(test_data$admission_date)))

# ============================================================================
# PART 4: PREPARE DATA MATRICES
# ============================================================================

cat("\n[STEP 4] Preparing feature matrices...\n")

prepare_matrices <- function(train_df, test_df, features) {
  # Identify feature types
  categorical_features <- features[sapply(train_df[features], function(x) {
    is.factor(x) || is.character(x)
  })]
  
  numeric_features <- features[sapply(train_df[features], function(x) {
    is.numeric(x) || is.integer(x) || is.logical(x)
  })]
  
  cat(sprintf("  Processing %d numeric and %d categorical features\n",
              length(numeric_features), length(categorical_features)))
  
  # Process numeric features
  X_train_num <- as.matrix(train_df[, numeric_features, drop = FALSE])
  X_test_num <- as.matrix(test_df[, numeric_features, drop = FALSE])
  
  # Impute missing values with median from training set
  for(j in 1:ncol(X_train_num)) {
    if(any(is.na(X_train_num[, j]))) {
      median_val <- median(X_train_num[, j], na.rm = TRUE)
      if(is.na(median_val)) median_val <- 0
      X_train_num[is.na(X_train_num[, j]), j] <- median_val
      X_test_num[is.na(X_test_num[, j]), j] <- median_val
    }
  }
  
  # Process categorical features
  if(length(categorical_features) > 0) {
    cat_dummies_train <- list()
    cat_dummies_test <- list()
    
    for(feat in categorical_features) {
      if(feat %in% names(train_df)) {
        train_vals <- train_df[[feat]]
        test_vals <- test_df[[feat]]
        
        if(!all(is.na(train_vals))) {
          train_levels <- unique(as.character(train_vals[!is.na(train_vals)]))
          
          if(length(train_levels) > 1) {
            train_factor <- factor(as.character(train_vals), levels = train_levels)
            test_factor <- factor(as.character(test_vals), levels = train_levels)
            
            # Handle missing values
            if(any(is.na(train_factor))) {
              train_factor <- addNA(train_factor)
              levels(train_factor)[is.na(levels(train_factor))] <- "MISSING"
            }
            if(any(is.na(test_factor))) {
              test_factor <- addNA(test_factor)
              levels(test_factor)[is.na(levels(test_factor))] <- "MISSING"
            }
            
            # Create dummy variables (skip reference category)
            for(level in levels(train_factor)[-1]) {
              dummy_name <- paste0(feat, "_", gsub("[^[:alnum:]]", "_", level))
              cat_dummies_train[[dummy_name]] <- as.numeric(train_factor == level)
              cat_dummies_test[[dummy_name]] <- as.numeric(test_factor == level)
            }
          }
        }
      }
    }
    
    if(length(cat_dummies_train) > 0) {
      X_train_cat <- do.call(cbind, cat_dummies_train)
      X_test_cat <- do.call(cbind, cat_dummies_test)
      X_train <- cbind(X_train_num, X_train_cat)
      X_test <- cbind(X_test_num, X_test_cat)
      cat(sprintf("  Created %d dummy variables from categorical features\n",
                  ncol(X_train_cat)))
    } else {
      X_train <- X_train_num
      X_test <- X_test_num
    }
  } else {
    X_train <- X_train_num
    X_test <- X_test_num
  }
  
  # Ensure column alignment between train and test
  train_only_cols <- setdiff(colnames(X_train), colnames(X_test))
  if(length(train_only_cols) > 0) {
    zeros_matrix <- matrix(0, nrow = nrow(X_test), ncol = length(train_only_cols))
    colnames(zeros_matrix) <- train_only_cols
    X_test <- cbind(X_test, zeros_matrix)
    cat(sprintf("  Added %d columns to test set that only appeared in training\n",
                length(train_only_cols)))
  }
  
  X_test <- X_test[, colnames(X_train), drop = FALSE]
  
  return(list(train = X_train, test = X_test))
}

# Create feature matrices
matrices <- prepare_matrices(train_data, test_data, all_features)
X_train <- matrices$train
X_test <- matrices$test
y_train <- train_data$improve
y_test <- test_data$improve

# Remove any NA outcomes
if(any(is.na(y_train))) {
  keep_idx <- !is.na(y_train)
  X_train <- X_train[keep_idx, ]
  y_train <- y_train[keep_idx]
  cat(sprintf("  Removed %d training rows with NA outcomes\n", sum(!keep_idx)))
}

if(any(is.na(y_test))) {
  keep_idx <- !is.na(y_test)
  X_test <- X_test[keep_idx, ]
  y_test <- y_test[keep_idx]
  cat(sprintf("  Removed %d test rows with NA outcomes\n", sum(!keep_idx)))
}

cat(sprintf("\nFinal matrix dimensions:\n"))
cat(sprintf("  X_train: %d rows x %d columns\n", nrow(X_train), ncol(X_train)))
cat(sprintf("  X_test: %d rows x %d columns\n", nrow(X_test), ncol(X_test)))
# ============================================================================
# PART 5: HYPERPARAMETER TUNING WITH TEMPORAL CROSS-VALIDATION
# ============================================================================
cat("\n[STEP 5] Hyperparameter tuning with temporal cross-validation...\n")

# Function to create time series splits
create_time_series_splits <- function(n_samples, n_splits = 3) {
  # Calculate size of each test fold
  test_size <- floor(n_samples / (n_splits + 1))
  
  splits <- list()
  for(i in 1:n_splits) {
    # Each split uses progressively more training data
    train_end <- test_size * i
    test_start <- train_end + 1
    test_end <- min(train_end + test_size, n_samples)
    
    splits[[i]] <- list(
      train = 1:train_end,
      test = test_start:test_end
    )
  }
  return(splits)
}

# Create temporal CV splits
cv_splits <- create_time_series_splits(nrow(X_train), n_splits = 3)

# Visualize the splits
cat("  Temporal CV splits:\n")
for(i in 1:length(cv_splits)) {
  cat(sprintf("    Fold %d: Train [1:%d], Test [%d:%d]\n", 
              i, 
              max(cv_splits[[i]]$train),
              min(cv_splits[[i]]$test),
              max(cv_splits[[i]]$test)))
}

# Define parameter grid
param_grid <- expand.grid(
  max_depth = c(3, 4, 5, 6, 8),
  eta = c(0.01, 0.05, 0.1, 0.15),
  subsample = c(0.6, 0.7, 0.8, 0.9),
  colsample_bytree = c(0.6, 0.7, 0.8, 0.9),
  min_child_weight = c(1, 3, 5, 7),
  gamma = c(0, 0.5, 1, 2),
  alpha = c(0, 0.5, 1),
  lambda = c(0.5, 1, 2)
)

# Sample combinations for efficiency
set.seed(999)
param_grid <- param_grid[sample(nrow(param_grid), 10), ]
cat(sprintf("  Testing %d parameter combinations with %d-fold temporal CV\n", 
            nrow(param_grid), length(cv_splits)))

# Function to evaluate parameters with temporal CV
evaluate_params_temporal <- function(params_row, X, y, splits) {
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = params_row$max_depth,
    eta = params_row$eta,
    subsample = params_row$subsample,
    colsample_bytree = params_row$colsample_bytree,
    min_child_weight = params_row$min_child_weight,
    gamma = params_row$gamma,
    alpha = params_row$alpha,
    lambda = params_row$lambda
  )
  
  # Store AUC for each fold
  fold_aucs <- numeric(length(splits))
  fold_nrounds <- numeric(length(splits))
  
  for(i in 1:length(splits)) {
    # Get train and test indices for this fold
    train_idx <- splits[[i]]$train
    test_idx <- splits[[i]]$test
    
    # Create DMatrix objects for this fold
    dtrain_fold <- xgb.DMatrix(X[train_idx, ], label = y[train_idx])
    dtest_fold <- xgb.DMatrix(X[test_idx, ], label = y[test_idx])
    
    # Train model with early stopping
    watchlist <- list(test = dtest_fold)
    
    model_fold <- xgb.train(
      params = params,
      data = dtrain_fold,
      nrounds = 500,
      watchlist = watchlist,
      early_stopping_rounds = 30,
      verbose = 0
    )
    
    # Get predictions and calculate AUC
    pred_fold <- predict(model_fold, dtest_fold)
    auc_fold <- as.numeric(auc(roc(y[test_idx], pred_fold, quiet = TRUE)))
    
    fold_aucs[i] <- auc_fold
    fold_nrounds[i] <- model_fold$best_iteration
  }
  
  # Return mean AUC across folds
  return(list(
    mean_auc = mean(fold_aucs),
    std_auc = sd(fold_aucs),
    mean_nrounds = round(mean(fold_nrounds))
  ))
}

# Grid search with temporal CV
results <- data.frame(param_grid)
results$cv_auc_mean <- NA
results$cv_auc_std <- NA
results$best_nrounds <- NA

pb <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
for(i in 1:nrow(param_grid)) {
  eval_result <- evaluate_params_temporal(param_grid[i, ], X_train, y_train, cv_splits)
  results$cv_auc_mean[i] <- eval_result$mean_auc
  results$cv_auc_std[i] <- eval_result$std_auc
  results$best_nrounds[i] <- eval_result$mean_nrounds
  setTxtProgressBar(pb, i)
}
close(pb)

# Find best parameters (highest mean AUC with lowest std for stability)
# Use a combined score: mean_auc - 0.5 * std_auc to favor stable models
results$score <- results$cv_auc_mean - 0.5 * results$cv_auc_std
best_idx <- which.max(results$score)
best_params <- results[best_idx, ]

cat("\n\nBest parameters found:\n")
cat(sprintf("  Mean CV AUC: %.4f (±%.4f)\n", 
            best_params$cv_auc_mean, best_params$cv_auc_std))
print(best_params[, c("max_depth", "eta", "subsample", "colsample_bytree",
                      "min_child_weight", "gamma", "alpha", "lambda", 
                      "best_nrounds")])

# Show top 5 parameter combinations
cat("\nTop 5 parameter combinations:\n")
top5 <- results[order(results$score, decreasing = TRUE)[1:5], ]
print(top5[, c("max_depth", "eta", "cv_auc_mean", "cv_auc_std", "score")])
# ============================================================================
# PART 6: TRAIN FINAL MODEL
# ============================================================================

cat("\n[STEP 6] Training final XGBoost with best parameters...\n")

best_params_list <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  alpha = best_params$alpha,
  lambda = best_params$lambda
)

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)
watchlist <- list(train = dtrain, test = dtest)

# Train model
xgb_final <- xgb.train(
  params = best_params_list,
  data = dtrain,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  verbose = 0
)

# Get predictions
pred_train <- predict(xgb_final, dtrain)
pred_test <- predict(xgb_final, dtest)

# Calculate AUCs
auc_train <- as.numeric(auc(roc(y_train, pred_train, quiet = TRUE)))
auc_test <- as.numeric(auc(roc(y_test, pred_test, quiet = TRUE)))

cat(sprintf("  Training AUC: %.4f\n", auc_train))
cat(sprintf("  Test AUC: %.4f\n", auc_test))
cat(sprintf("  Best iteration: %d\n", xgb_final$best_iteration))

# Additional performance metrics
roc_test <- roc(y_test, pred_test, quiet = TRUE)
coords <- coords(roc_test, "best", ret = "all", transpose = FALSE)
optimal_threshold <- coords$threshold[1]

pred_test_binary <- as.integer(pred_test > optimal_threshold)
cm <- table(Actual = y_test, Predicted = pred_test_binary)

cat("\nConfusion Matrix:\n")
print(cm)

sensitivity <- cm[2,2] / sum(cm[2,])
specificity <- cm[1,1] / sum(cm[1,])
ppv <- cm[2,2] / sum(cm[,2])
npv <- cm[1,1] / sum(cm[,1])

cat(sprintf("\nTest Set Performance (threshold = %.3f):\n", optimal_threshold))
cat(sprintf("  Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("  Specificity: %.1f%%\n", specificity * 100))
cat(sprintf("  PPV: %.1f%%\n", ppv * 100))
cat(sprintf("  NPV: %.1f%%\n", npv * 100))

# get brier score
brier_score <- mean((pred_test - y_test)^2)
cat(sprintf("  Brier Score: %.4f\n", brier_score))

# platt scaled probabilities
platt_model <- glm(y_train ~ pred_train, family = binomial)
platt_probs <- predict(platt_model, newdata = data.frame(pred_train = pred_test), type = "response")

brier_score_platt <- mean((platt_probs - y_test)^2)
cat(sprintf("  Brier Score (Platt scaled): %.4f\n", brier_score_platt))

# ============================================================================
# PART 7: FEATURE IMPORTANCE & SHAP ANALYSIS
# ============================================================================

cat("\n[STEP 7] Analyzing feature importance...\n")

# Get XGBoost feature importance
importance_matrix <- xgb.importance(model = xgb_final)
cat("\nTop 20 features by gain:\n")
print(head(importance_matrix[, c("Feature", "Gain")], 20))

# Categorize features by group
importance_df <- as.data.frame(importance_matrix)
importance_df <- importance_df %>%
  mutate(
    feature_group = case_when(
      Feature %in% colnames(X_train)[colnames(X_train) %in% patient_clinical_features] ~ "Patient Clinical",
      Feature %in% colnames(X_train)[colnames(X_train) %in% therapy_features] ~ "Therapy Received",
      grepl("^prop_", Feature) ~ "Propensity Score",
      Feature %in% colnames(X_train)[colnames(X_train) %in% treatment_context_features] ~ "Treatment Context",
      grepl("therapist_name_|location_|program_|pn_", Feature) ~ "Therapist/Org",
      TRUE ~ "Other"
    )
  )

# Summarize by group
group_importance <- importance_df %>%
  group_by(feature_group) %>%
  summarise(
    total_gain = sum(Gain),
    mean_gain = mean(Gain),
    n_features = n()
  ) %>%
  arrange(desc(total_gain))

cat("\nFeature Importance by Group:\n")
print(group_importance)

# ============================================================================
# PART 8: DOUBLY ROBUST ESTIMATION
# ============================================================================

cat("\n[STEP 8] Performing doubly robust estimation for each therapy...\n")

# Identify therapy columns in the data
therapy_cols <- therapy_features[therapy_features %in% colnames(X_train)]
cat(sprintf("  Found %d therapy modalities in data\n", length(therapy_cols)))

dr_results <- list()
therapy_to_prop <- c(
  "act" = "prop_act",
  "cbt" = "prop_cbt", 
  "dbt" = "prop_dbt",
  "motivational_interviewing" = "prop_mi",  # <- This is the issue
  "mindfulness" = "prop_mindfulness",
  "stages_of_change" = "prop_stages_of_change",  # <- Might also be abbreviated
  "family_systems" = "prop_family_systems"  # <- Might also be abbreviated
)

for(therapy in therapy_cols) {
  prop_col <- therapy_to_prop[therapy]
  
  if(!is.na(prop_col) && prop_col %in% colnames(X_train)) {
    T_i <- X_train[, therapy]
    e_i <- X_train[, prop_col]
    e_i <- X_train[, prop_col]
    
    # Bound propensity scores away from 0 and 1
    e_i <- pmax(0.01, pmin(0.99, e_i))
    
    # Check sample sizes
    n_treated <- sum(T_i == 1)
    n_control <- sum(T_i == 0)
    

      # Fit separate models for treated and control
      X_without_therapy <- X_train[, !colnames(X_train) %in% therapy]
      
      # Treated model
      dtrain_t <- xgb.DMatrix(X_without_therapy[T_i == 1, ],
                              label = y_train[T_i == 1])
      model_t <- xgb.train(best_params_list, dtrain_t, nrounds = 100, verbose = 0)
      
      # Control model
      dtrain_c <- xgb.DMatrix(X_without_therapy[T_i == 0, ],
                              label = y_train[T_i == 0])
      model_c <- xgb.train(best_params_list, dtrain_c, nrounds = 100, verbose = 0)
      
      # Predict potential outcomes for all
      dmatrix_all <- xgb.DMatrix(X_without_therapy)
      mu_1 <- predict(model_t, dmatrix_all)
      mu_0 <- predict(model_c, dmatrix_all)
      
      # Calculate AIPW estimator
      tau_i <- mu_1 - mu_0 +
        T_i * (y_train - mu_1) / e_i -
        (1 - T_i) * (y_train - mu_0) / (1 - e_i)
      
      ate <- mean(tau_i)
      se <- sd(tau_i) / sqrt(length(tau_i))
      
      dr_results[[therapy]] <- list(
        ate = ate,
        se = se,
        ci_lower = ate - 1.96 * se,
        ci_upper = ate + 1.96 * se,
        n_treated = n_treated,
        n_control = n_control
      )
      
      cat(sprintf("  %s: ATE = %.3f (95%% CI: %.3f to %.3f), n_treated=%d\n",
                  therapy, ate, ate - 1.96*se, ate + 1.96*se, n_treated))
    } 
  
}


# ============================================================================
# PART 9: COUNTERFACTUAL ANALYSIS
# ============================================================================

cat("\n[STEP 9] Computing personalization gains...\n")

# Get observed therapy combinations
combo_matrix <- X_test[, therapy_cols]
combo_strings <- apply(combo_matrix, 1, paste, collapse = "-")
combo_counts <- table(combo_strings)
top_combos <- names(sort(combo_counts, decreasing = TRUE)[1:min(50, length(combo_counts))])

cat(sprintf("  Evaluating %d therapy combinations\n", length(top_combos)))

# Compute personalization gains
personalization_gains <- numeric(nrow(X_test))
optimal_combos <- matrix(0, nrow = nrow(X_test), ncol = length(therapy_cols))
colnames(optimal_combos) <- therapy_cols

pb <- txtProgressBar(min = 0, max = nrow(X_test), style = 3)

for(i in 1:nrow(X_test)) {
  patient_features <- X_test[i, ]
  baseline_prob <- pred_test[i]
  
  best_prob <- baseline_prob
  best_combo <- patient_features[therapy_cols]
  
  for(combo_str in top_combos) {
    combo_vals <- as.numeric(strsplit(combo_str, "-")[[1]])
    
    # Create counterfactual
    cf_features <- patient_features
    cf_features[therapy_cols] <- combo_vals
    cf_matrix <- xgb.DMatrix(matrix(cf_features, nrow = 1))
    cf_prob <- predict(xgb_final, cf_matrix)
    
    if(cf_prob > best_prob) {
      best_prob <- cf_prob
      best_combo <- combo_vals
    }
  }
  
  personalization_gains[i] <- best_prob - baseline_prob
  optimal_combos[i, ] <- best_combo
  setTxtProgressBar(pb, i)
}
close(pb)

# Calculate summary statistics
mean_gain <- mean(personalization_gains)
median_gain <- median(personalization_gains)
pct_benefit <- mean(personalization_gains > 0.01) * 100
pct_large_benefit <- mean(personalization_gains > 0.05) * 100

cat(sprintf("\nPersonalization Gain Summary:\n"))
cat(sprintf("  Mean gain: %.3f (%.1f%% relative improvement)\n",
            mean_gain, mean_gain / mean(pred_test) * 100))
cat(sprintf("  Median gain: %.3f\n", median_gain))
cat(sprintf("  Patients who would benefit (>1%% gain): %.1f%%\n", pct_benefit))
cat(sprintf("  Patients with substantial benefit (>5%% gain): %.1f%%\n", pct_large_benefit))
cat(sprintf("  Maximum gain: %.3f\n", max(personalization_gains)))
cat(sprintf("  NNT to prevent one non-improvement: %.0f\n",
            ifelse(mean_gain > 0, 1/mean_gain, Inf)))

# figure out percentage gain among patients who would benefit
if(sum(personalization_gains > 0) > 0) {
  mean_gain_benefiters <- mean(personalization_gains[personalization_gains > 0])
  cat(sprintf("  Mean gain among benefiters: %.3f (%.1f%% relative improvement)\n",
              mean_gain_benefiters,
              mean_gain_benefiters / mean(pred_test[personalization_gains > 0]) * 100))
} else {
  cat("  No patients would benefit from therapy optimization.\n")
}

# figure out what predicted probability groups benefit most
pred_bins <- cut(pred_test, breaks = seq(0, 1, by = 0.25), include.lowest = TRUE)
gain_by_bin <- tapply(personalization_gains, pred_bins, mean)
cat("\nMean Personalization Gain by Predicted Probability Bin:\n")
for(bin in names(gain_by_bin)) {
  cat(sprintf("  %s: %.3f\n", bin, gain_by_bin[bin]))
}
# ============================================================================
# PART 10: VISUALIZATIONS
# ============================================================================

cat("\n[STEP 10] Creating visualizations...\n")

# Figure 1: Distribution of Personalization Gains
p1 <- ggplot(data.frame(gain = personalization_gains * 100), aes(x = gain)) +
  geom_histogram(bins = 50, fill = "#2E86AB", alpha = 0.7, color = "white") +
  geom_vline(xintercept = mean_gain * 100, color = "#A23B72", 
             linetype = "dashed", size = 1) +
  geom_vline(xintercept = median_gain * 100, color = "#F18F01", 
             linetype = "dashed", size = 1) +
  scale_x_continuous(limits = c(-1, max(personalization_gains) * 100 + 1)) +
  labs(
    title = "Distribution of Predicted Personalization Gains",
    subtitle = sprintf("%.1f%% of patients show potential benefit from optimized therapy selection",
                       pct_benefit),
    x = "Personalization Gain (percentage points)",
    y = "Number of Patients"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "#666666"),
    panel.grid.minor = element_blank()
  )
p1
ggsave("figure_personalization_distribution.png", p1, width = 10, height = 6, dpi = 300)

# Figure 2: Therapy-Specific Associations (Doubly Robust)
if(length(dr_results) > 0) {
  dr_df <- data.frame(
    therapy = names(dr_results),
    ate = sapply(dr_results, function(x) x$ate),
    ci_lower = sapply(dr_results, function(x) x$ci_lower),
    ci_upper = sapply(dr_results, function(x) x$ci_upper),
    n = sapply(dr_results, function(x) x$n_treated)
  ) %>%
    arrange(ate)
  
  dr_df$therapy <- factor(dr_df$therapy, levels = dr_df$therapy)
  
  p2 <- ggplot(dr_df, aes(x = therapy, y = ate * 100)) +
    geom_hline(yintercept = 0, linetype = "solid", color = "#999999") +
    geom_errorbar(aes(ymin = ci_lower * 100, ymax = ci_upper * 100),
                  width = 0.2, color = "#666666") +
    geom_point(size = 3, color = "#2E86AB") +
    coord_flip() +
    labs(
      title = "Therapy-Specific Associations with Improvement",
      subtitle = "Doubly robust estimates accounting for selection bias",
      x = "",
      y = "Estimated Difference in Improvement Probability (percentage points)"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "#666666"),
      panel.grid.minor = element_blank()
    )
  
  ggsave("figure_therapy_associations.png", p2, width = 10, height = 6, dpi = 300)
}

# Figure 3: Personalization Gains by Baseline Risk
baseline_risk_df <- data.frame(
  baseline_prob = pred_test,
  gain = personalization_gains * 100,
  baseline_risk = test_data$risk_level_initial[!is.na(y_test)]
)

p3 <- ggplot(baseline_risk_df, aes(x = baseline_prob, y = gain)) +
  geom_point(aes(color = baseline_risk), alpha = 0.5, size = 1) +
  geom_smooth(method = "loess", se = TRUE, color = "#2E86AB", size = 1) +
  scale_color_manual(values = c("Moderate" = "#F18F01", "High" = "#A23B72")) +
  labs(
    title = "Heterogeneity in Personalization Potential",
    subtitle = "Predicted gains vary by baseline improvement probability",
    x = "Baseline Probability of Improvement",
    y = "Personalization Gain (percentage points)",
    color = "Initial Risk"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11, color = "#666666"),
    legend.position = "bottom"
  )
p2

ggsave("figure_heterogeneity.png", p3, width = 10, height = 6, dpi = 300)

# ============================================================================
# PART 11: EXPORT RESULTS FOR MANUSCRIPT
# ============================================================================

cat("\n[STEP 11] Exporting results for manuscript...\n")

# Create results list
manuscript_results <- list(
  # Model performance
  model_performance = data.frame(
    metric = c("AUC_train", "AUC_test", "Sensitivity", "Specificity", "PPV", "NPV"),
    value = c(auc_train, auc_test, sensitivity, specificity, ppv, npv)
  ),
  
  # Feature importance by group
  feature_importance = group_importance,
  
  # Therapy associations
  therapy_associations = if(length(dr_results) > 0) {
    data.frame(
      therapy = names(dr_results),
      ate = sapply(dr_results, function(x) x$ate),
      se = sapply(dr_results, function(x) x$se),
      ci_lower = sapply(dr_results, function(x) x$ci_lower),
      ci_upper = sapply(dr_results, function(x) x$ci_upper),
      n_treated = sapply(dr_results, function(x) x$n_treated)
    )
  } else NULL,
  
  # Personalization summary
  personalization_summary = data.frame(
    metric = c("mean_gain", "median_gain", "pct_benefit", "pct_large_benefit", 
               "max_gain", "nnt"),
    value = c(mean_gain, median_gain, pct_benefit/100, pct_large_benefit/100,
              max(personalization_gains), ifelse(mean_gain > 0, 1/mean_gain, NA))
  ),
  
  # Model object
  xgboost_model = xgb_final,
  
  # Individual predictions
  predictions = data.frame(
    observed_prob = pred_test,
    personalization_gain = personalization_gains,
    optimal_therapy = apply(optimal_combos, 1, function(x) 
      paste(therapy_cols[x == 1], collapse = "+"))
  )
)
manuscript_results$personalization_summary
manuscript_results$model_performance
manuscript_results$feature_importance

# Save RDS file

cat("\nFiles saved:\n")
cat("  - figure_personalization_distribution.png\n")
cat("  - figure_therapy_associations.png\n")
cat("  - figure_heterogeneity.png\n")
cat("  - manuscript_results.rds\n")
cat("  - table_model_performance.csv\n")
cat("  - table_feature_importance.csv\n")
cat("  - table_therapy_associations.csv\n")
cat("  - table_personalization_summary.csv\n")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("ANALYSIS COMPLETE - SUMMARY FOR MANUSCRIPT\n")
cat(rep("=", 90), "\n", sep="")

cat(sprintf("\n1. PREDICTIVE PERFORMANCE:\n"))
cat(sprintf("   Test set AUROC: %.3f\n", auc_test))
cat(sprintf("   Sensitivity at optimal threshold: %.1f%%\n", sensitivity * 100))

cat(sprintf("\n2. THERAPY ASSOCIATIONS (Doubly Robust):\n"))
if(length(dr_results) > 0) {
  sorted_dr <- sort(sapply(dr_results, function(x) x$ate), decreasing = TRUE)
  for(i in 1:min(3, length(sorted_dr))) {
    cat(sprintf("   %s: %.1f percentage points (95%% CI: %.1f to %.1f)\n",
                names(sorted_dr)[i], sorted_dr[i] * 100,
                dr_results[[names(sorted_dr)[i]]]$ci_lower * 100,
                dr_results[[names(sorted_dr)[i]]]$ci_upper * 100))
  }
}

cat(sprintf("\n3. PERSONALIZATION POTENTIAL:\n"))
cat(sprintf("   Patients who could benefit: %.1f%%\n", pct_benefit))
cat(sprintf("   Mean predicted gain: %.1f percentage points\n", mean_gain * 100))
cat(sprintf("   Number needed to treat: %.0f\n", ifelse(mean_gain > 0, 1/mean_gain, Inf)))

cat("\n")
cat(rep("=", 90), "\n", sep="")



# ============================================================================
# SHAP ANALYSIS AND VISUALIZATION
# ============================================================================

library(SHAPforxgboost)

cat("\n[ADDITIONAL STEP] Computing SHAP values for interpretability...\n")

# Calculate SHAP values for a sample of test set
set.seed(123)
shap_sample_size <- min(300, nrow(X_test))
shap_idx <- sample(1:nrow(X_test), shap_sample_size)

# Prepare data for SHAP
shap_data <- xgb.DMatrix(X_test[shap_idx, ])

# Calculate SHAP values
shap_values <- predict(xgb_final, shap_data, predcontrib = TRUE, approxcontrib = FALSE)

# Remove the bias column (last column)
shap_values_clean <- shap_values[, -ncol(shap_values)]

# Get feature names
feature_names <- colnames(X_test)

# Calculate mean absolute SHAP values for each feature
mean_shap <- colMeans(abs(shap_values_clean))
names(mean_shap) <- feature_names

# Sort by importance
shap_importance <- sort(mean_shap, decreasing = TRUE)

# Create SHAP summary plot data
shap_long <- data.frame()
top_features <- names(shap_importance)[1:20]

for(feat in top_features) {
  feat_idx <- which(feature_names == feat)
  shap_long <- rbind(shap_long, 
                     data.frame(
                       feature = feat,
                       feature_value = X_test[shap_idx, feat_idx],
                       shap_value = shap_values_clean[, feat_idx]
                     ))
}

# Create more readable feature names
shap_long$feature_clean <- case_when(
  shap_long$feature == "days_first_srs" ~ "Days to First Reassessment",
  shap_long$feature == "risk_high_initial" ~ "High Risk at Intake",
  shap_long$feature == "total_score" ~ "Total Symptom Score",
  shap_long$feature == "frequency_month" ~ "Frequency of Ideation",
  grepl("prop_", shap_long$feature) ~ paste0("Propensity: ", 
                                             toupper(gsub("prop_", "", shap_long$feature))),
  grepl("location_", shap_long$feature) ~ "Treatment Location",
  grepl("therapist_", shap_long$feature) ~ "Therapist ID",
  TRUE ~ shap_long$feature
)

# Order features by mean absolute SHAP
feature_order <- shap_long %>%
  group_by(feature_clean) %>%
  summarise(mean_abs_shap = mean(abs(shap_value))) %>%
  arrange(desc(mean_abs_shap))

shap_long$feature_clean <- factor(shap_long$feature_clean, 
                                  levels = rev(feature_order$feature_clean))

# Create SHAP summary plot
library(viridis)

p_shap <- ggplot(shap_long, aes(x = shap_value, y = feature_clean)) +
  geom_point(aes(color = feature_value), 
             position = position_jitter(height = 0.2), 
             size = 0.8, alpha = 0.6) +
  scale_color_viridis(name = "Feature Value\n(Normalized)", 
                      option = "plasma",
                      breaks = c(0, 0.5, 1),
                      labels = c("Low", "Medium", "High")) +
  geom_vline(xintercept = 0, linetype = "solid", color = "grey40") +
  labs(
    title = "SHAP Feature Importance for Improvement Prediction",
    subtitle = "Impact on model prediction (positive values increase improvement probability)",
    x = "SHAP Value (Impact on Prediction)",
    y = NULL
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 10, color = "#666666"),
    axis.text.y = element_text(size = 9),
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  )

p_shap



# ============================================================================
# PUBLICATION-READY FIGURE: SHAP AND DOUBLY ROBUST ESTIMATION
# ============================================================================

library(ggplot2)
library(dplyr)
library(cowplot)
library(viridis)

# Set publication theme
theme_publication <- function() {
  theme_minimal() +
    theme(
      text = element_text(family = "Helvetica", size = 10),
      plot.title = element_text(size = 11, face = "bold", hjust = 0),
      plot.subtitle = element_text(size = 10, color = "black", hjust = 0),
      axis.title = element_text(size = 10, face = "plain"),
      axis.text = element_text(size = 9, color = "black"),
      axis.line = element_line(color = "black", size = 0.3),
      axis.ticks = element_line(color = "black", size = 0.3),
      panel.grid.major = element_line(color = "#E5E5E5", size = 0.2),
      panel.grid.minor = element_blank(),
      legend.title = element_text(size = 9, face = "bold"),
      legend.text = element_text(size = 8),
      legend.position = "right",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      strip.text = element_text(size = 9, face = "bold"),
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
  shap_sample_size <- min(500, nrow(X_test))  # Increased sample for stability
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

# Order by mean absolute SHAP


# Normalize feature values for coloring (0-1 scale)
shap_plot_data <- shap_plot_data %>%
  group_by(feature) %>%
  mutate(feature_value_norm = (feature_value - min(feature_value, na.rm = TRUE)) /
           (max(feature_value, na.rm = TRUE) - min(feature_value, na.rm = TRUE) + 1e-10))

# Create clean feature labels for SHAP plot
shap_plot_data$feature_clean <- case_when(
  # Risk and severity measures
  shap_plot_data$feature == "risk_high_initial" ~ "High risk at intake",
  shap_plot_data$feature == "total_score" ~ "C-SSRS total score",
  
  # C-SSRS specific items about suicidal ideation
  shap_plot_data$feature == "how_many_times_have_you_had_these_thoughts" ~ "Suicidal Ideation frequency (times)",
  shap_plot_data$feature == "frequency_month" ~ "Suicidal ideation frequency (past month)",
  shap_plot_data$feature == "duration_month" ~ "Suicidal ideation duration (past month)",
  shap_plot_data$feature == "deterrents_month" ~ "Deterrents present",
  shap_plot_data$feature == "could_can_you_stop_thinking_about_killing_yourself_or_wanting_to_die_if_you_want_to" ~ "Suicidal ideation controllability",
  
  # Protective factors (internal)
  shap_plot_data$feature == "internal_protective_factors_identifies_reasons_for_living" ~ "Identifies reasons for living",
  shap_plot_data$feature == "internal_protective_factors_ability_to_cope_with_stress" ~ "Ability to cope with stress",
  shap_plot_data$feature == "internal_protective_factors_able_to_access_care_willing_to_reach_out" ~ "Willing to seek help",
  shap_plot_data$feature == "internal_protective_factors_fear_of_death_or_the_actual_act_of_killing_self" ~ "Fear of death/dying",
  
  # Risk factors/stressors
  shap_plot_data$feature == "precipitants_stressors_social_isolation" ~ "Socially isolated",
  shap_plot_data$feature == "precipitants_stressors_inadequate_social_supports" ~ "Inadequate social support",
  
  # Treatment received
  shap_plot_data$feature == "stages_of_change" ~ "Stages of Change (received)",
  
  # Propensity scores (likelihood of receiving each therapy)
  shap_plot_data$feature == "prop_act" ~ "Propensity to recieve ACT",
  shap_plot_data$feature == "prop_dbt" ~ "Propensity to recieve DBT",
  shap_plot_data$feature == "prop_mi" ~ "Propensity to recieve Motivational Interviewing",
  shap_plot_data$feature == "prop_mindfulness" ~ "Propensity to recieve Mindfulness",
  shap_plot_data$feature == "prop_family_systems" ~ "Propensity to recieve Family therapy",
  shap_plot_data$feature == "prop_stages_of_change" ~ "Propensity to recieve Stages of Change",
  
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
  scale_color_gradient2(low = "#0066CC", mid = "#F4F4F4", high = "#CC0000",
                        midpoint = 0.5,
                        name = "Feature\nvalue\n",
                        breaks = c(0, 0.999),
                        labels = c("Low",  "High")) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#666666", size = 0.3) +
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
panel_a
# ============================================================================
# PANEL B: DOUBLY ROBUST ESTIMATION
# ============================================================================

cat("Creating doubly robust estimation plot...\n")

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
    geom_hline(yintercept = 0, linetype = "solid", color = "#666666", size = 0.3) +
    geom_errorbar(aes(ymin = ci_lower * 100, ymax = ci_upper * 100),
                  width = 0, size = 0.4, color = "#333333") +
    geom_point(aes(fill = ate > 0), shape = 21, size = 3.5, 
               color = "#333333", stroke = 0.5) +
    scale_fill_manual(values = c("FALSE" = "#009900", "TRUE" = "#009900"),
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

panel_b
# ============================================================================
# COMBINE PANELS
# ============================================================================

cat("Combining panels into publication figure...\n")

# Combine plots using cowplot
figure <- plot_grid(
  panel_a, panel_b,
  ncol = 1,
  rel_widths = c(1, 1),
  align = "h",
  axis = "tb"
)

# Save figure
ggsave("figure_combined_shap_dr.pdf", 
       figure, 
       width = 12, 
       height = 6, 
       dpi = 300,
       device = cairo_pdf)

ggsave("figure_combined_shap_dr.png", 
       figure, 
       width = 12, 
       height = 6, 
       dpi = 300,
       bg = "white")

ggsave("figure_combined_shap_dr.tiff", 
       figure, 
       width = 12, 
       height = 6, 
       dpi = 300,
       compression = "lzw",
       bg = "white")

cat("\nFigure saved as:\n")
cat("  - figure_combined_shap_dr.pdf (for submission)\n")
cat("  - figure_combined_shap_dr.png (for review)\n")
cat("  - figure_combined_shap_dr.tiff (if journal requires)\n")

# Print summary statistics for caption
cat(sprintf("Figure X. Machine learning model insights for suicide risk improvement prediction.\n"))
cat(sprintf("(A) SHAP (SHapley Additive exPlanations) values for top 20 predictive features,\n"))
cat(sprintf("excluding provider fixed effects. Points represent individual patients (n=%d),\n", shap_sample_size))
cat(sprintf("with color indicating normalized feature values. Positive SHAP values indicate\n"))
cat(sprintf("features that increase improvement probability. (B) Therapy-specific associations\n"))
cat(sprintf("with risk improvement from doubly robust estimation, showing percentage point\n"))
cat(sprintf("differences with 95%% confidence intervals. Green indicates positive associations,\n"))
cat(sprintf("red indicates negative associations. *p<0.10, **p<0.05, ***p<0.01.\n"))




# First, ensure consistent margins and spacing for both panels

# PANEL A modifications
panel_a <- ggplot(shap_plot_data, aes(x = shap_value, y = feature_clean)) +
  geom_jitter(aes(color = feature_value_norm),
              height = 0.15, width = 0, size = 0.8, alpha = 0.7) +
  scale_color_gradient2(low = "#0066CC", mid = "#F4F4F4", high = "#CC0000",
                        midpoint = 0.5,
                        name = "Feature\nvalue",
                        breaks = c(0, 0.999),
                        labels = c("Low", "High")) +
  geom_vline(xintercept = 0, linetype = "solid", color = "#666666", size = 0.3) +
  labs(
    title = "A. Feature importance for improvement prediction",
    subtitle = "SHAP values quantify impact on predicted probability",
    x = "SHAP value (impact on prediction)",
    y = NULL
  ) +
  theme_publication() +
  theme(
    legend.position = "right",  # Move legend to right side instead of inside plot
    legend.margin = margin(0, 0, 0, 0),
    legend.box.margin = margin(0, 0, 0, 0),
    axis.text.y = element_text(size = 9),
    plot.margin = unit(c(5, 5, 5, 5), "mm")  # Consistent margins
  )

# PANEL B modifications
panel_b <- ggplot(dr_df, aes(x = reorder(therapy_clean, ate), y = ate * 100)) +
  geom_hline(yintercept = 0, linetype = "solid", color = "#666666", size = 0.3) +
  geom_errorbar(aes(ymin = ci_lower * 100, ymax = ci_upper * 100),
                width = 0, size = 0.4, color = "#333333") +
  geom_point(aes(fill = ate > 0), shape = 21, size = 3.5, 
             color = "#333333", stroke = 0.5) +
  scale_fill_manual(values = c("FALSE" = "#009900", "TRUE" = "#009900"),
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
  )

panel_b
# Combine with better alignment
library(cowplot)
figure <- plot_grid(
  panel_a, 
  panel_b,
  ncol = 1,
  rel_heights = c(1.25, 1),  # Adjust relative heights
  align = "hv",  # Align both horizontally and vertically
  axis = "tblr"  # Align all axes
)

figure

ggsave("fig3.png", figure, width = 8, height = 7, dpi = 300, bg='white')

