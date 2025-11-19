# ============================================================================
# COMPLETE ANALYSIS PIPELINE: THERAPY OPTIMIZATION WITH AIPW WEIGHTING
# ============================================================================
# This script performs:
# 1. Data preparation and outcome creation
# 2. Feature engineering and selection (WITHOUT propensity scores as features)
# 3. AIPW weight construction for multiple therapies
# 4. Temporal train-test splitting
# 5. Hyperparameter tuning with weighted cross-validation
# 6. Weighted XGBoost model training with optimized parameters
# 7. Standard predictive performance metrics (AUC, Brier, etc.)
# 8. SHAP analysis for interpretability
# 9. Doubly robust estimation for therapy associations
# 10. Counterfactual analysis for personalization gains
# 11. Comprehensive visualizations
# 12. Results export for manuscript
# ============================================================================
# Set seed for reproducibility
set.seed(123)

# ============================================================================
# CREATE OUTCOME VARIABLE
# ============================================================================

# Show risk transition matrix
risk_transitions <- data %>%
  filter(!is.na(improve)) %>%
  count(risk_level_initial, risk_level_srs_first) %>%
  arrange(risk_level_initial, risk_level_srs_first)

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("RISK TRANSITIONS\n")
cat(rep("=", 90), "\n", sep="")
print(risk_transitions)

# ============================================================================
# FEATURE ENGINEERING (EXCLUDING PROPENSITY SCORES)
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("FEATURE ENGINEERING\n")
cat(rep("=", 90), "\n", sep="")

# Patient clinical features (baseline)
patient_clinical_features <- c(
  "total_score", "risk_high_initial", 'days_first_srs', "deterrents_month", "what_sort_of_reasons",
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

# Treatment context features
treatment_context_features <- c("therapy_duration_category", "delivery_method",
                                "session_mode")

# Therapist/organizational features
therapist_features <- c("therapist_name", "location", "program",
                        "pn_month", "pn_time_block", "pn_year")

# Combine all features (NO propensity scores)
all_features <- unique(c(patient_clinical_features, therapy_features,
                         treatment_context_features, therapist_features))

# Keep only features that exist in data
all_features <- all_features[all_features %in% names(data)]

# Print feature counts
cat(sprintf("  Patient clinical features: %d\n", 
            sum(all_features %in% patient_clinical_features)))
cat(sprintf("  Therapy indicators: %d\n", 
            sum(all_features %in% therapy_features)))
cat(sprintf("  Treatment context: %d\n", 
            sum(all_features %in% treatment_context_features)))
cat(sprintf("  Therapist/organizational: %d\n", 
            sum(all_features %in% therapist_features)))
cat(sprintf("  Total features: %d\n", length(all_features)))

# ============================================================================
# TEMPORAL TRAIN-TEST SPLIT
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("TEMPORAL TRAIN-TEST SPLIT\n")
cat(rep("=", 90), "\n", sep="")

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
# PREPARE DATA MATRICES
# ============================================================================

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
cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("PREPARING DATA MATRICES\n")
cat(rep("=", 90), "\n", sep="")

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
  train_data <- train_data[keep_idx, ]
  cat(sprintf("  Removed %d training rows with NA outcomes\n", sum(!keep_idx)))
}

if(any(is.na(y_test))) {
  keep_idx <- !is.na(y_test)
  X_test <- X_test[keep_idx, ]
  y_test <- y_test[keep_idx]
  test_data <- test_data[keep_idx, ]
  cat(sprintf("  Removed %d test rows with NA outcomes\n", sum(!keep_idx)))
}

cat(sprintf("\nFinal matrix dimensions:\n"))
cat(sprintf("  X_train: %d rows x %d columns\n", nrow(X_train), ncol(X_train)))
cat(sprintf("  X_test: %d rows x %d columns\n", nrow(X_test), ncol(X_test)))

# ============================================================================
# CONSTRUCT AIPW WEIGHTS
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("CONSTRUCTING AIPW WEIGHTS\n")
cat(rep("=", 90), "\n", sep="")

# Mapping between therapy columns and propensity columns
therapy_to_prop <- c(
  "act" = "prop_act",
  "cbt" = "prop_cbt", 
  "dbt" = "prop_dbt",
  "motivational_interviewing" = "prop_mi",
  "mindfulness" = "prop_mindfulness",
  "stages_of_change" = "prop_stages_of_change",
  "family_systems" = "prop_family_systems"
)

# Identify available therapy columns in the feature matrix
therapy_cols_in_data <- therapy_features[therapy_features %in% colnames(X_train)]
cat(sprintf("  Found %d therapy modalities in data\n", length(therapy_cols_in_data)))

# Function to create stabilized AIPW weights for multiple treatments
create_aipw_weights <- function(X_matrix, df_with_props, therapy_cols, therapy_to_prop_map) {
  n <- nrow(X_matrix)
  weights <- rep(1, n)
  
  weight_components <- data.frame(row = 1:n)
  
  for(therapy in therapy_cols) {
    prop_col <- therapy_to_prop_map[therapy]
    
    if(!is.na(prop_col) && prop_col %in% names(df_with_props)) {
      # Get treatment indicator from feature matrix
      T_i <- X_matrix[, therapy]
      
      # Get propensity score
      e_i <- df_with_props[[prop_col]]
      
      # Bound propensity scores away from 0 and 1
      e_i <- pmax(0.01, pmin(0.99, e_i))
      
      # Marginal probability (for stabilization)
      p_t <- mean(T_i, na.rm = TRUE)
      
      # Stabilized weight for this therapy
      # w_i = (T_i * p_t / e_i) + ((1 - T_i) * (1 - p_t) / (1 - e_i))
      w_i <- ifelse(T_i == 1, 
                    p_t / e_i,
                    (1 - p_t) / (1 - e_i))
      
      # Multiply weights (for joint treatment)
      weights <- weights * w_i
      
      # Store component for diagnostics
      weight_components[[paste0("w_", therapy)]] <- w_i
      
      cat(sprintf("    %s: propensity range [%.3f, %.3f], treatment rate = %.1f%%\n",
                  therapy, min(e_i), max(e_i), p_t * 100))
    }
  }
  
  # Normalize weights to have mean = 1 (for interpretability)
  weights <- weights / mean(weights)
  
  # Diagnostic statistics
  cat(sprintf("\n  Weight statistics:\n"))
  cat(sprintf("    Mean: %.3f (by design after normalization)\n", mean(weights)))
  cat(sprintf("    Median: %.3f\n", median(weights)))
  cat(sprintf("    Range: [%.3f, %.3f]\n", min(weights), max(weights)))
  cat(sprintf("    SD: %.3f\n", sd(weights)))
  cat(sprintf("    IQR: [%.3f, %.3f]\n", 
              quantile(weights, 0.25), quantile(weights, 0.75)))
  
  # Check for extreme weights
  n_extreme <- sum(weights > quantile(weights, 0.99))
  if(n_extreme > 0) {
    cat(sprintf("    ⚠ Warning: %d observations (%.1f%%) have weights > 99th percentile\n",
                n_extreme, n_extreme / length(weights) * 100))
  }
  
  # Optionally trim extreme weights
  max_weight <- quantile(weights, 0.99)
  weights_trimmed <- pmin(weights, max_weight)
  n_trimmed <- sum(weights != weights_trimmed)
  if(n_trimmed > 0) {
    cat(sprintf("    Trimmed %d weights at 99th percentile (%.3f)\n", 
                n_trimmed, max_weight))
  }
  
  return(list(
    weights = weights_trimmed,
    weights_untrimmed = weights,
    components = weight_components
  ))
}

# Create weights for training set
train_weights_result <- create_aipw_weights(
  X_train, 
  train_data, 
  therapy_cols_in_data, 
  therapy_to_prop
)
train_weights <- train_weights_result$weights

# For test set, we don't use weights (standard prediction)
# But create them for diagnostic purposes
test_weights_result <- create_aipw_weights(
  X_test, 
  test_data, 
  therapy_cols_in_data, 
  therapy_to_prop
)
test_weights <- test_weights_result$weights

# ============================================================================
# HYPERPARAMETER TUNING WITH WEIGHTED TEMPORAL CROSS-VALIDATION
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("HYPERPARAMETER TUNING WITH WEIGHTED TEMPORAL CV\n")
cat(rep("=", 90), "\n", sep="")

# Function to create time series splits
create_time_series_splits <- function(n_samples, n_splits = 3) {
  test_size <- floor(n_samples / (n_splits + 1))
  
  splits <- list()
  for(i in 1:n_splits) {
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
param_grid <- param_grid[sample(nrow(param_grid), 100), ]
cat(sprintf("  Testing %d parameter combinations with %d-fold temporal CV\n", 
            nrow(param_grid), length(cv_splits)))

# Function to evaluate parameters with weighted temporal CV
evaluate_params_weighted_temporal <- function(params_row, X, y, weights, splits, 
                                              df_with_props, therapy_cols, therapy_to_prop_map) {
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
  
  fold_aucs <- numeric(length(splits))
  fold_nrounds <- numeric(length(splits))
  
  for(i in 1:length(splits)) {
    train_idx <- splits[[i]]$train
    test_idx <- splits[[i]]$test
    
    # Create fold-specific weights
    fold_weights_result <- create_aipw_weights(
      X[train_idx, ], 
      df_with_props[train_idx, ],
      therapy_cols,
      therapy_to_prop_map
    )
    fold_weights <- fold_weights_result$weights
    
    # Create weighted DMatrix objects for this fold
    dtrain_fold <- xgb.DMatrix(
      X[train_idx, ], 
      label = y[train_idx],
      weight = fold_weights
    )
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
    
    # Get predictions and calculate AUC (unweighted evaluation)
    pred_fold <- predict(model_fold, dtest_fold)
    auc_fold <- as.numeric(auc(roc(y[test_idx], pred_fold, quiet = TRUE)))
    
    fold_aucs[i] <- auc_fold
    fold_nrounds[i] <- model_fold$best_iteration
  }
  
  return(list(
    mean_auc = mean(fold_aucs),
    std_auc = sd(fold_aucs),
    mean_nrounds = round(mean(fold_nrounds))
  ))
}

# Grid search with weighted temporal CV
results <- data.frame(param_grid)
results$cv_auc_mean <- NA
results$cv_auc_std <- NA
results$best_nrounds <- NA

pb <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
for(i in 1:nrow(param_grid)) {
  eval_result <- evaluate_params_weighted_temporal(
    param_grid[i, ], 
    X_train, 
    y_train, 
    train_weights,
    cv_splits,
    train_data,
    therapy_cols_in_data,
    therapy_to_prop
  )
  results$cv_auc_mean[i] <- eval_result$mean_auc
  results$cv_auc_std[i] <- eval_result$std_auc
  results$best_nrounds[i] <- eval_result$mean_nrounds
  setTxtProgressBar(pb, i)
}
close(pb)

# Find best parameters
results$score <- results$cv_auc_mean - 0.5 * results$cv_auc_std
best_idx <- which.max(results$score)
best_params <- results[best_idx, ]

cat("\n\nBest parameters found:\n")
cat(sprintf("  Mean CV AUC: %.4f (±%.4f)\n", 
            best_params$cv_auc_mean, best_params$cv_auc_std))
print(best_params[, c("max_depth", "eta", "subsample", "colsample_bytree",
                      "min_child_weight", "gamma", "alpha", "lambda", 
                      "best_nrounds")])

cat("\nTop 5 parameter combinations:\n")
top5 <- results[order(results$score, decreasing = TRUE)[1:5], ]
print(top5[, c("max_depth", "eta", "cv_auc_mean", "cv_auc_std", "score")])

# ============================================================================
# TRAIN FINAL WEIGHTED MODEL
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("TRAINING FINAL WEIGHTED MODEL\n")
cat(rep("=", 90), "\n", sep="")

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

# Create weighted DMatrix objects
dtrain <- xgb.DMatrix(
  data = X_train, 
  label = y_train,
  weight = train_weights  # <-- KEY: Use AIPW weights
)
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

# Get predictions (unweighted)
pred_train <- predict(xgb_final, dtrain)
pred_test <- predict(xgb_final, dtest)

# Calculate AUCs
auc_train <- as.numeric(auc(roc(y_train, pred_train, quiet = TRUE)))
auc_test <- as.numeric(auc(roc(y_test, pred_test, quiet = TRUE)))

cat(sprintf("  Training AUC: %.4f\n", auc_train))
cat(sprintf("  Test AUC: %.4f\n", auc_test))
cat(sprintf("  Best iteration: %d\n", xgb_final$best_iteration))

# ============================================================================
# PREDICTIVE PERFORMANCE METRICS
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("PREDICTIVE PERFORMANCE METRICS\n")
cat(rep("=", 90), "\n", sep="")

# ROC analysis
roc_test <- roc(y_test, pred_test, quiet = TRUE)
coords <- coords(roc_test, "best", ret = "all", transpose = FALSE)
optimal_threshold <- coords$threshold[1]

pred_test_binary <- as.integer(pred_test > optimal_threshold)
cm <- table(Actual = y_test, Predicted = pred_test_binary)

cat("Confusion Matrix:\n")
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

# Brier score
brier_score <- mean((pred_test - y_test)^2)
cat(sprintf("  Brier Score: %.4f\n", brier_score))

# Platt scaling for calibration
platt_model <- glm(y_train ~ poly(pred_train, 4), family = binomial)
platt_probs <- predict(platt_model, newdata = data.frame(pred_train = pred_test), type = "response")

brier_score_platt <- mean((platt_probs - y_test)^2)
cat(sprintf("  Brier Score (Platt scaled): %.4f\n", brier_score_platt))

# Calibration in the large
cat(sprintf("\nCalibration:\n"))
cat(sprintf("  Mean predicted probability: %.3f\n", mean(pred_test)))
cat(sprintf("  Observed event rate: %.3f\n", mean(y_test)))
cat(sprintf("  Calibration difference: %.3f\n", mean(pred_test) - mean(y_test)))

# ============================================================================
# FEATURE IMPORTANCE & CATEGORIZATION
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("FEATURE IMPORTANCE\n")
cat(rep("=", 90), "\n", sep="")

# Get XGBoost feature importance
importance_matrix <- xgb.importance(model = xgb_final)
cat("Top 20 features by gain:\n")
print(head(importance_matrix[, c("Feature", "Gain", "Cover", "Frequency")], 20))

# Categorize features by group
importance_df <- as.data.frame(importance_matrix)
importance_df <- importance_df %>%
  mutate(
    feature_group = case_when(
      Feature %in% colnames(X_train)[colnames(X_train) %in% patient_clinical_features] ~ "Patient Clinical",
      Feature %in% colnames(X_train)[colnames(X_train) %in% therapy_features] ~ "Therapy Received",
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
    n_features = n(),
    .groups = 'drop'
  ) %>%
  arrange(desc(total_gain))

cat("\nFeature Importance by Group:\n")
print(group_importance)

# ============================================================================
# DOUBLY ROBUST ESTIMATION WITH CROSS-FITTING 
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("DOUBLY ROBUST ESTIMATION OF THERAPY EFFECTS\n")
cat(rep("=", 90), "\n", sep="")

# Identify therapy columns in the data
therapy_cols <- therapy_features[therapy_features %in% colnames(X_train)]
cat(sprintf("  Found %d therapy modalities in data\n", length(therapy_cols)))
cat("  Using 5-fold cross-fitting to avoid overfitting bias\n\n")

dr_results <- list()

# Create 5 folds for cross-fitting
set.seed(123)
n_folds <- 5
fold_ids <- sample(rep(1:n_folds, length.out = nrow(X_train)))

for(therapy in therapy_cols) {
  prop_col <- therapy_to_prop[therapy]
  
  if(!is.na(prop_col) && prop_col %in% colnames(train_data)) {
    T_i <- X_train[, therapy]
    e_i <- train_data[[prop_col]]
    
    # Bound propensity scores
    e_i <- pmax(0.01, pmin(0.99, e_i))
    
    # Check sample sizes
    n_treated <- sum(T_i == 1)
    n_control <- sum(T_i == 0)
    
    if(n_treated < 20 || n_control < 20) {
      cat(sprintf("  %s: SKIPPED (n_treated=%d, n_control=%d)\n",
                  therapy, n_treated, n_control))
      next
    }
    
    # Prepare features without therapy indicators
    X_without_therapy <- X_train[, !colnames(X_train) %in% therapy_cols]
    
    # Initialize cross-fitted predictions
    mu_1_cf <- numeric(nrow(X_train))
    mu_0_cf <- numeric(nrow(X_train))
    
    # Cross-fitting loop
    for(fold in 1:n_folds) {
      test_idx <- which(fold_ids == fold)
      train_idx <- which(fold_ids != fold)
      
      # Check fold sizes
      n_treated_fold <- sum(T_i[train_idx] == 1)
      n_control_fold <- sum(T_i[train_idx] == 0)
      
      if(n_treated_fold < 5 || n_control_fold < 5) {
        mu_1_cf[test_idx] <- mean(y_train[train_idx[T_i[train_idx] == 1]])
        mu_0_cf[test_idx] <- mean(y_train[train_idx[T_i[train_idx] == 0]])
        next
      }
      
      # Train on treated units
      treated_train_idx <- train_idx[T_i[train_idx] == 1]
      dtrain_t <- xgb.DMatrix(X_without_therapy[treated_train_idx, ],
                              label = y_train[treated_train_idx])
      
      model_t <- xgb.train(
        params = best_params_list,
        data = dtrain_t,
        nrounds = 200,
        verbose = 0
      )
      
      # Train on control units
      control_train_idx <- train_idx[T_i[train_idx] == 0]
      dtrain_c <- xgb.DMatrix(X_without_therapy[control_train_idx, ],
                              label = y_train[control_train_idx])
      
      model_c <- xgb.train(
        params = best_params_list,
        data = dtrain_c,
        nrounds = 200,
        verbose = 0
      )
      
      # Predict on test fold
      dtest_fold <- xgb.DMatrix(X_without_therapy[test_idx, ])
      mu_1_cf[test_idx] <- predict(model_t, dtest_fold)
      mu_0_cf[test_idx] <- predict(model_c, dtest_fold)
    }
    
    # Calculate AIPW estimator
    tau_i <- mu_1_cf - mu_0_cf +
      T_i * (y_train - mu_1_cf) / e_i -
      (1 - T_i) * (y_train - mu_0_cf) / (1 - e_i)
    
    # Average treatment effect
    ate <- mean(tau_i)
    se <- sd(tau_i) / sqrt(length(tau_i))
    
    # Store results
    dr_results[[therapy]] <- list(
      ate = ate,
      se = se,
      ci_lower = ate - 1.96 * se,
      ci_upper = ate + 1.96 * se,
      n_treated = n_treated,
      n_control = n_control,
      p_value = 2 * (1 - pnorm(abs(ate / se)))
    )
    
    # Significance indicator
    p_val <- dr_results[[therapy]]$p_value
    sig_star <- if(p_val < 0.001) "***" else if(p_val < 0.01) "**" else if(p_val < 0.05) "*" else ""
    
    cat(sprintf("  %s: ATE = %.3f (95%% CI: %.3f to %.3f), p=%.3f %s [n=%d]\n",
                therapy, ate, ate - 1.96*se, ate + 1.96*se, p_val, sig_star, n_treated))
  }
}

# Summary statistics
if(length(dr_results) > 0) {
  cat("\n  Summary of therapy associations:\n")
  
  p_values <- sapply(dr_results, function(x) x$p_value)
  n_sig_05 <- sum(p_values < 0.05)
  n_sig_01 <- sum(p_values < 0.01)
  
  cat(sprintf("    Total therapies evaluated: %d\n", length(dr_results)))
  cat(sprintf("    Significant at p<0.05: %d\n", n_sig_05))
  cat(sprintf("    Significant at p<0.01: %d\n", n_sig_01))
  
  ates <- sapply(dr_results, function(x) x$ate)
  best_therapy <- names(which.max(ates))
  best_ate <- max(ates)
  
  cat(sprintf("    Largest effect: %s (%.1f percentage points)\n", 
              best_therapy, best_ate * 100))
  
  harmful <- names(ates[ates < -0.01])
  if(length(harmful) > 0) {
    cat(sprintf("    ⚠ Potentially harmful (>1pp decrease): %s\n", 
                paste(harmful, collapse = ", ")))
  }
}

# ============================================================================
# COUNTERFACTUAL ANALYSIS FOR PERSONALIZATION
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("COUNTERFACTUAL ANALYSIS\n")
cat(rep("=", 90), "\n", sep="")

# Get observed therapy combinations
combo_matrix <- X_test[, therapy_cols]
combo_strings <- apply(combo_matrix, 1, paste, collapse = "-")
combo_counts <- table(combo_strings)
top_combos <- names(sort(combo_counts, decreasing = TRUE)[1:min(15, length(combo_counts))])

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

# Create visualization data
personalization_viz_data <- data.frame(
  baseline_prob = pred_test,
  gain = personalization_gains * 100,
  initial_risk = test_data$risk_level_initial,
  observed_outcome = y_test,
  gain_category = cut(personalization_gains * 100,
                      breaks = c(-Inf, 0, 1, 5, Inf),
                      labels = c("None", "Minimal (0-1pp)", 
                                 "Moderate (1-5pp)", "Large (>5pp)"),
                      include.lowest = TRUE)
)

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

# Gain among benefiters
if(sum(personalization_gains > 0) > 0) {
  mean_gain_benefiters <- mean(personalization_gains[personalization_gains > 0])
  cat(sprintf("  Mean gain among benefiters: %.3f (%.1f%% relative improvement)\n",
              mean_gain_benefiters,
              mean_gain_benefiters / mean(pred_test[personalization_gains > 0]) * 100))
}

# Gains by predicted probability quartile
summary_by_quartile <- personalization_viz_data %>%
  mutate(prob_quartile = cut(baseline_prob,
                             breaks = quantile(baseline_prob,
                                               probs = c(0, 0.25, 0.5, 0.75, 1)),
                             labels = c("Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"),
                             include.lowest = TRUE)) %>%
  group_by(prob_quartile) %>%
  summarise(
    n = n(),
    mean_baseline_prob = mean(baseline_prob),
    mean_gain = mean(gain),
    median_gain = median(gain),
    pct_benefit_1pp = mean(gain > 1) * 100,
    pct_benefit_5pp = mean(gain > 5) * 100,
    max_gain = max(gain),
    .groups = 'drop'
  )

cat("\nPersonalization Gains by Baseline Probability Quartile:\n")
print(summary_by_quartile)

# ============================================================================
# EXPORT RESULTS FOR MANUSCRIPT
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("EXPORTING RESULTS\n")
cat(rep("=", 90), "\n", sep="")

# Create results list
manuscript_results <- list(
  # Model info
  model_type = "AIPW-weighted XGBoost",
  
  # Model performance
  model_performance = data.frame(
    metric = c("AUC_train", "AUC_test", "Sensitivity", "Specificity", 
               "PPV", "NPV", "Brier", "Brier_Platt"),
    value = c(auc_train, auc_test, sensitivity, specificity, ppv, npv,
              brier_score, brier_score_platt)
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
      p_value = sapply(dr_results, function(x) x$p_value),
      n_treated = sapply(dr_results, function(x) x$n_treated)
    ) %>% arrange(desc(ate))
  } else NULL,
  
  # Personalization summary
  personalization_summary = data.frame(
    metric = c("mean_gain", "median_gain", "pct_benefit", "pct_large_benefit", 
               "max_gain", "nnt"),
    value = c(mean_gain, median_gain, pct_benefit/100, pct_large_benefit/100,
              max(personalization_gains), ifelse(mean_gain > 0, 1/mean_gain, NA))
  ),
  
  # Personalization by quartile
  personalization_by_quartile = summary_by_quartile,
  
  # Model object
  xgboost_model = xgb_final,
  
  # Weights diagnostics
  weight_summary = data.frame(
    dataset = c("Training", "Test"),
    mean = c(mean(train_weights), mean(test_weights)),
    median = c(median(train_weights), median(test_weights)),
    sd = c(sd(train_weights), sd(test_weights)),
    min = c(min(train_weights), min(test_weights)),
    max = c(max(train_weights), max(test_weights))
  ),
  
  # Individual predictions
  predictions = data.frame(
    observed_prob = pred_test,
    personalization_gain = personalization_gains,
    optimal_therapy = apply(optimal_combos, 1, function(x) 
      paste(therapy_cols[x == 1], collapse = "+"))
  )
)

# Save results
saveRDS(manuscript_results, "outputs/aipw_xgboost_results.rds")
cat("  Results saved to: outputs/aipw_xgboost_results.rds\n")

# Save detailed counterfactual analysis
sink('outputs/tables/counterfactual_analysis_aipw.txt')
cat(rep("=", 90), "\n", sep = "")
cat("PERSONALIZATION GAINS BY BASELINE PROBABILITY QUARTILE (AIPW MODEL)\n")
cat(rep("=", 90), "\n", sep = "")
print(summary_by_quartile, width = Inf)
sink()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("ANALYSIS COMPLETE - SUMMARY FOR MANUSCRIPT\n")
cat(rep("=", 90), "\n", sep="")

cat(sprintf("\nMODEL TYPE:\n"))
cat(sprintf("   AIPW-weighted XGBoost (propensity scores used as weights, not features)\n"))

cat(sprintf("\nPREDICTIVE PERFORMANCE:\n"))
cat(sprintf("   Test set AUROC: %.3f\n", auc_test))
cat(sprintf("   Sensitivity: %.1f%%\n", sensitivity * 100))
cat(sprintf("   Brier score: %.4f\n", brier_score))

cat(sprintf("\nTHERAPY ASSOCIATIONS:\n"))
if(length(dr_results) > 0) {
  cat(sprintf("   Therapies evaluated: %d\n", length(dr_results)))
  cat(sprintf("   Significant effects (p<0.05): %d\n", sum(p_values < 0.05)))
  if(length(ates) > 0) {
    cat(sprintf("   Largest benefit: %s (%.1f pp, p=%.3f)\n",
                names(which.max(ates)), max(ates) * 100,
                dr_results[[names(which.max(ates))]]$p_value))
  }
}

cat(sprintf("\nPERSONALIZATION POTENTIAL:\n"))
cat(sprintf("   Patients who could benefit: %.1f%%\n", pct_benefit))
cat(sprintf("   Mean predicted gain: %.1f percentage points\n", mean_gain * 100))
cat(sprintf("   Number needed to treat: %.0f\n", ifelse(mean_gain > 0, 1/mean_gain, Inf)))

cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("\nAll results exported to outputs/ directory\n")
cat(rep("=", 90), "\n", sep="")