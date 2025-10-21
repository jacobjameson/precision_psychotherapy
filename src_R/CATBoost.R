# ============================================================================
# ENHANCED XGBOOST PIPELINE: THERAPY OPTIMIZATION FOR SUICIDE RISK REDUCTION
# ============================================================================
# Improvements include:
# - Treatment-patient interaction features
# - Boruta feature selection
# - Class weighting for imbalance
# - Enhanced hyperparameter optimization
# - Additional feature engineering
# ============================================================================

# Load required libraries
library(dplyr)
library(xgboost)
library(pROC)
library(ggplot2)
library(tidyr)
library(Boruta)

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
# PART 2: FEATURE ENGINEERING (ENHANCED)
# ============================================================================
cat("\n[STEP 2] Defining and creating features...\n")

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

# Keep only therapy features that exist in data
therapy_features <- therapy_features[therapy_features %in% names(data)]

# Therapy propensity scores
propensity_features <- names(data)[grepl("^prop_", names(data))]

# Treatment context features
treatment_context_features <- c("therapy_duration_category", "delivery_method",
                                "session_mode", "intake_to_pn", "days_first_srs")

# Therapist/organizational features
therapist_features <- c("therapist_name", "location", "program",
                        "pn_month", "pn_time_block", "pn_year")

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

# ============================================================================
# PART 4: CREATE INTERACTION FEATURES
# ============================================================================
cat("\n[STEP 4] Creating treatment-patient interaction features...\n")

# Create interaction features between therapies and key patient characteristics
interaction_features <- c()

for(therapy in therapy_features) {
  if(therapy %in% names(train_data)) {
    # Interaction with risk level
    feat_name_risk <- paste0(therapy, "_x_high_risk")
    train_data[[feat_name_risk]] <- train_data[[therapy]] * train_data$risk_high_initial
    test_data[[feat_name_risk]] <- test_data[[therapy]] * test_data$risk_high_initial
    interaction_features <- c(interaction_features, feat_name_risk)
    
    # Interaction with severity (if exists)
    if("total_score" %in% names(train_data)) {
      feat_name_score <- paste0(therapy, "_x_total_score")
      train_data[[feat_name_score]] <- train_data[[therapy]] * train_data$total_score
      test_data[[feat_name_score]] <- test_data[[therapy]] * test_data$total_score
      interaction_features <- c(interaction_features, feat_name_score)
    }
    
    # Interaction with age group
    if("adolescent" %in% names(train_data)) {
      feat_name_age <- paste0(therapy, "_x_adolescent")
      train_data[[feat_name_age]] <- train_data[[therapy]] * train_data$adolescent
      test_data[[feat_name_age]] <- test_data[[therapy]] * test_data$adolescent
      interaction_features <- c(interaction_features, feat_name_age)
    }
  }
}

# Create therapy combination features
train_data$n_therapies <- rowSums(train_data[, therapy_features, drop = FALSE])
test_data$n_therapies <- rowSums(test_data[, therapy_features, drop = FALSE])

# Key therapy combinations
if(all(c("cbt", "dbt") %in% therapy_features)) {
  train_data$has_cbt_dbt <- as.integer((train_data$cbt == 1) & (train_data$dbt == 1))
  test_data$has_cbt_dbt <- as.integer((test_data$cbt == 1) & (test_data$dbt == 1))
}

if(all(c("motivational_interviewing", "cbt") %in% therapy_features)) {
  train_data$has_mi_cbt <- as.integer((train_data$motivational_interviewing == 1) & (train_data$cbt == 1))
  test_data$has_mi_cbt <- as.integer((test_data$motivational_interviewing == 1) & (test_data$cbt == 1))
}

# Therapy intensity indicator
train_data$high_therapy_intensity <- as.integer(train_data$n_therapies >= 3)
test_data$high_therapy_intensity <- as.integer(test_data$n_therapies >= 3)

combination_features <- c("n_therapies", "high_therapy_intensity")
if("has_cbt_dbt" %in% names(train_data)) combination_features <- c(combination_features, "has_cbt_dbt")
if("has_mi_cbt" %in% names(train_data)) combination_features <- c(combination_features, "has_mi_cbt")

cat(sprintf("  Created %d interaction features\n", length(interaction_features)))
cat(sprintf("  Created %d combination features\n", length(combination_features)))

# Update feature lists
all_features <- unique(c(
  patient_clinical_features, 
  therapy_features,
  propensity_features, 
  treatment_context_features,
  therapist_features,
  interaction_features,
  combination_features
))

# Keep only features that exist in data
all_features <- all_features[all_features %in% names(train_data)]

cat(sprintf("  Total features available: %d\n", length(all_features)))

# ============================================================================
# PART 5: PREPARE DATA MATRICES
# ============================================================================
cat("\n[STEP 5] Preparing feature matrices...\n")

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

cat(sprintf("\nInitial matrix dimensions:\n"))
cat(sprintf("  X_train: %d rows x %d columns\n", nrow(X_train), ncol(X_train)))
cat(sprintf("  X_test: %d rows x %d columns\n", nrow(X_test), ncol(X_test)))

# ============================================================================
# PART 6: FEATURE SELECTION WITH BORUTA
# ============================================================================
cat("\n[STEP 6] Running Boruta feature selection...\n")
cat("  This may take a few minutes...\n")

# Run Boruta algorithm
set.seed(123)
boruta_result <- Boruta(
  x = X_train,
  y = as.factor(y_train),
  maxRuns = 30,  # Reduced for speed, increase to 100 for production
  doTrace = 1,
  getImp = getImpXgboost  # Use XGBoost for importance calculation
)

# Get confirmed and tentative features
confirmed_features <- getSelectedAttributes(boruta_result, withTentative = FALSE)
tentative_features <- c()
if(length(boruta_result$finalDecision[boruta_result$finalDecision == "Tentative"]) > 0) {
  tentative_features <- names(boruta_result$finalDecision[boruta_result$finalDecision == "Tentative"])
}

# Include confirmed and tentative features
selected_features <- c(confirmed_features, tentative_features)

# Ensure we keep therapy features and propensity scores (for counterfactual analysis)
essential_features <- c(
  colnames(X_train)[colnames(X_train) %in% therapy_features],
  colnames(X_train)[grepl("^prop_", colnames(X_train))],
  colnames(X_train)[colnames(X_train) %in% interaction_features]
)
selected_features <- unique(c(selected_features, essential_features))

cat(sprintf("\n  Boruta results:\n"))
cat(sprintf("    Confirmed important: %d features\n", length(confirmed_features)))
cat(sprintf("    Tentative: %d features\n", length(tentative_features)))
cat(sprintf("    Including essential therapy features: %d\n", length(essential_features)))
cat(sprintf("    Total selected: %d features (from %d)\n", 
            length(selected_features), ncol(X_train)))

# Filter matrices to selected features
X_train <- X_train[, selected_features, drop = FALSE]
X_test <- X_test[, selected_features, drop = FALSE]

# ============================================================================
# PART 7: HYPERPARAMETER TUNING
# ============================================================================
cat("\n[STEP 7] Hyperparameter tuning for XGBoost...\n")

# Calculate class weight for imbalance
pos_weight <- sum(y_train == 0) / sum(y_train == 1)
cat(sprintf("  Class weight (pos_weight): %.3f\n", pos_weight))

# Define expanded parameter grid
param_grid <- expand.grid(
  max_depth = c(3, 4, 5, 6, 7, 8),
  eta = c(0.01, 0.03, 0.05, 0.08, 0.1, 0.15),
  subsample = c(0.6, 0.7, 0.8, 0.9),
  colsample_bytree = c(0.6, 0.7, 0.8, 0.9),
  min_child_weight = c(1, 3, 5, 7, 10),
  gamma = c(0, 0.5, 1, 2),
  alpha = c(0, 0.5, 1, 2),
  lambda = c(0.5, 1, 2, 3)
)

# Sample combinations for efficiency
set.seed(100)
n_combinations <- min(150, nrow(param_grid))  # Increased from 100
param_grid <- param_grid[sample(nrow(param_grid), n_combinations), ]
cat(sprintf("  Testing %d parameter combinations with 5-fold CV\n", nrow(param_grid)))

# Function to evaluate parameters
evaluate_params <- function(params_row, X, y, nfolds = 5, pos_weight = 1) {
  set.seed(123)
  
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
    lambda = params_row$lambda,
    scale_pos_weight = pos_weight
  )
  
  cv_result <- xgb.cv(
    params = params,
    data = xgb.DMatrix(X, label = y),
    nrounds = 500,
    nfold = nfolds,
    early_stopping_rounds = 30,
    verbose = 0,
    prediction = FALSE,
    stratified = TRUE  # Ensure stratified folds
  )
  
  best_auc <- max(cv_result$evaluation_log$test_auc_mean, na.rm = TRUE)
  best_iter <- which.max(cv_result$evaluation_log$test_auc_mean)
  
  return(list(auc = best_auc, nrounds = best_iter))
}

# Grid search with progress bar
results <- data.frame(param_grid)
results$cv_auc <- NA
results$best_nrounds <- NA

pb <- txtProgressBar(min = 0, max = nrow(param_grid), style = 3)
for(i in 1:nrow(param_grid)) {
  eval_result <- evaluate_params(param_grid[i, ], X_train, y_train, pos_weight = pos_weight)
  results$cv_auc[i] <- eval_result$auc
  results$best_nrounds[i] <- eval_result$nrounds
  setTxtProgressBar(pb, i)
}
close(pb)

# Find best parameters
best_idx <- which.max(results$cv_auc)
best_params <- results[best_idx, ]

cat("\n\nBest parameters found:\n")
print(best_params[, c("max_depth", "eta", "subsample", "colsample_bytree",
                      "min_child_weight", "gamma", "alpha", "lambda", 
                      "cv_auc", "best_nrounds")])

# ============================================================================
# PART 8: TRAIN FINAL MODEL
# ============================================================================
cat("\n[STEP 8] Training final XGBoost with best parameters...\n")

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
  lambda = best_params$lambda,
  scale_pos_weight = pos_weight
)

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest <- xgb.DMatrix(data = X_test, label = y_test)
watchlist <- list(train = dtrain, test = dtest)

# Train model with more rounds
xgb_final <- xgb.train(
  params = best_params_list,
  data = dtrain,
  nrounds = 2000,  # Increased from 1000
  watchlist = watchlist,
  early_stopping_rounds = 50,  # Increased from 30
  verbose = 1,
  print_every_n = 100
)

# Get predictions
pred_train <- predict(xgb_final, dtrain)
pred_test <- predict(xgb_final, dtest)

# Calculate AUCs
auc_train <- as.numeric(auc(roc(y_train, pred_train, quiet = TRUE)))
auc_test <- as.numeric(auc(roc(y_test, pred_test, quiet = TRUE)))

cat(sprintf("\n  Training AUC: %.4f\n", auc_train))
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

if(nrow(cm) == 2 && ncol(cm) == 2) {
  sensitivity <- cm[2,2] / sum(cm[2,])
  specificity <- cm[1,1] / sum(cm[1,])
  ppv <- cm[2,2] / sum(cm[,2])
  npv <- cm[1,1] / sum(cm[,1])
  
  cat(sprintf("\nTest Set Performance (threshold = %.3f):\n", optimal_threshold))
  cat(sprintf("  Sensitivity: %.1f%%\n", sensitivity * 100))
  cat(sprintf("  Specificity: %.1f%%\n", specificity * 100))
  cat(sprintf("  PPV: %.1f%%\n", ppv * 100))
  cat(sprintf("  NPV: %.1f%%\n", npv * 100))
}

# Calculate additional metrics
brier_score <- mean((pred_test - y_test)^2)
cat(sprintf("  Brier Score: %.4f\n", brier_score))

# ============================================================================
# PART 9: FEATURE IMPORTANCE & SHAP ANALYSIS
# ============================================================================
cat("\n[STEP 9] Analyzing feature importance...\n")

# Get XGBoost feature importance
importance_matrix <- xgb.importance(model = xgb_final)
cat("\nTop 20 features by gain:\n")
print(head(importance_matrix[, c("Feature", "Gain", "Cover", "Frequency")], 20))

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
      Feature %in% interaction_features ~ "Treatment Interactions",
      Feature %in% combination_features ~ "Therapy Combinations",
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
    top_feature = Feature[which.max(Gain)]
  ) %>%
  arrange(desc(total_gain))

cat("\nFeature Importance by Group:\n")
print(group_importance)

# ============================================================================
# PART 10: DOUBLY ROBUST ESTIMATION
# ============================================================================
cat("\n[STEP 10] Performing doubly robust estimation for each therapy...\n")

# Identify therapy columns in the data
therapy_cols <- therapy_features[therapy_features %in% colnames(X_train)]
cat(sprintf("  Found %d therapy modalities in data\n", length(therapy_cols)))

dr_results <- list()

# Map therapy names to propensity score columns
therapy_to_prop <- c(
  "act" = "prop_act",
  "cbt" = "prop_cbt",
  "dbt" = "prop_dbt",
  "motivational_interviewing" = "prop_mi",
  "mindfulness" = "prop_mindfulness",
  "stages_of_change" = "prop_stages_of_change",
  "family_systems" = "prop_family_systems"
)

for(therapy in therapy_cols) {
  prop_col <- therapy_to_prop[therapy]
  
  if(!is.na(prop_col) && prop_col %in% colnames(X_train)) {
    T_i <- X_train[, therapy]
    e_i <- X_train[, prop_col]
    
    # Bound propensity scores away from 0 and 1
    e_i <- pmax(0.01, pmin(0.99, e_i))
    
    # Check sample sizes
    n_treated <- sum(T_i == 1)
    n_control <- sum(T_i == 0)
    
    if(n_treated >= 20 && n_control >= 20) {  # Minimum sample size check
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
    } else {
      cat(sprintf("  %s: Insufficient sample size (n_treated=%d, n_control=%d)\n",
                  therapy, n_treated, n_control))
    }
  }
}

# ============================================================================
# PART 11: COUNTERFACTUAL ANALYSIS
# ============================================================================
cat("\n[STEP 11] Computing personalization gains...\n")

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
    
    # Update interaction features for counterfactual
    for(therapy in therapy_cols) {
      if(paste0(therapy, "_x_high_risk") %in% names(cf_features)) {
        cf_features[paste0(therapy, "_x_high_risk")] <- 
          combo_vals[which(therapy_cols == therapy)] * cf_features["risk_high_initial"]
      }
      if(paste0(therapy, "_x_total_score") %in% names(cf_features)) {
        cf_features[paste0(therapy, "_x_total_score")] <- 
          combo_vals[which(therapy_cols == therapy)] * cf_features["total_score"]
      }
      if(paste0(therapy, "_x_adolescent") %in% names(cf_features)) {
        cf_features[paste0(therapy, "_x_adolescent")] <- 
          combo_vals[which(therapy_cols == therapy)] * cf_features["adolescent"]
      }
    }
    
    # Update combination features
    cf_features["n_therapies"] <- sum(combo_vals)
    if("high_therapy_intensity" %in% names(cf_features)) {
      cf_features["high_therapy_intensity"] <- as.integer(sum(combo_vals) >= 3)
    }
    
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

# ============================================================================
# PART 12: VISUALIZATIONS
# ============================================================================
cat("\n[STEP 12] Creating visualizations...\n")

# [Visualization code remains the same as your original]
# ... [Include all your visualization code here] ...

# ============================================================================
# PART 13: EXPORT RESULTS
# ============================================================================
cat("\n[STEP 13] Exporting results for manuscript...\n")

# [Export code remains the same as your original]
# ... [Include all your export code here] ...

# ============================================================================
# FINAL SUMMARY
# ============================================================================
cat("\n")
cat(rep("=", 90), "\n", sep="")
cat("ANALYSIS COMPLETE - ENHANCED MODEL SUMMARY\n")
cat(rep("=", 90), "\n", sep="")

cat(sprintf("\n1. FEATURE SELECTION:\n"))
cat(sprintf("   Original features: %d\n", length(all_features)))
cat(sprintf("   After Boruta selection: %d\n", length(selected_features)))

cat(sprintf("\n2. PREDICTIVE PERFORMANCE:\n"))
cat(sprintf("   Test set AUROC: %.3f\n", auc_test))
cat(sprintf("   Brier Score: %.4f\n", brier_score))
if(exists("sensitivity")) {
  cat(sprintf("   Sensitivity: %.1f%%\n", sensitivity * 100))
}

cat(sprintf("\n3. THERAPY ASSOCIATIONS (Doubly Robust):\n"))
if(length(dr_results) > 0) {
  sorted_dr <- sort(sapply(dr_results, function(x) x$ate), decreasing = TRUE)
  for(i in 1:min(3, length(sorted_dr))) {
    cat(sprintf("   %s: %.1f percentage points (95%% CI: %.1f to %.1f)\n",
                names(sorted_dr)[i], sorted_dr[i] * 100,
                dr_results[[names(sorted_dr)[i]]]$ci_lower * 100,
                dr_results[[names(sorted_dr)[i]]]$ci_upper * 100))
  }
}

cat(sprintf("\n4. PERSONALIZATION POTENTIAL:\n"))
cat(sprintf("   Patients who could benefit: %.1f%%\n", pct_benefit))
cat(sprintf("   Mean predicted gain: %.1f percentage points\n", mean_gain * 100))

cat("\n")
cat(rep("=", 90), "\n", sep="")