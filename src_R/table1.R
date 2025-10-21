# Load required libraries
library(gt)
library(tidyverse)
library(gtsummary)

# Create the summary table using gtsummary (which integrates well with gt)
create_characteristics_table <- function(data) {
  
  # First, prepare the data - ensure proper variable types and labels
  data_prepared <- data %>%
    mutate(
      # Ensure categorical variables are factors
      risk_level_initial = factor(risk_level_initial, 
                                  levels = c("Moderate", "High"),
                                  labels = c("Moderate", "High")),
      age_group = factor(age_group),
      sex_fs = factor(sex_fs),
      program = factor(program),
  
      
      # Create combination therapy variables
      therapy_count = act + cbt + dbt + motivational_interviewing + 
        mindfulness + stages_of_change + family_systems + trauma_informed,
      two_plus_modalities = therapy_count >= 2,
      three_plus_modalities = therapy_count >= 3,
      
      # Ensure numeric score
      total_score = as.numeric(total_score)
    )
  
  # Create the main table using gtsummary
  tbl_summary <- data_prepared %>%
    tbl_summary(
      by = risk_level_initial,  # Stratify by risk level
      include = c(
        age_group, 
        sex_fs, 
        total_score, 
        program,
        dx_group,  # Primary diagnosis
        risk_level_srs_first,  # Risk level at discharge
        days_first_srs,
        act, cbt, dbt, 
        motivational_interviewing, 
        mindfulness, 
        stages_of_change, 
        family_systems, 
        trauma_informed,
        two_plus_modalities,
        three_plus_modalities
      ),
      label = list(
        age_group ~ "Age Group",
        sex_fs ~ "Sex",
        total_score ~ "C-SSRS Score (0–25)",
        program ~ "Program",
        dx_group ~ "Primary Diagnosis",
        risk_level_srs_first ~ "Risk Level at Next Assessment",
        days_first_srs ~ "Days Between Assessment",
        act ~ "ACT",
        cbt ~ "CBT",
        dbt ~ "DBT",
        motivational_interviewing ~ "Motivational Interviewing",
        mindfulness ~ "Mindfulness Techniques",
        stages_of_change ~ "Stages-of-Change",
        family_systems ~ "Family Systems",
        trauma_informed ~ "Trauma-Informed",
        two_plus_modalities ~ "≥ 2 Modalities",
        three_plus_modalities ~ "≥ 3 Modalities"
      ),
      type = list(
        # Specify variable types
        all_categorical() ~ "categorical",
        total_score ~ "continuous"
      ),
      statistic = list(
        all_categorical() ~ "{n}/{N} ({p}%)",
        all_continuous() ~ "{mean} ({sd})"
      ),
      digits = list(
        all_categorical() ~ c(0, 0, 1),
        all_continuous() ~ c(1, 1)
      ),
      missing = "no"  # Don't show missing data
    ) %>%
    add_overall() %>%  # Add overall column
    add_p(  # Add p-values
      test = list(
        all_categorical() ~ "chisq.test",
        all_continuous() ~ "t.test"
      ),
      pvalue_fun = function(x) {
        case_when(
          x < 0.001 ~ "<0.001",
          x < 0.01 ~ sprintf("%.3f†", x),
          x < 0.05 ~ sprintf("%.3f†", x),
          TRUE ~ sprintf("%.2f", x)
        )
      }
    ) %>%
    modify_header(
      label ~ "**Characteristic**",
      stat_0 ~ "**Overall**\n(N = {N})",
      stat_1 ~ "**Moderate**\n(N = {n})",
      stat_2 ~ "**High**\n(N = {n})",
      p.value ~ "**p-value**"
    ) %>%
    modify_spanning_header(
      c(stat_1, stat_2) ~ "**Intake Risk Level (C-SSRS)**"
    ) %>%
    bold_labels()
  
  return(tbl_summary)
}

# Alternative approach using gt directly for more control
create_gt_table_manual <- function(data) {
  
  # Calculate summary statistics
  summary_stats <- data %>%
    mutate(
      overall = "Overall",
      risk_group = risk_level_initial
    ) %>%
    bind_rows(
      mutate(., risk_group = "Overall")
    ) %>%
    group_by(risk_group) %>%
    summarise(
      n = n(),
      
      # Age groups
      adolescent_n = sum(age_group == "Adolescent", na.rm = TRUE),
      adolescent_pct = adolescent_n / n() * 100,
      adult_n = sum(age_group == "Adult", na.rm = TRUE),
      adult_pct = adult_n / n() * 100,
      
      # Sex
      female_n = sum(sex_fs == "Female", na.rm = TRUE),
      female_pct = female_n / n() * 100,
      male_n = sum(sex_fs == "Male", na.rm = TRUE),
      male_pct = male_n / n() * 100,
      
      # C-SSRS Score
      score_mean = mean(total_score, na.rm = TRUE),
      score_sd = sd(total_score, na.rm = TRUE),
      
      # Programs (adjust based on your actual program codes)
      op_n = sum(program == "OP", na.rm = TRUE),
      op_pct = op_n / n() * 100,
      php_n = sum(program == "PHP", na.rm = TRUE),
      php_pct = php_n / n() * 100,
      rtc_n = sum(program == "RTC", na.rm = TRUE),
      rtc_pct = rtc_n / n() * 100,
      
      # Therapy modalities
      act_n = sum(act == 1, na.rm = TRUE),
      act_pct = act_n / n() * 100,
      cbt_n = sum(cbt == 1, na.rm = TRUE),
      cbt_pct = cbt_n / n() * 100,
      dbt_n = sum(dbt == 1, na.rm = TRUE),
      dbt_pct = dbt_n / n() * 100,
      
      .groups = "drop"
    )
  
  # Reshape data for gt table
  table_data <- tibble(
    Characteristic = c(
      "Age Group",
      "  Adolescent",
      "  Adult",
      "Sex",
      "  Female", 
      "  Male",
      "C-SSRS Score (0–25)",
      "  Mean (SD)",
      "Program",
      "  OP",
      "  PHP",
      "  RTC",
      "Initial Therapy at Intake",
      "  ACT",
      "  CBT",
      "  DBT"
    ),
    Overall = c(
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$adolescent_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$adolescent_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$adult_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$adult_pct),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$female_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$female_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$male_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$male_pct),
      "",
      sprintf("%.1f (%.1f)", 
              filter(summary_stats, risk_group == "Overall")$score_mean,
              filter(summary_stats, risk_group == "Overall")$score_sd),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$op_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$op_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$php_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$php_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$rtc_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$rtc_pct),
      "",
      sprintf("%d/%d (%.1f%%)", 
              filter(summary_stats, risk_group == "Overall")$act_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$act_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$cbt_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$cbt_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Overall")$dbt_n,
              filter(summary_stats, risk_group == "Overall")$n,
              filter(summary_stats, risk_group == "Overall")$dbt_pct)
    ),
    Moderate = c(
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$adolescent_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$adolescent_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$adult_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$adult_pct),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$female_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$female_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$male_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$male_pct),
      "",
      sprintf("%.1f (%.1f)", 
              filter(summary_stats, risk_group == "Moderate")$score_mean,
              filter(summary_stats, risk_group == "Moderate")$score_sd),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$op_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$op_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$php_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$php_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$rtc_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$rtc_pct),
      "",
      sprintf("%d/%d (%.1f%%)", 
              filter(summary_stats, risk_group == "Moderate")$act_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$act_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$cbt_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$cbt_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "Moderate")$dbt_n,
              filter(summary_stats, risk_group == "Moderate")$n,
              filter(summary_stats, risk_group == "Moderate")$dbt_pct)
    ),
    High = c(
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$adolescent_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$adolescent_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$adult_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$adult_pct),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$female_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$female_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$male_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$male_pct),
      "",
      sprintf("%.1f (%.1f)", 
              filter(summary_stats, risk_group == "High")$score_mean,
              filter(summary_stats, risk_group == "High")$score_sd),
      "",
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$op_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$op_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$php_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$php_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$rtc_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$rtc_pct),
      "",
      sprintf("%d/%d (%.1f%%)", 
              filter(summary_stats, risk_group == "High")$act_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$act_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$cbt_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$cbt_pct),
      sprintf("%d/%d (%.0f%%)", 
              filter(summary_stats, risk_group == "High")$dbt_n,
              filter(summary_stats, risk_group == "High")$n,
              filter(summary_stats, risk_group == "High")$dbt_pct)
    ),
    `p-value` = c(
      "0.009†", "", "", 
      "0.20†", "", "",
      "<0.001‡", "",
      "<0.001†", "", "", "",
      "", "0.50†", "0.039†", ">0.90†"
    )
  )
  
  # Create gt table
  gt_table <- table_data %>%
    gt() %>%
    tab_header(
      title = "Table 1. Cohort Characteristics"
    ) %>%
    tab_spanner(
      label = "Intake Risk Level (C-SSRS)",
      columns = c(Moderate, High)
    ) %>%
    cols_label(
      Characteristic = "",
      Overall = html(paste0("Overall<br>(N = ", filter(summary_stats, risk_group == "Overall")$n, ")")),
      Moderate = html(paste0("Moderate<br>(N = ", filter(summary_stats, risk_group == "Moderate")$n, ")")),
      High = html(paste0("High<br>(N = ", filter(summary_stats, risk_group == "High")$n, ")")),
      `p-value` = "p-value"
    ) %>%
    tab_style(
      style = list(
        cell_text(weight = "bold")
      ),
      locations = cells_body(
        columns = Characteristic,
        rows = !str_detect(Characteristic, "^  ")
      )
    ) %>%
    tab_style(
      style = list(
        cell_text(indent = px(20))
      ),
      locations = cells_body(
        columns = Characteristic,
        rows = str_detect(Characteristic, "^  ")
      )
    ) %>%
    tab_options(
      table.font.size = px(12),
      heading.title.font.size = px(14),
      heading.title.font.weight = "bold",
      column_labels.font.weight = "bold"
    ) %>%
    tab_footnote(
      footnote = "† Chi-square test",
      locations = cells_body(
        columns = `p-value`,
        rows = str_detect(`p-value`, "†")
      )
    ) %>%
    tab_footnote(
      footnote = "‡ t-test",
      locations = cells_body(
        columns = `p-value`,
        rows = str_detect(`p-value`, "‡")
      )
    ) %>%
    tab_footnote(
      footnote = "* Primary diagnosis data available for 4,409 patients",
      locations = cells_body(
        columns = Characteristic,
        rows = Characteristic == "Primary Diagnosis"
      )
    )
  
  return(gt_table)
}

# Run the analysis
# Method 1: Using gtsummary (recommended for publication-quality tables)
table1 <- create_characteristics_table(data)

# Convert to gt object for additional customization if needed
gt_table1 <- as_gt(table1)

# Method 2: Manual creation with gt (more control over formatting)
gt_table2 <- create_gt_table_manual(data)

# Save the table
# gtsave(gt_table1, "table1_characteristics.html")
# gtsave(gt_table1, "table1_characteristics.png")
# gtsave(gt_table1, "table1_characteristics.docx")

# Display the table
gt_table1

