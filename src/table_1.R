
create_characteristics_table <- function(data) {
  
  data_prepared <- data %>%
    mutate(
      risk_level_initial = factor(risk_level_initial, 
                                  levels = c("Moderate", "High"),
                                  labels = c("Moderate", "High")),
      age_group = factor(age_group),
      sex_fs = factor(sex_fs),
      program = factor(program),
  
        therapy_count = act + cbt + dbt + motivational_interviewing + 
        mindfulness + stages_of_change + family_systems + trauma_informed,
      two_plus_modalities = therapy_count >= 2,
      three_plus_modalities = therapy_count >= 3,
      
      total_score = as.numeric(total_score)
    )
  
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

table1 <- create_characteristics_table(data)

# save as png
gtsave(as_gt(table1), "outputs/tables/table1.png")
