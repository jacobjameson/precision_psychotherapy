library(ComplexHeatmap)

# Your existing data prep
therapy_labels <- c(
  "cbt" = "CBT",
  "dbt" = "DBT",
  "act" = "ACT",
  "motivational_interviewing" = "Motivational Interviewing",
  "mindfulness" = "Mindfulness",
  "stages_of_change" = "Stages of Change",
  "family_systems" = "Family Systems"
)

upset_data <- data %>%
  select(all_of(names(therapy_labels))) %>%
  mutate(across(everything(), ~ . == 1))

colnames(upset_data) <- therapy_labels
upset_data <- upset_data[rowSums(upset_data) > 0, ]

# Convert to matrix
upset_matrix <- as.matrix(upset_data)

# Make combination matrix
m <- make_comb_mat(upset_matrix)
m <- m[comb_size(m) >= 10]

# Calculate total for reference
total_patients <- nrow(upset_data)


# Save as high-res PNG
png("outputs/figures/figure1.png", 
    width = 10, height = 4, units = "in", res = 300)

UpSet(
  m, 
  comb_order = order(-comb_size(m)),
  top_annotation = HeatmapAnnotation(
    "Patients per\n Combination" = anno_barplot(
      comb_size(m), 
      border = FALSE, 
      gp = gpar(fill = "#C1121F"),
      height = unit(4, "cm"),
      add_numbers = F,
      numbers_gp = gpar(fontsize = 8)
    ),
    annotation_name_side = "left",
    annotation_name_rot = 0,
    annotation_name_gp = gpar(fontsize = 11)
  ),
  right_annotation = rowAnnotation(
    "Set Size" = anno_barplot(
      set_size(m),
      border = FALSE,
      gp = gpar(fill = "#E76F51"),
      width = unit(3, "cm"),
      add_numbers = TRUE,
      numbers_gp = gpar(fontsize = 9)
    ),
    annotation_name_side = "bottom",
    annotation_name_rot = 0,
    annotation_name_gp = gpar(fontsize = 11)
  ),
  pt_size = unit(4, "mm"),
  lwd = 2,
  comb_col = "black",
  bg_col = "grey95",
  bg_pt_col = "grey80",
  row_names_gp = gpar(fontsize = 11),
  column_title = NULL
)

dev.off()
