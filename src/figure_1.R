library(tidyverse)
library(ComplexUpset)

therapy_labels <- c(
  "cbt" = "CBT",
  "dbt" = "DBT",
  "act" = "ACT",
  "motivational_interviewing" = "MI",
  "mindfulness" = "Mindfulness",
  "stages_of_change" = "SoC",
  "family_systems" = "FS"
)

# Prepare data: convert to logicals
upset_data <- data %>%
  select(all_of(names(therapy_labels))) %>%
  mutate(across(everything(), ~ . == 1))

# Rename columns
colnames(upset_data) <- therapy_labels
upset_data <- upset_data[rowSums(upset_data) > 0, ]

# Create UpSet plot
upset_plot <- upset(
  data = upset_data,
  sets = colnames(upset_data),         # <-- use `sets=`, not positional
  name = 'Therapy Modality',
  min_size = 10,
  width_ratio = 0.1,
  height_ratio = 1,
  sort_sets = 'descending',
  sort_intersections_by = 'degree',
  n_intersections = 40,
  
  base_annotations = list(
    'Intersection size' = intersection_size(
      text_mapping = aes(label = !!upset_text_percentage(digits = 0, sep = '')),
      text = list(size = 3)
    ) + ylab('Intersection Size')
  ),
  
  set_sizes = (
    upset_set_size() +
      geom_bar(fill = 'steelblue', width = 0.7) +
      theme(axis.text.x = element_text(angle = 90))
  ),
  
  matrix = intersection_matrix(
    geom = geom_point(shape = 19, size = 4, color = 'black'),
    segment = geom_segment(color = 'black', linewidth = 1)
  ),
  
  stripes = c('grey95', 'white')
)

# Display and save
print(upset_plot)
ggsave("upset_treatments.png", plot = upset_plot, width = 12, height = 8, dpi = 300)