
# Prepare data with monthly aggregation
improvement_over_time <- data %>%
  mutate(
    year_month = floor_date(as.Date(admission_date), "month")
  ) %>%
  group_by(year_month) %>%
  summarise(
    improvement_rate = mean(improve, na.rm = TRUE),
    sample_size = n(),
    se = sqrt(improvement_rate * (1 - improvement_rate) / sample_size),
    ci_lower = improvement_rate - 1.96 * se,
    ci_upper = improvement_rate + 1.96 * se
  ) %>%
  ungroup()

# Calculate the 80/20 temporal split date
split_n <- floor(0.8 * nrow(data))
split_date <- as.Date(sort(data$admission_date)[split_n])

# Calculate period averages
training_avg <- data %>%
  filter(as.Date(admission_date) < split_date) %>%
  summarise(avg = mean(improve, na.rm = TRUE)) %>%
  pull(avg)

test_avg <- data %>%
  filter(as.Date(admission_date) >= split_date) %>%
  summarise(avg = mean(improve, na.rm = TRUE)) %>%
  pull(avg)

# Get date ranges for positioning annotations
date_range <- range(improvement_over_time$year_month)
training_mid <- as.Date(min(improvement_over_time$year_month)) + 
  as.numeric(split_date - min(improvement_over_time$year_month)) / 2
test_mid <- split_date + 
  as.numeric(max(improvement_over_time$year_month) - split_date) / 2

# Create the plot
p <- ggplot(improvement_over_time, aes(x = year_month, y = improvement_rate)) +
  # Add horizontal lines for period averages
  geom_segment(aes(x = min(year_month), xend = split_date, 
                   y = training_avg, yend = training_avg),
               linetype = "dotted", color = "#2980B9", linewidth = 0.8, alpha = 0.8) +
  geom_segment(aes(x = split_date, xend = max(year_month), 
                   y = test_avg, yend = test_avg),
               linetype = "dotted", color = "#27AE60", linewidth = 0.8, alpha = 0.8) +
  
  # Add vertical line for train/test split
  geom_vline(xintercept = split_date, 
             linetype = "dashed", color = "#E74C3C", linewidth = 0.8, alpha = 0.7) +
  
  # Add shaded confidence interval
  geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), 
              alpha = 0.2, fill = "#3498DB") +
  
  # Add smoothed trend line
  geom_smooth(method = "loess", se = FALSE, color = "#2C3E50", 
              linewidth = 1, linetype = "solid") +
  
  # Add monthly points sized by sample size
  geom_point(color = "#3498DB", alpha = 0.7) +
  
  # Add connecting line
  geom_line(color = "#3498DB", alpha = 0.4, linewidth = 0.5) +
  
  
  # Annotations for training period average
  annotate("text", x = training_mid, y = training_avg,
           label = sprintf("Training Mean\n%.1f%%", training_avg * 100),
           color = "black", size = 3.5, fontface = "bold", 
           hjust = 0.5, vjust = -1.5) +
  
  # Annotations for test period average
  annotate("text", x = test_mid, y = test_avg,
           label = sprintf("Test Mean\n%.1f%%", test_avg * 100),
           color = "black", size = 3.5, fontface = "bold", 
           hjust = 0.5, vjust = -1.5) +
  
  # Labels and theme
  labs(
    x = "Admission Date",
    y = "Proportion Achieving Risk Reduction",
    size = "Monthly\nSample Size",
  ) +
  
  scale_y_continuous(
    labels = percent_format(accuracy = 1),
    limits = c(0.3, 1),
    breaks = seq(0, 1, 0.1)
  ) +
  
  scale_x_date(
    date_breaks = "3 months",
    date_labels = "%b\n%Y"
  ) +
  
  scale_size_continuous(range = c(2, 8)) +
  
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 14, margin = margin(b = 5)),
    plot.subtitle = element_text(size = 11, color = "#555555", margin = margin(b = 15)),
    plot.caption = element_text(size = 8, color = "#666666", hjust = 0, margin = margin(t = 10)),
    axis.title = element_text(face = "bold", size = 10),
    axis.text = element_text(size = 9),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#E0E0E0", linewidth = 0.3),
    legend.position = "right",
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),
    plot.margin = margin(15, 15, 15, 15)
  )

print(p)

ggsave("outputs/figures/figure_e2.png", p, 
       width = 7, height = 4, dpi = 300, bg = "white")
