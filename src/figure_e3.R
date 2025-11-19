# ============================================================================
# PERSONALIZATION GAIN VISUALIZATION
# ============================================================================

# Calculate binned statistics for overlay
bin_stats <- personalization_viz_data %>%
  mutate(prob_bin = cut(baseline_prob, breaks = 20)) %>%
  group_by(prob_bin) %>%
  summarise(
    mean_prob = mean(baseline_prob),
    mean_gain = mean(gain),
    median_gain = median(gain),
    q25 = quantile(gain, 0.25),
    q75 = quantile(gain, 0.75),
    n = n(),
    .groups = 'drop'
  ) %>%
  filter(n >= 5)

# Main plot
p_personalization <- ggplot(personalization_viz_data, 
                            aes(x = baseline_prob, y = gain)) +
  # Individual points colored by gain category
  geom_point(aes(color = gain_category, shape = initial_risk),
             alpha = 0.9, size = 1.5) +
  
  # Smoothed trend line (LOESS)
  geom_smooth(method = "loess", 
              color = "#2E86AB", 
              fill = "#2E86AB",
              alpha = 0.2,
              size = 1.2,
              se = TRUE) +
  
  
  # Reference line at zero
  geom_hline(yintercept = 0, linetype = "solid", 
             color = "gray30", size = 0.5) +
  
  # Threshold lines
  geom_hline(yintercept = 1, linetype = "dotted", 
             color = "gray50", size = 0.4, alpha = 0.7) +
  geom_hline(yintercept = 5, linetype = "dotted", 
             color = "gray50", size = 0.4, alpha = 0.7) +
  
  # Color scale
  scale_color_manual(
    name = "Predicted Personalization\nGain",
    values = c("None" = "#CCCCCC",
               "Minimal (0-1pp)" = "#F4A261",
               "Moderate (1-5pp)" = "#E76F51",
               "Large (>5pp)" = "#C1121F")
  ) +
  
  # Shape scale
  scale_shape_manual(
    name = "Initial Risk",
    values = c("Moderate" = 16, "High" = 17)
  ) +
  
  # Scales
  scale_x_continuous(
    labels = percent_format(accuracy = 1),
    breaks = seq(0, 1, by = 0.1),
    expand = c(0.01, 0.01)
  ) +
  scale_y_continuous(
    breaks = seq(-2, 20, by = 2),
    expand = c(0.02, 0.02)
  ) +
  
  # Labels
  labs(
    title = "Heterogeneity in Personalized Therapy Selection Benefits",
    x = "Baseline Predicted Probability of Improvement",
    y = "Predicted Personalization Gain \n(percentage points)"
  ) +
  
  # Theme
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text( size = 14, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "gray30", hjust = 0, 
                                 margin = margin(b = 15)),
    plot.caption = element_text(size = 9, color = "gray50", hjust = 0,
                                margin = margin(t = 10)),
    axis.title = element_text(size = 11,),
    axis.text = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "white", color = "gray80"),
    legend.key.size = unit(0.8, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", size = 0.3),
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
    plot.margin = margin(15, 15, 15, 15)
  )


# ============================================================================
# ADDITIONAL PANEL: DISTRIBUTION BY RISK GROUP
# ============================================================================

# Create violin/box plot by risk group
p_by_risk <- ggplot(personalization_viz_data, 
                    aes(x = initial_risk, y = gain, fill = initial_risk)) +
  geom_violin(alpha = 0.6, scale = "width") +
  geom_boxplot(width = 0.2, alpha = 0.8, outlier.size = 1) +
  geom_hline(yintercept = 0, linetype = "solid", color = "gray30") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", alpha = 0.5) +
  scale_fill_manual(
    values = c("Moderate" = "#F18F01", "High" = "#A23B72"),
    guide = "none"
  ) +
  labs(
    title = "Predicted Personalization Gains by Initial Risk Level",
    x = "Initial Risk Level",
    y = "Predicted Personalization Gain \n(percentage points)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text( size = 14, hjust = 0),
    plot.subtitle = element_text(size = 11, color = "gray30", hjust = 0, 
                                 margin = margin(b = 15)),
    plot.caption = element_text(size = 9, color = "gray50", hjust = 0,
                                margin = margin(t = 10)),
    axis.title = element_text(size = 11,),
    axis.text = element_text(size = 10, color = "black"),
    legend.position = "right",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    legend.background = element_rect(fill = "white", color = "gray80"),
    legend.key.size = unit(0.8, "cm"),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "gray90", size = 0.3),
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
    plot.margin = margin(15, 15, 15, 15)
  )


# make cowplot of main plot and additional panel
p_combined <- cowplot::plot_grid(
  p_personalization,
  p_by_risk + theme(legend.position = "none"),
  ncol = 1,
  rel_heights = c(1, 1),
  labels = c("A", "B"),
  label_size = 14,
  align = "v"
)

p_combined

ggsave(
  plot = p_combined,
  filename = "outputs/figures/figure_e3.png",
  width = 7,
  height = 8,
  dpi = 300, bg = "white"
)

ggsave(
  plot = p_combined,
  filename = "outputs/figures/figure_e3.pdf",
  width = 7,
  height = 8,
  dpi = 300, bg = "white"
)

