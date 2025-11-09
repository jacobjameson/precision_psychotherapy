# ============================================================================
# ROC CURVE VISUALIZATION
# ============================================================================

colors <- c(
  "Patient Features" = "#10B981",
  "Therapist Factors" = "#F59E0B",
  "Combined" = "#6366F1"
)

plot_list <- list()

for(i in seq_along(results)) {
  res <- results[[i]]
  
  plot_data <- rbind(
    data.frame(
      fpr = res$patient$fpr, 
      tpr = res$patient$tpr,
      model = "Patient Features",
      auc = res$patient$auc
    ),
    data.frame(
      fpr = res$therapist$fpr, 
      tpr = res$therapist$tpr,
      model = "Therapist Factors",
      auc = res$therapist$auc
    ),
    data.frame(
      fpr = res$combined$fpr, 
      tpr = res$combined$tpr,
      model = "Combined",
      auc = res$combined$auc
    )
  )
  
  # Set factor order for legend
  plot_data$model <- factor(
    plot_data$model,
    levels = c("Patient Features", "Therapist Factors", "Combined")
  )
  
  # Get p-value for annotation
  p_val <- res$delong_tests$t_vs_p$p_value
  p_text <- if(!is.na(p_val)) {
    if(p_val < 0.001) "P<.001" else sprintf("P=%.3f", p_val)
  } else ""
  
  # Create the plot
  p <- ggplot(plot_data, aes(x = fpr, y = tpr, color = model)) +
    
    # Diagonal reference line
    geom_abline(intercept = 0, slope = 1, 
                linetype = "dotted", color = "black", size = 0.5) +
    
    # geom line with line types
    geom_line(size = 1) +
    
    
    
    # Colors
    scale_color_manual(
      values = colors,
      labels = sprintf("%s (AUC = %.2f)", 
                       names(colors), 
                       c(res$patient$auc, res$therapist$auc, res$combined$auc))
    ) +
    
    # Clean axes
    scale_x_continuous(
      "False Positive Rate", 
      breaks = seq(0, 1, 0.25),
      limits = c(0, 1),
      expand = c(0.01, 0.01)
    ) +
    scale_y_continuous(
      "True Positive Rate", 
      breaks = seq(0, 1, 0.25),
      limits = c(0, 1),
      expand = c(0.01, 0.01)
    ) +
    
    # Title and theme
    labs(
      title = res$therapy,
      x = "False Positive Rate",
      y = "True Positive Rate",
      color = NULL,
      linetype = NULL,
    ) +
    theme_bw(base_size = 11) +
    theme(
      # Title
      plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
      
      # Axes
      axis.title = element_text(size = 11),
      axis.text = element_text(size = 10, color = "black"),
      axis.ticks = element_line(color = "black"),
      
      # Legend
      legend.position = c(0.98, 0.02),
      legend.justification = c(1, 0),
      legend.background = element_rect(fill = "white", color = "black", size = 0.3),
      legend.text = element_text(size = 9),
      legend.key.size = unit(0.5, "lines"),
      legend.margin = margin(2, 2, 2, 2),
      
      # Square aspect
      aspect.ratio = 1
    ) +
    coord_fixed()
  plot_list[[i]] <- p
}

combined_plot <- do.call(gridExtra::grid.arrange, c(
  plot_list, 
  ncol = 4
))

ggsave(
  "outputs/figures/figure2.png",
  combined_plot,
  width = 13,
  height = 7,
  dpi = 300,
  bg = "white"
)