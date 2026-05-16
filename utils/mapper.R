library(ggplot2)
library(ggspatial)
library(sf)

# Decorations: bar-style scale (bottom right) + orienteering compass (top left)
map_decorations <- function(arrow_loc = "tl", scale_loc = "br") {
  list(
    annotation_scale(
      location = scale_loc,
      style    = "bar",
      bar_cols = c("black", "white"),
      height   = unit(0.2, "cm"),
      text_cex = 0.7
    ),
    annotation_north_arrow(
      location    = arrow_loc,
      which_north = "true",
      style       = north_arrow_fancy_orienteering(),
      height      = unit(1.2, "cm"),
      width       = unit(1.0, "cm")
    )
  )
}

# BNG axis labels: meters -> "X km" with thousands separator
bng_labels <- function() {
  list(
    scale_x_continuous(
      labels = function(x) paste0(format(x / 1000, big.mark = ","), " km")
    ),
    scale_y_continuous(
      labels = function(x) paste0(format(x / 1000, big.mark = ","), " km")
    )
  )
}

# Minimal theme — keeps grid + axis labels visible, no axis titles
theme_thames_map <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid  = element_line(colour = "grey80", linewidth = 0.2),
      axis.title  = element_blank(),
      axis.text   = element_text(colour = "grey30", size = base_size - 2),
      plot.title  = element_text(face = "bold", size = base_size + 1)
    )
}

make_inset <- function(gauges, ids, buffer = 2000, title, base_size = 8) {
  bx <- sf::st_bbox(gauges %>% dplyr::filter(id %in% ids)) +
    c(-buffer, -buffer, buffer, buffer)

  ggplot() +
    annotation_map_tile(type = "cartolight", zoomin = 0, progress = "none") +
    geom_sf(data = gauges, size = 1.8, colour = "black") +
    ggrepel::geom_label_repel(
      data = gauges %>% dplyr::filter(id %in% ids),
      aes(geometry = geometry, label = label),
      stat               = "sf_coordinates",
      size               = 2.4,
      box.padding        = 0.3,
      segment.size       = 0.2,
      min.segment.length = 0,
      max.overlaps       = Inf,
      seed               = 1
    ) +
    coord_sf(
      xlim   = c(bx["xmin"], bx["xmax"]),
      ylim   = c(bx["ymin"], bx["ymax"]),
      crs    = 27700,
      datum  = sf::st_crs(27700),
      expand = FALSE
    ) +
    bng_labels() +
    map_decorations() +
    theme_thames_map(base_size = base_size) +
    labs(title = title)
}
