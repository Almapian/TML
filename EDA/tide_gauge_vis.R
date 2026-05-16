library(tidyverse)
library(sf)
library(ggrepel)
library(ggspatial)
library(patchwork)
source("./utils/mapper.R")

# ---- Gauges, numbered west -> east in BNG ----
gauges <- read.csv("./3_Cleaned/00_gauge_metadata.csv") %>%
  select(SiteName, Easting, Northing) %>%
  st_as_sf(coords = c("Easting", "Northing"), crs = 27700) %>%
  mutate(east = st_coordinates(.)[, 1]) %>%
  arrange(east) %>%
  mutate(id    = row_number(),
         label = paste0(id, ". ", SiteName))

# ---- Main map: full Thames extent ----
bb <- st_bbox(gauges) + c(-15000, -10000, 15000, 15000)

main <- ggplot() +
  annotation_map_tile(type = "cartolight", zoomin = -1, progress = "none") +
  geom_sf(data = gauges, size = 2, colour = "black") +
  geom_label_repel(
    data = gauges,
    aes(geometry = geometry, label = label),
    stat               = "sf_coordinates",
    size               = 2.8,
    label.padding      = unit(0.15, "lines"),
    box.padding        = 0.45,
    point.padding      = 0.25,
    segment.colour     = "grey25",
    segment.size       = 0.25,
    min.segment.length = 0,
    max.overlaps       = Inf,
    seed               = 1
  ) +
  coord_sf(
    xlim   = c(bb["xmin"], bb["xmax"]),
    ylim   = c(bb["ymin"], bb["ymax"]),
    crs    = 27700,
    datum  = sf::st_crs(27700),   # <- BNG graticules
    expand = FALSE
  ) +
  bng_labels() +
  map_decorations() +
  theme_thames_map() +
  labs(title = "Thames tide gauge network")

# ---- London inset ----
london_ids <- 1:4   # adjust after inspecting `gauges`
london_box <- st_bbox(gauges %>% filter(id %in% london_ids)) +
  c(-3000, -2000, 3000, 2000)

# Which gauges live in insets (so they shouldn't be labelled on the main map)
inset_ids <- c(1:4, 9:11, 15:16)

# ---- Main map: label only non-inset gauges ----
bb <- st_bbox(gauges) + c(-15000, -10000, 15000, 15000)

main <- ggplot() +
  annotation_map_tile(type = "cartolight", zoomin = -1, progress = "none") +
  geom_sf(data = gauges, size = 2, colour = "black") +
  geom_label_repel(
    data = gauges %>% filter(!id %in% inset_ids),     # <- key change
    aes(geometry = geometry, label = label),
    stat               = "sf_coordinates",
    size               = 2.8,
    label.padding      = unit(0.15, "lines"),
    box.padding        = 0.45,
    point.padding      = 0.25,
    segment.colour     = "grey25",
    segment.size       = 0.25,
    min.segment.length = 0,
    max.overlaps       = Inf,
    seed               = 1
  ) +
  coord_sf(
    xlim = c(bb["xmin"], bb["xmax"]),
    ylim = c(bb["ymin"], bb["ymax"]),
    crs = 27700, datum = sf::st_crs(27700),
    expand = FALSE
  ) +
  bng_labels() +
  map_decorations() +
  theme_thames_map() +
  labs(title = "Thames tide gauge network")

# ---- Three insets ----
london_inset  <- make_inset(gauges, ids = 1:4,   buffer = 3000, title = "Inset: Greater London")
coryton_inset <- make_inset(gauges, ids = 9:11,  buffer = 2000, title = "Inset: Coryton")
margate_inset <- make_inset(gauges, ids = 15:16, buffer = 1000, title = "Inset: Margate")

# ---- Combine: main on left, insets stacked on right ----
final <- main +
  (london_inset / coryton_inset / margate_inset) +
  plot_layout(widths = c(2.2, 1))

ggsave("./outputs/thames_tide_gauges.pdf", final,
       width = 11, height = 9, device = cairo_pdf)
ggsave("./outputs/thames_tide_gauges.png", final,
       width = 11, height = 9, dpi = 300)
