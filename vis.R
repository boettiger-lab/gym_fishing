library(tidyverse)

fishing <- read_csv("fishing_discrete.csv", col_names = c("time", "state", "harvest"))
d <-dim(fishing)[1] / 5
fishing$rep <- as.character(vapply(1:5, rep, integer(d), d))

fishing %>% 
  #filter(time < 200) %>%
  ggplot(aes(time, state, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 50) %>%
  ggplot(aes(time, harvest, col = rep)) + geom_line() + facet_wrap(~rep)





fishing <- read_csv("fishing.csv", col_names = c("time", "state", "harvest"))
d <-dim(fishing)[1] / 5
fishing$rep <- as.character(vapply(1:5, rep, integer(d), d))

fishing %>% 
  filter(time < 400) %>%
  ggplot(aes(time, state, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 400) %>%
  ggplot(aes(time, harvest, col = rep)) + geom_line() + facet_wrap(~rep)
