library(tidyverse)

fishing <- read_csv("fishing.csv", 
                    col_names = c("time", "state", "harvest", "action", "reward"))
d <- max(fishing$time)
n <-dim(fishing)[1] / d

fishing$rep <- as.character(vapply(1:n, rep, integer(d), d))

fishing %>% 
  #filter(time < 200) %>%
  ggplot(aes(time, state, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 150) %>%
  ggplot(aes(time, harvest, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 150) %>%
  ggplot(aes(time, action, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 150) %>%
  ggplot(aes(time, reward, col = rep)) + geom_line() + facet_wrap(~rep)


