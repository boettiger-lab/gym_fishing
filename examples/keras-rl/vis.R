fs::file_delete("fishing-100.csv")

library(reticulate)
#reticulate::py_discover_config()
#reticulate::use_virtualenv("/opt/venv/keras-rl")
reticulate::source_python("fishing_discrete.py")

library(tidyverse)
fishing <- read_csv("fishing.csv", 
                    col_names = c("time", "state", "harvest", "action"))

d <- max(fishing$time)
n <-dim(fishing)[1] / d

fishing$rep <- as.character(vapply(1:n, rep, integer(d), d))

## Reward is calculated as net (cumulative) reward without any discounting (gamma = 1)
gamma <- 1.0
price <- 1.0
fishing <- fishing %>% 
  group_by(rep) %>% 
  mutate(reward = cumsum(price * harvest * gamma^time))

fishing %>% summarize(max(reward))

fishing %>% 
  filter(time < 100) %>%
  ggplot(aes(time, state, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 150) %>%
  ggplot(aes(time, harvest, col = rep)) + geom_line() + facet_wrap(~rep)

fishing %>% 
  filter(time < 30) %>%
  ggplot(aes(time, action, col = rep)) + geom_point() + facet_wrap(~rep)

fishing %>% 
  filter(time < 30) %>%
  ggplot(aes(time, reward, col = rep)) + geom_point() + facet_wrap(~rep)


