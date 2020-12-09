library(tidyverse)
library(dplyr)
library(ggplot2)

# plotting MSE v epoch
read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/perceptron/ep_df.csv")%>%
  ggplot(., aes(x=epochs, y=error))+
  geom_path(colour = "steelblue")+
  theme_minimal()+
  ylab("mean-squared error (MSE)")+
  theme(axis.line = element_line(), axis.ticks = element_line())

data.frame(x = c(1, 1, -1, -1),
           y = c(1, -1, 1, -1),
           parity = c('+', '-', '-', '+')) %>%
  ggplot(., aes(x=x, y=y, color=parity))+
  geom_point(size = 4)+
  xlab(expression(paste('x'[1], sep='')))+
  ylab(expression(paste('x'[2], sep='')))+
  theme_minimal()+
  geom_hline(yintercept = 0)+
  geom_vline(xintercept = 0)
  #theme(axis.line = element_line())
