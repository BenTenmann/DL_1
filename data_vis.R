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
