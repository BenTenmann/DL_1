library(tidyverse)
library(dplyr)
library(ggplot2)
library(plotrix)
library(colorspace)

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

read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/eps_test.csv")%>%
  ggplot(., aes(epsilon, final_error))+
  geom_path(color = "steelblue")+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())

df1 <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/noalpha.csv")
df2 <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/alpha.csv")

data.frame(error = c(df1$error, df2$error),
           epoch = rep(df1$epochs, 2),
           alpha = rep(c('yes', 'no'), each = length(df1$epochs)))%>%
  group_by(alpha)%>%
  ggplot(., aes(epoch, error, color=alpha))+
  geom_path()+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())


df <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/alpha_test.csv")
df$alpha <- as.factor(rep(seq(0.01, 1, 0.1), each = 500))

pal = sequential_hcl(10, palette = 'Plasma')
df %>%
  group_by(alpha) %>%
  ggplot(., aes(x=epochs, y=errors, color=alpha))+
  geom_path()+
  theme_minimal()+
  scale_color_manual(values = pal)+
  theme(axis.line = element_line(), axis.ticks = element_line())

library(data.table)
data <- fread("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MLP/epsilon_test.csv", sep = ",")

data$epsilon <- as.factor(rep(seq(0.1, 5.09, 0.01), each = 500))
pal2 <- sequential_hcl(length(seq(0.1, 5.09, 1)), palette = "Plasma")

data %>%
  filter(epsilon %in% as.factor(seq(0.1, 5.09, 1))) %>%
  group_by(epsilon)%>%
  ggplot(., aes(x=epochs, y=errors, color=epsilon))+
  geom_path()+
  theme_minimal()+
  scale_color_manual(values = pal2)+
  theme(axis.line = element_line(), axis.ticks = element_line())


dat <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/perceptron/perc_out.csv")
data.frame(error = c(dat$err_train, dat$err_test),
           epoch = rep(seq(1,200,1), 2),
           type = rep(c('train', 'test'), each=200))%>%
  group_by(type)%>%
  ggplot(., aes(epoch, error, color=type))+
  geom_path()+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())




dat2 <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/perceptron/perc_out.csv")
data.frame(error = c(dat2$err_train, dat2$err_test, dat$err_train, dat$err_test),
           epoch = rep(seq(1,200,1), 4),
           type = rep(c('train_par', 'test_par', 'train', 'test'), each=200))%>%
  group_by(type)%>%
  ggplot(., aes(epoch, error, color=type))+
  geom_path()+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())


dada <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MNIST/hist_2.csv")
data.frame(data = c(dada$loss, dada$acc, dada$val_loss, dada$val_acc),
           type_A = rep(c('train', 'validation'), each=40),
           type_B = rep(rep(c('loss', 'acc'), each = 20), 2),
           ints = rep(rep(c(0.2625, 0.9287), each=20), 2),
           epoch = rep(seq(1,20,1), 4))%>%
  group_by(type_B)%>%
  ggplot(., aes(epoch, data, color=type_A))+
  geom_path()+
  theme_minimal()+
  scale_color_discrete(name='type')+
  theme(axis.line = element_line(), axis.ticks = element_line(), axis.title.y = element_blank())+
  facet_grid(type_B~.)+
  geom_hline(aes(yintercept=ints), linetype='dashed', color='black', alpha=0.6)



dad <- read.csv("/Users/benjamintenmann/Desktop/CompBio/Assignments/DL_1/MNIST/hist_cnn.csv")
data.frame(data = c(dad$val_loss, dad$val_accuracy),
           type_B = rep(c('loss', 'acc'), each = 60),
           typ = rep(rep(c('Adam', 'Adagrad', 'RMSprop'), each = 20), 2),
           epoch = rep(rep(seq(1,20,1), 3), 2))%>%
  group_by(type_B)%>%
  ggplot(., aes(x=epoch, y=data, color=typ))+
  geom_path()+
  scale_color_discrete(name='optimizer')+
  theme(axis.line = element_line(), axis.ticks = element_line(), axis.title.y = element_blank())+
  facet_grid(type_B~.)

data.frame(accuracy = c(0.9267, 0.8623, 0.9272, 0.9834, 0.9219, 0.9817, 0.9892, 0.9387, 0.9892),
           optimizer = rep(c('Adam', 'Adagrad', 'RMSprop'), 3),
           model = rep(c('perceptron', 'MLP', 'CNN'), each = 3))%>%
  group_by(model)%>%
  ggplot(., aes(optimizer, accuracy, fill=model))+
  geom_bar(stat='identity', position = 'dodge')+
  scale_fill_brewer(palette = 'Paired')+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())

data.frame(accuracy = c(0.9834, 0.9219, 0.9817),
           optimizer = c('Adam', 'Adagrad', 'RMSprop'))%>%
  ggplot(., aes(optimizer, accuracy, fill=optimizer))+
  geom_bar(stat='identity')+
  theme_minimal()+
  theme(axis.line = element_line(), axis.ticks = element_line())

data.frame(data = c(561, 275, 162, 9, 106, 3),
           type_a = rep(c('misclassified', 'unclassified'), 3),
           type_b = rep(c('perceptron', 'MLP', 'CNN'), each = 2))%>%
  group_by(type_b)%>%
  ggplot(., aes(type_b, data, fill=type_a))+
  geom_bar(stat='identity', position = 'dodge')+
  theme_minimal()+
  scale_fill_brewer(palette = 'Paired', name='error type')+
  xlab('model')+
  ylab('number of digits')+
  theme(axis.line = element_line(), axis.ticks = element_line())
