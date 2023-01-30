setwd("C:/Users/fra_m/OneDrive/Desktop/Business Data Analytics/Assessment")
library(paran)
library(psych)
library(car)
library(reshape2)
library(Rcmdr)
library(FSA)
library(sjPlot)
library(lsr)
library(heplots)
library(dplyr)
library(gridExtra)
library(ggplot2)

# Data handling ----------------------------------------------------------------

df = read.csv('Data1.csv',header = TRUE)
df =  na.omit(df) # if you look to dimensions its clear there where no NA

# Handling the dataset by making independant variables categorical
df$instr = as.factor(df$instr)
df$class = as.factor(df$class)
df$nb.repeat = as.factor(df$nb.repeat)
df$attendance = as.factor(df$attendance)
df$difficulty = as.factor(df$difficulty)
str(df)

# Principal components Analysis ----------------------------------------------------------------------

names(df[6]) # chech which column is the first question because the analysis 
# is performed just on the survey questions in order to find out latent variables
df_pca = df[,6:33]
# scling the dataset to work just on question related varibales

KMO(df_pca) #Kaiser-Meyer-Olkin Test 

# MSA > 0.7 so there is enough relevance for me to say that there is
# enough redundancy to justify the application of dimensionality reduction methods


#Fit a full model to check scree plot
Scree_plot <- prcomp(df_pca)
screeplot(Scree_plot, type = "lines")

# For doublechecking I'll perform a parallel analysis 
paran(df_pca)

# The result of the preliminar analysis shows the presence of two main components
# so its performed a PCA analysis to have two output components
fit_pca = pca(df_pca,2)
fit_pca 

#first 12 question load into component #2 --> student satisfaction about course, 
#the last 16 load into component #1 --> Instructor satisfaction

# I include the two components in the dataset in order to use them as variables

df$score_ins = fit_pca$scores[,1] # Instructor rating score
df$score_course = fit_pca$scores[,2] # Student satisfaction score

#-------------------------------------------------------------------------------
# Parametric assumptions check 

# Normality check 
shapiro.test(df$score_ins)  #doesn't work because the sample  size is > 5000 
qqnorm(df$score_ins) # use qualitative analysis --> condition not met
plot(density(df$score_ins), main="Density_score_ins")

qqnorm(df$score_course) # condition not met
plot(density(df$score_course), "Density_score_course")

# is clear that there are some observation that cause peaks in the distribution
# can be due to the presence of zero variance observations

df_pca$var_q = apply(df_pca,1,var) # calculate for each row its variance and storing it in a new variable
df_pca$no_var = ifelse(df_pca$var_q == 0, 1,0) # defining a binary variable for the identification the 0 variance observations
to_rm = which(df_pca$no_var == 1) # extracting the rows with no variance
df_norm = df[-to_rm,]# removing 0 variance obs from the dataset
write.csv(df_norm,'exploratory.csv') # exporting the final manipulated dataset

plot(density(df_norm$score_ins),main="Density_score_ins")
qqnorm(df_norm$score_ins) # condition now met

plot(density(df_norm$score_course),main="Density_score_stud")
qqnorm(df_norm$score_course) # condition now met

chisq.test(df_norm$difficulty,df_norm$nb.repeat) 
# chi-squared test to support assumption in Appendix I

# Homogeneity of variance --> according to Levene's test is not met in neither of
# the variables, but we looking at the boxplots we can assume that variance is 
# fairly homogeneous

leveneTest(df_norm$score_ins ~ df_norm$attendance)
boxplot(df_norm$score_ins ~ df_norm$attendance)

leveneTest(df_norm$score_course~ df_norm$attendance)
boxplot(df_norm$score_course ~ df_norm$attendance)

leveneTest(df_norm$score_ins ~ df_norm$difficulty)
boxplot(df_norm$score_ins ~ df_norm$difficulty)

leveneTest(df_norm$score_course ~ df_norm$difficulty)
boxplot(df_norm$score_course ~ df_norm$difficulty)

# Group sizes --> ratio between larger and smaller group (difficulty) is 
# approximately 3 which can be considered remarkable so by reading 
# at the result it must be taken in consideration that could be 
# biased by this size difference

table(df_norm$difficulty)
table(df_norm$attendance)
table(df_norm$instr)

# Homogeneity of covariances  <----------------------
boxM(cbind(score_ins,score_course) ~ difficulty, data = df_norm) 
boxM(cbind(score_ins,score_course) ~ attendance, data = df_norm)
# assumption do not met in any of the two cases so results must 
# be presented cautiosly 

# ----------------------------------------------------------------------------------

# Analysis n.1
# I'll use a MANOVA model in order to understand if the perceived difficulty and 
# attendance level have an effect on the scores
fit = manova(cbind(score_ins,score_course) ~ difficulty*attendance, data = df_norm)
summary(fit) #checking the significance of effects overall
summary.aov(fit) # individual ANOVAs 

# they affect significantly both student satisfaction score and instructor score 
# so I proceed in a further analysis for both dependant variables using 
# individual ANOVAs

fit_ins = aov(score_ins ~ difficulty*attendance, data=df_norm)
summary(fit_ins) #checking for the significance of the effect
etaSquared(fit_ins) # checking for the magnitude of the effect

fit_stud = aov(score_course ~ difficulty*attendance, data=df_norm)
summary(fit_stud)
etaSquared(fit_stud) 

# having the resuts of individual ANOVAs is possible to proceed with a 
# post hoc analysis to determine where the differences lies

TukeyHSD(fit_ins)
TukeyHSD(fit_stud) 

#-----------------------------------------------------------------------------------
# studying the linear effect of the two independant variables on the two dependant

lm_ins = lm(score_ins ~ difficulty+attendance, data=df_norm)
summary(lm_ins)

lm_stud = lm(score_course ~ difficulty, data=df_norm)
summary(lm_stud)

#---------------------------------------------------------------------------------
# CI plot using ggplot and dplyr
diffvscourse = df_norm %>% group_by(difficulty) %>% 
  summarize(mean = mean(score_course), sd = sd(score_course)/sqrt(n()), lower_bound = mean - 1.96*sd, upper_bound = mean + 1.96*sd )
plot_1 = ggplot (diffvscourse, aes(x = difficulty, y = mean, group = 1)) +
  geom_errorbar(aes(ymin = lower_bound, ymax= upper_bound), width = 0.2, size = 1, color = 'forest green') +
  geom_line(size = 1, color = 'forest green') +
  geom_point(size = 2, color ='dark green' ) +
  ylab ('student satisfaction about the course') + 
  xlab('Difficulty') +
  scale_x_discrete(labels = c("Very Easy", "Easy", "Medium", "Difficult", "Very Difficult"))

diffvsins = df_norm %>% group_by(difficulty) %>% 
  summarize(mean = mean(score_ins), sd = sd(score_ins)/sqrt(n()), lower_bound = mean - 1.96*sd, upper_bound = mean + 1.96*sd )
plot_2 = ggplot (diffvsins, aes(x = difficulty, y = mean, group = 1)) +
  geom_errorbar(aes(ymin = lower_bound, ymax= upper_bound), width = 0.2, size = 1, color = 'red') +
  geom_line(size = 1, color = 'red' ) +
  geom_point(size = 2, color = 'dark red') +
  ylab ('student satisfaction about the instructor') + 
  xlab('Difficulty') +
  scale_x_discrete(labels = c("Very Easy", "Easy", "Medium", "Difficult", "Very Difficult"))

attvsins = df_norm %>% group_by(attendance) %>% 
  summarize(mean = mean(score_ins), sd = sd(score_ins)/sqrt(n()), lower_bound = mean - 1.96*sd, upper_bound = mean + 1.96*sd )
plot_3 = ggplot (attvsins, aes(x = attendance, y = mean, group = 1)) +
  geom_errorbar(aes(ymin = lower_bound, ymax= upper_bound), width = 0.2, size = 1, color = 'steelblue') +
  geom_line(size = 1, color = 'steelblue') +
  geom_point(size = 2, color = 'dark blue') +
  ylab ('student satisfaction about the instructor') + 
  xlab('Attendance') +
  scale_x_discrete(labels = c('No Att', 'Poor Att', 'Medium Att', 'Good Att', 'High Att'))

grid.arrange(plot_3, plot_2, nrow = 1, ncol = 2)
grid.arrange(plot_1, nrow = 1, ncol = 1)
  
  
  

       