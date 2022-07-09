## ----echo=FALSE-----------------------------------------------------------------
options(width = 80L)


## ----require_dplyr,echo=FALSE---------------------------------------------------
require(dplyr)


## ----install_dplyr,eval=FALSE---------------------------------------------------
## # This first line of code is required only 1x when you
## # need to install a package for the *first* time.
## install.packages("dplyr")
## # This next line of code is required *every* time you open
## # up R and want to *use* the package.
## require(dplyr)


## ----10k_foot_overview, textsize="small",fig.height=3,fig.width=8,eval=FALSE,results="hide"----
## webSite <- 'https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ozone.data'
## # Load the data into memory, fit a linear model, summarize results.
## read.csv(webSite, sep = '\t', header = TRUE) %>%
##   lm(formula = ozone ~ radiation) %>%
##   summary


## ----setupggplot2,eval=FALSE----------------------------------------------------
## install.packages("ggplot2")


## ----loadggplot2----------------------------------------------------------------
require(ggplot2)


## ----head_of_mpg----------------------------------------------------------------
head(mpg)


## ----barplotofmanufacturers, fig.height=2---------------------------------------
ggplot(data = mpg, aes(x = manufacturer)) +
    geom_bar()


## ----pointcloud,fig.height=3----------------------------------------------------
ggplot(data = mpg, mapping = aes(x = displ, y = cty)) +
    geom_point()


## ----colouringplots,fig.height=3------------------------------------------------
ggplot(data = mpg, mapping = aes(x = displ, y = cty, colour = drv)) +
    geom_point()


## ----plot_with_overlapping_points,fig.height=3----------------------------------
ggplot(data = mpg, mapping = aes(x = displ, y = cty)) +
    geom_point() +
    geom_hline(yintercept = 10)


## ----geomjitter,fig.height=3----------------------------------------------------
ggplot(data = mpg, mapping = aes(x = displ, y = cty, colour = cty < 10)) +
    geom_jitter()


## ----geom_smooth_first_example,fig.height=2,message=FALSE-----------------------
ggplot(mpg, aes(x = displ, y = cty)) +
    geom_smooth()


## ----geom_smooth_second_separate_models,fig.height=3,message=FALSE--------------
ggplot(mpg, aes(x = displ, y = cty, colour = drv, linetype = drv)) +
    geom_smooth()


## ----head_of_diamonds,echo=FALSE------------------------------------------------
head(diamonds, 3)


## ----stat_identity,fig.height=2-------------------------------------------------
ggplot(diamonds, aes(x = cut)) +
    geom_bar()


## ----explicit_use_of_computed_geom_var,eval=FALSE-------------------------------
## ggplot(diamonds, aes(x = cut)) +
##     geom_bar(stat = "count")


## ----afterstat,eval=FALSE-------------------------------------------------------
## ggplot(diamonds, aes(x = cut, y = after_stat(count / sum(count)))) + geom_bar()


## ----first_calculation----------------------------------------------------------
10^2 + 36


## ----percentoflifenjoyingdegree,eval=FALSE--------------------------------------
## (2021 - 2017) / (2021 - 1989) * 100


## ----our_first_variable---------------------------------------------------------
a <- 4


## ----our_first_variable_calculation---------------------------------------------
a * 5  # We can perform calculations on a variable, e.g.


## ----reassign_variable----------------------------------------------------------
a <- a + 10
a


## ----eval=FALSE-----------------------------------------------------------------
## nYearsSinceGraduation / age * 100


## ----simple_concat--------------------------------------------------------------
vals <- c(4, 7, 10)


## ----manual_avg, eval=FALSE-----------------------------------------------------
## (4 + 7 + 10) / 3


## ----arithmetic_avg,results="hide"----------------------------------------------
mean(vals)  # Outputs the scalar value 7, of course!


## ----eval=FALSE-----------------------------------------------------------------
## myVector <- c(4, 5, 8, 11)
## sum(myVector)


## ----eval=FALSE,echo=FALSE------------------------------------------------------
## sum(1:100)


## ----random_normal,eval=FALSE---------------------------------------------------
## rnorm(10)


## ----normal_distribution_hist,fig.height=2.5,eval=1-----------------------------
rnorm(1e5) %>% 
    hist(breaks = 100, main = "A Standard Normal Distribution")
# In ggplot2 (and using data.frame's):
df <- data.frame(x = rnorm(1e5))
ggplot(df, aes(x)) + geom_histogram(bins = 100)


## ----ac_ts,fig.height= 2.5,eval=1-----------------------------------------------
rnorm(1e5) %>% cumsum %>% plot(type = "l", main = "Stock Market Simulation")
# Again, if we wanted to use ggplot2 we'd have to use data.frame's. Not disadvantageous.
df <- data.frame(x_var = 1:1e5, y_var = cumsum(rnorm(1e5)))
ggplot(df, aes(x = x_var, y = y_var)) + geom_line()


## ----gets,eval=FALSE------------------------------------------------------------
## # But it's true that in R, to assign a variable,  you can write either write:
## variable <- rnorm(10)
## variable = rnorm(10)  # <-- But! Writing this is also valid!


## ----equalsforargumentinfunction,eval=FALSE-------------------------------------
## sample(x = 100)  # Randomly permutes the first 100 positive integers


## ----lsexamplewithgets----------------------------------------------------------
rm(list=ls())  # Clear workspace (don't save this in your scripts).
result <- sample(x <- 10)
ls()


## ----eval=FALSE-----------------------------------------------------------------
## plot(runif(100, -1e3, 1e3))
## # df <- data.frame(x = 1:100, y = runif(100, -1e3, 1e3))
## # ggplot(df, aes(x, y)) + geom_point()


## ----downloadfoulballs,cache=TRUE-----------------------------------------------
download.file(
  url = file.path(
    "https://raw.githubusercontent.com",
    "fivethirtyeight/data/master/foul-balls",
    "foul-balls.csv"
  ),
  destfile = "~/Downloads/foul-balls.csv"
)


## ----loadfoulballs--------------------------------------------------------------
data <- read.csv("~/Downloads/foul-balls.csv")


## ----squish_matchup,echo=FALSE--------------------------------------------------
data$matchup <- gsub(".* (.*) vs .* (.*)$", "\\1 vs \\2", data$matchup, ignore.case = TRUE)


## ----row_subsetting_dfs_like_matrices-------------------------------------------
# Request the first 3 rows using *slicing*.
data %>% slice(1:3)
# Request the first 3 rows and the first 3 columns.
data %>% slice(1:3) %>% select(1:3)
# Request rows 2 and 5 as well as two columns by name.
data %>% slice(c(2,5)) %>% select(game_date, type_of_hit)


## ----using_dollar_sign_extraction-----------------------------------------------
table(data$matchup)


## ----dplyr_counting, eval=FALSE-------------------------------------------------
## data %>% group_by(matchup) %>% count()


## ----headfoulballs--------------------------------------------------------------
head(data, n = 4)


## ----dimensionsofdata, eval=FALSE-----------------------------------------------
## dim(data)
## str(data)
## summary(data)


## ----plotdf,eval=FALSE----------------------------------------------------------
## plot(data)


## ----dplyr_arrange, eval=FALSE--------------------------------------------------
## data %>% arrange(matchup)


## ----order_rows_desc------------------------------------------------------------
data %>% arrange(matchup, decreasing = TRUE) %>% {rbind(head(., n = 2), tail(., n = 2))}


## ----basic_filter---------------------------------------------------------------
filter(data, type_of_hit == "Ground") %>% head(n = 5)


## ----multiple_predicate_filter--------------------------------------------------
filter(data, type_of_hit == "Ground", exit_velocity > 90) %>% head(n = 5)


## ----or_predicate_filter,eval=FALSE---------------------------------------------
## filter(df, predicate_1 | predicate_2)


## ----multiple_predicate_filter_logical_and--------------------------------------
filter(data, type_of_hit == "Ground" & exit_velocity > 90) %>% head(n = 5)


## ----adding_vars_dfs------------------------------------------------------------
# Add a column by using integer position.
# Note that if we do this, the new variable name
# is of the vorm Vk, where k is an integer describing
# the column number.
data[, ncol(data) + 1] <- data$game_date

head(data[, c("game_date", "V8")], n = 3)

# We can also (more readably) assign to a new column
# and give it an interpretable name at the same time.
data[, "zone_gt_2"] <- data[, "used_zone"] > 2
data[1:5, "zone_gt_2"]

# We can also use our technique of using operator\$ to
# grab columns.
data$logged_exit_velocity <- log(data$exit_velocity, base = 2)
data %>%
  filter(!is.na(exit_velocity)) %>%
  select(ends_with("exit_velocity")) %>%
  head(n = 3)


## ----mutate_example-------------------------------------------------------------
data %>%
    mutate(transform_of_zones = predicted_zone + sqrt(camera_zone) + used_zone^2) %>%
    select(matchup, predicted_zone, camera_zone, used_zone, transform_of_zones) %>%
    filter(!is.na(transform_of_zones)) %>%
    arrange(transform_of_zones, decreasing = TRUE) %>%
    head(n = 3)


## ----mutate_example2------------------------------------------------------------
data %>%
    group_by(matchup) %>%
    mutate(
      last_game_played = lag(game_date),
      cumulative_velocity = cumsum(ifelse(is.na(exit_velocity), 0, exit_velocity))
    ) %>%
    select(matchup, game_date, last_game_played, exit_velocity, cumulative_velocity) %>%
    head(n = 4)


## ----confusionmatrixoffoulballs-------------------------------------------------
confusion_matrix <- table(predicted = data$predicted_zone, 
                          observed = data$camera_zone)
confusion_matrix[1:3, c(1:2, 4)]  # Just look at the first few rows and columns...


## -------------------------------------------------------------------------------
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
paste0("Accuracy = ", round(accuracy * 100, 2), "%")


## ----replacenasforexitvelocity--------------------------------------------------
missing_idx <- which(is.na(data$exit_velocity))
data[missing_idx, "exit_velocity"] <- mean(data$exit_velocity, na.rm = TRUE)


## ----readfoulballs,cache=TRUE---------------------------------------------------
link  <- paste0("https://raw.githubusercontent.com/",
                "fivethirtyeight/data/master/foul-balls/",
                "foul-balls.csv")
fouls <- read.csv(link)


## ----copypaste, eval=FALSE------------------------------------------------------
## data <- read.delim("clipbloard")    # Works on Windows, Linux
## data <- read.delim(pipe("pbpaste")) # Works on iOS.


## ----dplyr_group_by,messages=FALSE,warnings=FALSE-------------------------------
# install.packages("dplyr")
require(dplyr)
fouls %>%
    group_by(type_of_hit) %>%
    summarize(avg = mean(exit_velocity, na.rm = TRUE),
              std =   sd(exit_velocity, na.rm = TRUE))


## ----install_reshape2,eval=FALSE------------------------------------------------
## install.packages("reshape2")


## ----gather_example-------------------------------------------------------------
# Use "gather" to go from wide to long.
long <- fouls %>%
    group_by(type_of_hit) %>%
    # Ie we only executed summarize, we'd get one column per variable.
    summarize(avg = mean(exit_velocity, na.rm = TRUE),
              std =   sd(exit_velocity, na.rm = TRUE)) %>%
    # If we add a 'melt' command, we can collect values into a common column.
    reshape2::melt(id.vars = "type_of_hit")
print(long)


## ----reshape_wide---------------------------------------------------------------
# Now do the opposite, go from long to wide via pivot_wider.
# For this, I personally prefer reshape2 package.
wide <- reshape2::dcast(long, type_of_hit ~ variable)
print(wide)


## ----scrape_bshare, cache=TRUE, size="small"------------------------------------
# Download a zip file of bike-share data.
download.file(paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/00275/",
                     "Bike-Sharing-Dataset.zip"),
              destfile = "bikesharing.zip")
# Unzip the contents and create a corresponding data directory. Load, and inspect.
unzip(zipfile = "bikesharing.zip", exdir = "bikesharing")
bshare <- read.csv("bikesharing/hour.csv")
bshare %>% slice(1:3) %>% select(1:8)


## ----hist_rideshares, size="small", fig.height=2, fig.width=8-------------------
ggplot(bshare, aes(x = cnt)) +
    geom_histogram(bins = 50) +
    labs(title = "Histogram of Rideshares",
         x = "# Rideshares",
         y = "# Observations")


## ----seasonality_rideshares, fig.height=2, fig.width=8--------------------------
bshare %>%
    group_by(mnth) %>%
    summarise(ttl = sum(cnt)) %>%
    ggplot(aes(x = mnth, y = ttl)) +
    geom_line() +
    labs(title = "Seasonality in Bicycle Ridesharing",
         x = "Calendar Month",
         y = "# Rideshares")


## ----leaving_out_---------------------------------------------------------------
m <- lm(cnt ~ mnth + hr + holiday + weekday + workingday + temp + hum + windspeed,
        data = bshare)


## ----non_linear_reln_season, size="small", fig.height=2.2, fig.width=8----------
# Determine if any non-linear relationships were left out
# of the month feature.
bshare %>%
    mutate(residual = resid(m)) %>%
    group_by(mnth) %>%
    summarise(mean_residual = mean(residual)) %>%
    ggplot(aes(x = mnth, y = mean_residual)) +
    geom_line() +
    labs(title = "Rideshares don't follow a linear relationship with Month",
         x = "Month",
         y = "Average Prediction Error")


## ----non_linear_rect, fig.height=3, fig.width=8---------------------------------
m <- lm(cnt ~ I(factor(mnth)) + hr + holiday + weekday + workingday + temp + hum
        + windspeed, data = bshare)

# Verify non-linear relationships have been explicitly modeled.
bshare %>% 
    mutate(residual = resid(m)) %>% 
    group_by(mnth) %>% 
    summarise(mean_residual = mean(residual)) %>%
    summarise(all(mean_residual < 1e-7))


## ----fig.height=3, fig.width=8, fig.show="hold"---------------------------------
# Very that our linearity assumption was violated; we see clear commute and
# day-of-week effects.
bshare %>% 
    mutate(residual = resid(m)) %>% 
    group_by(hr) %>% 
    summarise(mean_residual = mean(residual)) %>%
    ggplot(aes(x = hr, y = mean_residual)) +
    geom_line() +
    labs(title = "We Forgot Commute-Effects",    
         x = "Hour",    
         y = "Average Residual")


## -------------------------------------------------------------------------------
m <- lm(cnt ~ I(factor(mnth)) + I(factor(hr)) + I(factor(weekday)) +
        I(factor(workingday)) + holiday + temp + hum + windspeed, data = bshare)
# summary(m)  <-- This prints useful output but its a bit verbose.


## ----spotting_outliers_phase_zero,fig.height=2,message=FALSE--------------------
ggplot(diamonds, aes(x = y)) +
    geom_histogram()


## ----spotting_outliers_phase_one,fig.height=2,message=FALSE---------------------
ggplot(diamonds, aes(x = y)) +
    geom_histogram() +
    # Truncate the y-axis to [0, 10] to help see the mass at x = 0, 32, and 59.
    coord_cartesian(ylim = c(0, 10))


## ----replacing_outliers,fig.height=2,warning=FALSE------------------------------
diamonds %>%
    mutate(y = ifelse(y < 3 | y > 20, NA, y)) %>%
    ggplot(aes(x = y)) +
    geom_histogram(bins = 20)


## ----vector_addition------------------------------------------------------------
u <- c(1, 2, 4, 8, 16)
v <- seq(from = 0, to = 1, length.out = length(u))  # (0, 1/4, 1/2, 3/4, 1)


## ----indexing_into_vectors------------------------------------------------------
u[3]        # R is 1-indexed.
u[c(3, 5)]  # Grabs the third and fifth element from 'u'.
u[3] <- 0   # We may access and overwrite individual elems.


## -------------------------------------------------------------------------------
rbind(u, v) # We can also row-bind two vectors to get a matrix; see below.
u + v       # Vector addition is performed element-wise.


## ----echo=FALSE-----------------------------------------------------------------
set.seed(1)


## ----logicalfilter--------------------------------------------------------------
vals <- rnorm(n = 100)
negs <- vals[vals < 0]  # Only keep 'vals' that are negative.
summary(negs)           # Notice the maximum is negative, by construction.


## ----headofvalsbinarycondition--------------------------------------------------
head(vals < 0)  # Notice that vals < 0 returns a Boolean vector.
data <- cbind(original = vals, predicate = vals < 0)
head(data)


## ----ifelse_example-------------------------------------------------------------
data <- 1:10
ifelse(data < 5, 0, data)


## ----first_matrix---------------------------------------------------------------
mat <- matrix(data = 1:9, nrow = 3)
print(mat)


## ----evalu=FALSE----------------------------------------------------------------
p <- 31:60
Q <- matrix(31:60, nrow = 5, ncol = 6)


## ----eval=FALSE, echo=FALSE-----------------------------------------------------
## p[15]
## Q[5,3]


## ----eval=FALSE,echo=FALSE------------------------------------------------------
## mean(p)
## mean(Q)


## ----eval=FALSE,echo=FALSE------------------------------------------------------
## colSums(Q)
## colMeans(Q)


## ----first_list,echo=c(1,3:4)---------------------------------------------------
l <- list(one = 1, two = 1:2, five = seq(0, 1, length.out = 5))
l
l$five + 10
l$newVar <- sin(l$five)  # We can assign a new element to a list on the fly.


## -------------------------------------------------------------------------------
# Create some salt and pepper packages.
# (Each contains a different number of particles and ratio of salt-pepper)
sap <- list(package_A = list(serv_one = rbinom(20, 1, prob = 50/100),
                             serv_two = rbinom(14, 1, prob = 30/100)),
            package_B = rbinom(28, 1, prob = 55/100),
            package_C = rbinom(36, 1, prob = 40/100))


## ----first_fun------------------------------------------------------------------
Greeting <- function(name, salutation = "Hello", punctuation = "!") {
    paste0(salutation, ", ", name, punctuation)
}


## ----greetings, results="hold"--------------------------------------------------
Greeting("Andreas",  salutation = "god morgen")
Greeting("Santucci", salutation = "buon giorno")


## ----equivgreeting--------------------------------------------------------------
Greeting <- function(name, salutation = "Hello", punctuation = "!") {
    return(paste0(salutation, ", ", name, punctuation))
}


## ----eval=FALSE-----------------------------------------------------------------
## MyFunction <- function(x) {}


## ----eval=FALSE-----------------------------------------------------------------
## MyFunction(1:10)


## ----eval=FALSE-----------------------------------------------------------------
## DaysSinceBorn <- function(birthdate = as.Date("1989-09-04")) {}


## ----my_square------------------------------------------------------------------
RSQU <- function(y, x)    cor(y, x)^2               # Univariate case.
RSQM <- function(y, yhat) 1 - sum((y - yhat)^2) /   # OLS: 1 - RSS / TSS.
                              sum((y - mean(y))^2)
RSQG <- function(y, yhat) cor(y, yhat)^2            # General model.


## ----eval=FALSE-----------------------------------------------------------------
## x <- c(1:10)
## x > 8


## ----check_id-------------------------------------------------------------------
CheckID <- function(birthdate, K = 18) {
    # Your job is to fill in a sub-routine here.
    # Return true only if birthdate at least K years.
}


## ----logical_indexing-----------------------------------------------------------
set.seed(20210813)
x <- runif(300)   # Sample from [0,1] uniformly at random.
idx <- x < 1/2    # In expectation, 1/2 of these evaluate TRUE.
# Not run: x[idx] (Return elements of 'x' for which 'idx' is TRUE.)


## -------------------------------------------------------------------------------
v = c(1,3,5,7,8)
which(v%%2 == 0)

## -------------------------------------------------------------------------------
which(v%%2 != 0)


## ----bool_arith,results="hold"--------------------------------------------------
sum(idx)
mean(idx)


## ----eval=FALSE-----------------------------------------------------------------
## FALSE + TRUE == 1L
## TRUE  + TRUE == 2L
## mean(c(FALSE, TRUE))
## # What is the value of the following expression, in expectation?
## mean(rnorm(1e3) < 0)


## ----reload_fouls,cache=TRUE----------------------------------------------------
link  <- paste0("https://raw.githubusercontent.com/",
                "fivethirtyeight/data/master/foul-balls/",
                "foul-balls.csv")
data <- read.csv(link)


## ----size="small"---------------------------------------------------------------
for (column in colnames(data))
    print(paste("Column", column, "has", 
                sum(is.na(data[[column]])), 
                "missing values"))


## ----concatenationforloopstrategy-----------------------------------------------
result <- c()
for (column in colnames(data))
    result <- c(result, sum(is.na(data[[column]])))


## ----preallocationforloopstrategy-----------------------------------------------
result <- vector(mode = "numeric", length = ncol(data))
for (i in seq_along(data))
    result[i] <- sum(is.na(data[[i]]))


## ----sapplyisna-----------------------------------------------------------------
sapply(colnames(data), function(colname)
    sum(is.na(data[[colname]])))


## -------------------------------------------------------------------------------
NUnique <- function(x) length(unique(x))
uniques <- sapply(fouls, NUnique)


## ----eval=FALSE-----------------------------------------------------------------
## NumericCols <- function(df) {}


## ----first_pipe_example,message=FALSE,warning=FALSE-----------------------------
require(magrittr)  # We load this package so we can use %>%.
cols <- sapply(fouls, is.numeric) %>% 
    Filter(f = identity) %>% 
    names()
transformed_cols = paste("transformed", cols)


## ----size="small"---------------------------------------------------------------
require(purrr)
sapply(fouls, compose(length, unique))


## ----eval=FALSE-----------------------------------------------------------------
## download.file(
##   file.path(
##     "https://data.bls.gov/cew/data/files/2018/csv/",
##     "2018_qtrly_by_area.zip"
##   ),
##   "~/Downloads/2018.q1-q4.by_area.zip"
## )
## unzip("~/Downloads/2018.q1-q4.by_area.zip",
##       exdir = "~/Downloads/2018.q1-q4.by_area")


## -------------------------------------------------------------------------------
# There are 4,428 files in the directory!
fnames <- list.files(path = "~/Downloads/2018.q1-q4.by_area/2018.q1-q4.by_area",
                     full.names = TRUE)
d <- lapply(fnames[3:5], read.csv) %>%  # Load several files...
    rbindlist()                         # ...and "stack" them together.


## ----avg_fouls_by_type_of_hit---------------------------------------------------
aggregate(
  x = fouls[["exit_velocity"]], 
  by = list(fouls$type_of_hit), 
  FUN = mean, 
  na.rm = TRUE
)


## ----avg_exit_velocity_by_type_of_hit_and_predicted_zone,size="small"-----------
aggregate(
  x = fouls[["exit_velocity"]], 
  by = fouls[, c("type_of_hit", "predicted_zone")], 
  FUN = mean, 
  na.rm = TRUE
)


## -------------------------------------------------------------------------------
# install.packages("data.table")
require(data.table)
setDT(fouls)  # Convert the data.frame to a data.table
fouls[, .(avg = mean(exit_velocity, na.rm = TRUE),
          std =   sd(exit_velocity, na.rm = TRUE)),
      by = type_of_hit]


## ----lmmpg----------------------------------------------------------------------
m <- lm(formula = cty ~ displ, data = mpg)


## ----viewingcoeffslm------------------------------------------------------------
summary(m)


## ----plotlm,eval=FALSE----------------------------------------------------------
## plot(m)


## ----predictlm------------------------------------------------------------------
df <- data.frame(displ = seq(min(mpg$displ), max(mpg$displ), length.out = 20))
head(df)
df$predicted_cty <- predict(object = m, newdata = df)
head(df)
# plot(df, type = 'l')


## ----plotresidualslm,fig.height=4.5---------------------------------------------
plot(x = mpg$displ, y = m$residuals)


## ----eval=FALSE-----------------------------------------------------------------
## plot(x = jitter(...), y = jitter(...))


## ----quadratictermlm------------------------------------------------------------
m2 <- lm(cty ~ displ + I(displ^2), data = mpg)


## ----quadraticprediction--------------------------------------------------------
df$predict_cty <- predict(m2, df)


## ----plottingnonlinearpredictionslm,fig.height=4--------------------------------
plot(df$displ, df$predict_cty, type = "l")
points(x = jitter(mpg$displ), y = jitter(mpg$cty), col = "red")


## ----categoricalregressionlm----------------------------------------------------
m3 <- lm(cty ~ manufacturer, data = mpg)


## ----coeffsforcategoricalregression---------------------------------------------
coef(m3)


## ----setupbinaryclassificationdata,cache=TRUE-----------------------------------
download.file(
  url = file.path("https://archive.ics.uci.edu/ml",
                  "machine-learning-databases/00426",
                  "Autism-Adult-Data%20Plus%20Description%20File.zip"),
  destfile = "~/Downloads/autism.zip"
)
unzip("~/Downloads/autism.zip", files = "Autism-Adult-Data.arff", exdir = "~/Downloads")


## -------------------------------------------------------------------------------
require(foreign)
data <- read.arff("~/Downloads/Autism-Adult-Data.arff")
str(data)


## ----firstglm-------------------------------------------------------------------
data <- na.omit(data)
m <- glm(A10_Score ~ age + gender + ethnicity + jundice + austim + contry_of_res, 
         family = "binomial", data)


## ----predictglm-----------------------------------------------------------------
data$prediction <- predict(m, data, type = "response")


## ----calibrationglm-------------------------------------------------------------
aggregate(x = as.integer(as.character(data$A10_Score)), 
          by = list(prediction = round(data$prediction, 1)), 
          FUN = mean, na.rm = TRUE)


## ----quantilescut---------------------------------------------------------------
breaks <- quantile(data$prediction, probs = seq(0, 1, by = 0.1),
                   na.rm = TRUE)
data$bucket <- cut(data$prediction, breaks, include.lowest = TRUE)


## ----aggregatepredictionsusingquantiles-----------------------------------------
outcomes <- 
    aggregate(x = as.integer(as.character(data$A10_Score)), 
              by = list(prediction = data$bucket), 
              FUN = mean, na.rm = TRUE)
outcomes


## ----medianbucketpredictions,fig.height=4---------------------------------------
median_predictions <- 
    aggregate(x = data$prediction, 
              by = list(prediction = data$bucket), 
              FUN = median, na.rm = TRUE)
summ <- merge(outcomes, median_predictions, 
              by = "prediction", 
              suffixes = c("_outcome", "_median_prediction"))
summ

ggplot(summ, aes(x = x_median_prediction, y = x_outcome)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0) +
    labs(title = "Visualizing Calibration of a Classification Model",
         subtitle = "Data are grouped into Deciles",
         x = "Median Prediction",
         y = "Average Observed Label")


## ----creation_of_df_example,eval=FALSE------------------------------------------
## df <- data.frame(variable1 = 1:10,
##                  variable2 = rnorm(10))


## ----first_principles_stats_setup,echo=FALSE,eval=FALSE-------------------------
## xs <- replicate(n = 3, expr = rnorm(1e3))
## df <- data.frame(a = xs[, 1],
##                  b = xs[, 1] + xs[, 2],
##                  c = xs[, 1] + xs[, 2] + xs[, 3])


## ----soln_for_variance_of_first_principles_ex,echo=FALSE,eval=FALSE-------------
## sapply(df, var)


## ----plot_exercise,size="footnotesize",fig.width=6.5,fig.height=4.15,eval=FALSE----
## # Assuming we have (only!) columns "a", "b", "c" in our `df`.
## m <- reshape2::melt(df)  # <-- Automatically creates long data.frame.
## m <- m %>% group_by(variable) %>% mutate(idx = seq_along(value))
## ggplot(m, aes(x = value, fill = variable)) +
##     geom_histogram(bins = 50) +
##     facet_wrap(~variable, nrow = 3)


## ----random_df------------------------------------------------------------------

df <- data.frame(x = rnorm(6), y = rpois(6, lambda = 1), z = rexp(6))


## ----echo=1:2,eval=FALSE--------------------------------------------------------
## majors <- c("bio", "chem", "math", "english", "statistics")
## df <- data.frame(major = sample(majors, size = 100, repl = TRUE),
##                  gpa   = runif(100, 2.5, 4))
## df %>%
##     group_by(major) %>%
##     summarise(prop_gt_3p5 = mean(gpa > 3.5))


## -------------------------------------------------------------------------------
d <- data.frame(i = letters,
                x = runif(26),
                y = rnorm(26),
                z = rnorm(26))
d %>% slice(c(1:3, nrow(d)))

# Pull out the first, third, and last row by ID.
d %>% filter(i %in% c("a", "c", "z"))


## ----conditional_correlation,eval=FALSE,echo=FALSE------------------------------
## d %>%
##     filter(i %in% c("a", "c", "z")) %>%
##     summarise(correlation = cor(y, z))


## ----density_plot,eval=FALSE,echo=FALSE-----------------------------------------
## ggplot(mpg, aes(x = cty, colour = drv)) +
##     geom_density()


## ----density_plot_with_alpha,eval=FALSE,echo=FALSE------------------------------
## ggplot(mpg, aes(x = cty, fill = drv)) +
##     geom_density(alpha = 1/2)


## ----fig.height = 4, fig.width = 8.5,echo=FALSE,eval=FALSE----------------------
## ggplot(fouls, aes(x = type_of_hit, y = exit_velocity)) +
##     geom_boxplot() +
##     labs(title = "Exit Velocity ~ Type of Hit",
##          x = "Type of Hit", y = "Exit Velocity")
## plot(fouls\$type\_of\_hit, fouls\$exit\_velocity)


## ----dates----------------------------------------------------------------------
dates <- seq.Date(from = as.Date("2021-08-01"),
                  to   = as.Date("2021-12-31"),
                  by   = "1 day")


## -------------------------------------------------------------------------------
year <- "2012"
beg_date <- as.Date(paste(year, "01-01", sep = "-"))


## ----names----------------------------------------------------------------------
names <- c("Andreas Santucci", "Ada Lovelace")
df <- data.frame(name = names)


## ----soln_dates_strsplit_exercise,eval=FALSE,echo=FALSE-------------------------
## # Solution 1:
## strsplit(as.character(dates), split = "-", fixed=TRUE) %>%
##     do.call(rbind, .)
## 
## # Solution 2:
## data.frame("date" = dates) %>%
##     tidyr::separate(col = "date",
##                     into = c("year", "month", "day"),
##                     sep = "-")


## ----preview_starwars,eval=FALSE------------------------------------------------
## head(starwars)


## ----install_stringr,eval=FALSE-------------------------------------------------
## install.packages("stringr")


## ----creating_sample_calendar_data,warning=FALSE--------------------------------
  dir.create("calendar_months_data", showWarnings = FALSE)
  for (i in 1:12) {
      d <- data.frame(id = letters, y  = rnorm(26))
      write.csv(d, file = paste0("calendar_months_data/month_", i, ".csv"),
                row.names = FALSE)
  }
