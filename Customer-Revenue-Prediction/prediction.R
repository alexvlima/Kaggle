############
### PATH ###
############

getwd() 
setwd('~/Documents/Kaggle/Google-Analytics-Customer-Revenue-Prediction')

#################
### LIBRARIES ###
#################

library(caret)
library(ggalluvial)
library(xgboost)
library(jsonlite)
library(lubridate)
library(knitr)
library(Rmisc)
library(scales)
library(countrycode)
library(highcharter)
library(glmnet)
library(keras)
library(forecast)
library(zoo)
library(magrittr)
library(tidyverse)

###############
### DATASET ###
###############

set.seed(0)

tr <- read_csv("./Input/train.csv")
te <- read_csv("./Input/test.csv")
subm <- read_csv("./Input/sample_submission.csv")

#################
### JSON TYPE ###
#################

flatten_json <- . %>% 
  str_c(., collapse = ",") %>% 
  str_c("[", ., "]") %>% 
  fromJSON(flatten = T)

parse <- . %>% 
  bind_cols(flatten_json(.$device)) %>%
  bind_cols(flatten_json(.$geoNetwork)) %>% 
  bind_cols(flatten_json(.$trafficSource)) %>% 
  bind_cols(flatten_json(.$totals)) %>% 
  select(-device, -geoNetwork, -trafficSource, -totals)

tr <- parse(tr)
te <- parse(te)

########################
### DIFF IN DATASETS ###
########################

glimpse(tr)
glimpse(te)

setdiff(names(tr), names(te))
tr %<>% select(-one_of("campaignCode")) 
# The test set lacks two columns. 
# One column is a target variable transactionRevenue. 
# The second column (campaignCode) we remove from the train set.

########################
### CONSTANT COLUMNS ###
########################

fea_uniq_values <- sapply(tr, n_distinct)
(fea_del <- names(fea_uniq_values[fea_uniq_values == 1]))

tr %<>% select(-one_of(fea_del))
te %<>% select(-one_of(fea_del))

#####################
### MISSING VALUE ###
#####################

is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
                                  "(not set)", "<NA>", "unknown.unknown",  "(none)")

tr %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))
te %<>% mutate_all(funs(ifelse(is_na_val(.), NA, .)))

#######################
### TARGET VARIABLE ###
#######################

y <- as.numeric(tr$transactionRevenue)
tr$transactionRevenue <- NULL
summary(y)

y[is.na(y)] <- 0
summary(y)

##############
### MODELS ###
##############

grp_mean <- function(x, grp) ave(x, grp, FUN = function(x) mean(x, na.rm = TRUE))

id <- te[, "fullVisitorId"]
tri <- 1:nrow(tr)

tr_te <- tr %>% 
  bind_rows(te) %>% 
  mutate(date = ymd(date),
         year = year(date) %>% factor(),
         month = month(date) %>% factor(),
         week = week(date) %>% factor(),
         day = day(date) %>% factor(),
         hits = as.integer(hits),
         pageviews = as.integer(pageviews),
         bounces = as.integer(bounces),
         newVisits = as.integer(newVisits),
         isMobile = ifelse(isMobile, 1L, 0L),
         isTrueDirect = ifelse(isTrueDirect, 1L, 0L),
         adwordsClickInfo.isVideoAd = ifelse(!adwordsClickInfo.isVideoAd, 0L, 1L)) %>% 
  select(-date, -fullVisitorId, -visitId, -sessionId) %>% 
  mutate_if(is.character, factor) %>% 
  mutate(pageviews_mean_vn = grp_mean(pageviews, visitNumber),
         hits_mean_vn = grp_mean(hits, visitNumber),
         pageviews_mean_country = grp_mean(pageviews, country),
         hits_mean_country = grp_mean(hits, country),
         pageviews_mean_city = grp_mean(pageviews, city),
         hits_mean_city = grp_mean(hits, city)) %T>% 
  glimpse()

submit <- . %>% 
  as_tibble() %>% 
  set_names("y") %>% 
  mutate(y = ifelse(y < 0, 0, expm1(y))) %>% 
  bind_cols(id) %>% 
  group_by(fullVisitorId) %>% 
  summarise_(y = log1p(sum(y))) %>% 
  right_join(
    read_csv('./Input/sample_submission.csv'), 
    by = c("fullVisitorId"="fullVisitorId")) %>% 
  mutate(PredictedLogRevenue = round(y, 5)) %>% 
  select(-y) %>% 
  write_csv(sub)


##############
### GLMNET ###
##############

tr_te_ohe <- tr_te %>% 
  mutate_if(is.factor, fct_lump, prop = 0.025) %>% 
  mutate_if(is.numeric, funs(ifelse(is.na(.), 0L, .))) %>% 
  mutate_if(is.factor, fct_explicit_na) %>% 
  select(-adwordsClickInfo.isVideoAd) %>% 
  model.matrix(~.-1, .) %>% 
  scale() %>% 
  round(4)

X <- tr_te_ohe[tri, ] 
X_test <- tr_te_ohe[-tri, ]

rm(tr_te_ohe); invisible(gc())

m_glm <- cv.glmnet(X, log1p(y), alpha = 0, family="gaussian", 
                   type.measure = "mse", nfolds = 7)

pred_glm_tr <- predict(m_glm, X, s = "lambda.min") %>% c()
pred_glm <- predict(m_glm, X_test, s = "lambda.min") %>% c()
sub <- "glmnet_gs.csv"
submit(pred_glm)

#############
### KERAS ###
#############

# install_keras()
m_nn <- keras_model_sequential() 
m_nn %>% 
  layer_dense(units = 256, activation = "relu", input_shape = ncol(X)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 1, activation = "linear")

m_nn %>% compile(loss = "mean_squared_error",
                 metrics = custom_metric("rmse", function(y_true, y_pred) 
                   k_sqrt(metric_mean_squared_error(y_true, y_pred))),
                 optimizer = optimizer_adadelta())

history <- m_nn %>% 
  fit(X, log1p(y), 
      epochs = 50, 
      batch_size = 128, 
      verbose = 0, 
      validation_split = 0.2,
      callbacks = callback_early_stopping(patience = 5))

pred_nn_tr <- predict(m_nn, X) %>% c()
pred_nn <- predict(m_nn, X_test) %>% c()
sub <- "keras_gs.csv"
submit(pred_nn)

###########
### XGB ###
###########

tr_te_xgb <- tr_te %>% 
  mutate_if(is.factor, as.integer) %>% 
  glimpse()

dtest <- xgb.DMatrix(data = data.matrix(tr_te_xgb[-tri, ]))
tr_te_xgb <- tr_te_xgb[tri, ]
idx <- ymd(tr$date) < ymd("20170701")
dtr <- xgb.DMatrix(data = data.matrix(tr_te_xgb[idx, ]), label = log1p(y[idx]))
dval <- xgb.DMatrix(data = data.matrix(tr_te_xgb[!idx, ]), label = log1p(y[!idx]))
dtrain <- xgb.DMatrix(data = data.matrix(tr_te_xgb), label = log1p(y))
cols <- colnames(tr_te_xgb)
rm(tr_te_xgb); invisible(gc)

p <- list(objective = "reg:linear",
          booster = "gbtree",
          eval_metric = "rmse",
          nthread = 4,
          eta = 0.025,
          max_depth = 8,
          min_child_weight = 5,
          gamma = 0.05,
          subsample = 0.9,
          colsample_bytree = 0.8,
          nrounds = 2000)

set.seed(0)
m_xgb <- xgb.train(p, dtr, p$nrounds, list(val = dval), 
                   print_every_n = 100, early_stopping_rounds = 100)

xgb.importance(cols, model = m_xgb) %>% 
  xgb.plot.importance(top_n = 20)

pred_xgb_tr <- predict(m_xgb, dtrain)
pred_xgb <- predict(m_xgb, dtest) 
sub <- "xgb_gs.csv"
submit(pred_xgb)

