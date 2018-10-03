############
### PATH ###
############

getwd() 
setwd('~/Documents/Kaggle/Google-Analytics-Customer-Revenue-Prediction')

###############
### LIBRARY ###
###############

library(tidyverse)
library(jsonlite)
library(data.table)
library(lubridate)
library(gridExtra)
library(countrycode)
library(highcharter)
library(ggExtra)
library(glmnet)
library(keras)
library(forecast)
library(knitr)
library(Rmisc)
library(caret)
library(ggalluvial)
library(xgboost)
library(zoo)

###############
### DATASET ###
###############

dtrain <- read_csv('./Input/train.csv')
nrow(dtrain)
glimpse(dtrain)

ID_unique <- unique(dtrain$fullVisitorId)
length(ID_unique)

dtest <- read_csv('./Input/test.csv')
# glimpse(test)

# Submission for Kaggle
submission <- read_csv("sample_submission.csv")

#####################
### PREPROCESSING ###
#####################

# View(dtrain$totals)
sample(dtrain$totals,100)
class(dtrain$totals)

# convert date column from character to Date class
dtrain$date <- as.Date(as.character(dtrain$date), format='%Y%m%d')

# convert visitStartTime to POSIXct
dtrain$visitStartTime <- as_datetime(dtrain$visitStartTime)

# treating json columns
tr_device <- paste("[", paste(dtrain$device, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_geoNetwork <- paste("[", paste(dtrain$geoNetwork, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_totals <- paste("[", paste(dtrain$totals, collapse = ","), "]") %>% fromJSON(flatten = T)
tr_trafficSource <- paste("[", paste(dtrain$trafficSource, collapse = ","), "]") %>% fromJSON(flatten = T)

dtrain <- cbind(dtrain, tr_device, tr_geoNetwork, tr_totals, tr_trafficSource) %>%
  as.data.table()

# drop the old json columns
dtrain[, c('device', 'geoNetwork', 'totals', 'trafficSource') := NULL]

# values to convert to NA
na_vals <- c('unknown.unknown', '(not set)', 'not available in demo dataset', 
             '(not provided)', '(none)', '<NA>')

for(col in names(dtrain)) {
  
  set(dtrain, i=which(dtrain[[col]] %in% na_vals), j=col, value=NA)
  
}

# get number of unique values in each column
unique <- sapply(dtrain, function(x) { length(unique(x[!is.na(x)])) })

# subset to == 1
one_val <- names(unique[unique <= 1])

# but keep bounces and newVisits
one_val = setdiff(one_val, c('bounces', 'newVisits'))

# drop columns from dtrain
dtrain[, (one_val) := NULL]

glimpse(dtrain)

# character columns to convert to numeric
num_cols <- c('hits', 'pageviews', 'bounces', 'newVisits',
              'transactionRevenue')

# change columns to numeric
dtrain[, (num_cols) := lapply(.SD, as.numeric), .SDcols=num_cols]

# Divide transactionRevenue by 1,000,000
dtrain[, transactionRevenue := transactionRevenue / 1e+06]

######################
### MISSING VALUES ###
######################

data.table(
  pmiss = sapply(dtrain, function(x) { (sum(is.na(x)) / length(x)) }),
  column = names(dtrain)
) %>%
  ggplot(aes(x = reorder(column, -pmiss), y = pmiss)) +
  geom_bar(stat = 'identity', fill = 'steelblue') + 
  scale_y_continuous(labels = scales::percent) + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(
    title='Missing data by feature',
    x='Feature',
    y='% missing')

###################
### EXPLORATION ###
###################

time_range <- range(dtrain$date)
print(time_range)

###############################
## target variable (revenue) ##
###############################

rev_range <- round(range(dtrain$transactionRevenue, na.rm=TRUE), 2)
print(rev_range)

# distribution of revenue from individual visits
dtrain %>% 
  ggplot(aes(x=log(transactionRevenue), y=..density..)) + 
  geom_histogram(fill='steelblue', na.rm=TRUE, bins=40) + 
  geom_density(aes(x=log(transactionRevenue)), fill='orange', color='orange', alpha=0.3, na.rm=TRUE) + 
  labs(
    title = 'Distribution of transaction revenue',
    x = 'Natural log of transaction revenue'
  )

# daily revenue over the time period
g1 <- dtrain[, .(n = .N), by=date] %>%
  ggplot(aes(x=date, y=n)) + 
  geom_line(color='steelblue') +
  geom_smooth(color='orange') + 
  labs(
    x='',
    y='Visits (000s)',
    title='Daily visits'
  )

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)), by=date] %>%
  ggplot(aes(x=date, y=revenue)) + 
  geom_line(color='steelblue') +
  geom_smooth(color='orange') + 
  labs(
    x='',
    y='Revenue (unit dollars)',
    title='Daily transaction revenue'
  )

grid.arrange(g1, g2, nrow=2)

# revenue by hour of day
g1 <-
  dtrain[, .(visitHour = hour(visitStartTime))][
    , .(visits = .N), by = visitHour] %>%
  ggplot(aes(x = visitHour, y = visits / 1000)) +
  geom_line(color = 'steelblue', size = 1) +
  geom_point(color = 'steelblue', size = 2) +
  labs(
    x = 'Hour of day',
    y = 'Visits (000s)',
    title = 'Aggregate visits by hour of day (UTC)',
    subtitle = 'August 1, 2016 to August 1, 2017'
    
  )

g2 <-
  dtrain[, .(transactionRevenue, visitHour = hour(visitStartTime))][
    , .(revenue = sum(transactionRevenue, na.rm =
                        T)), by = visitHour] %>%
  ggplot(aes(x = visitHour, y = revenue / 1000)) +
  geom_line(color = 'steelblue', size = 1) +
  geom_point(color = 'steelblue', size = 2) +
  labs(
    x = 'Hour of day',
    y = 'Transaction revenue (000s)',
    title = 'Aggregate revenue by hour of day (UTC)',
    subtitle = 'August 1, 2016 to August 1, 2017'
    
  )

grid.arrange(g1, g2, nrow = 2)

# transaction revenue grouped by channel
g1 <- dtrain[, .(n = .N), by=channelGrouping] %>%
  ggplot(aes(x=reorder(channelGrouping, -n), y=n/1000)) +
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Channel Grouping',
       y='Visits (000s)',
       title='Visits by channel grouping')

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)), by=channelGrouping] %>%
  ggplot(aes(x=reorder(channelGrouping, revenue), y=revenue/1000)) +
  geom_bar(stat='identity', fill='steelblue') +
  coord_flip() + 
  labs(x='Channel Grouping',
       y='Revenue (dollars, 000s)',
       title='Total revenue by channel grouping')


g3 <- dtrain[, .(meanRevenue = mean(transactionRevenue, na.rm=TRUE)), by=channelGrouping] %>%
  ggplot(aes(x=reorder(channelGrouping, meanRevenue), y=meanRevenue)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  coord_flip() + 
  labs(x='', 
       y='Revenue (dollars)',
       title='Mean revenue by channel grouping')

g1
grid.arrange(g2, g3, ncol = 2)

#####################
## device features ##
#####################

# visits/revenue by device 
g1 <- dtrain[, .(n=.N/1000), by=operatingSystem][
  n > 0.001
  ] %>%
  ggplot(aes(x=reorder(operatingSystem, -n), y=n)) + 
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Operating System', 
       y='# of visits in data set (000s)',
       title='Distribution of visits by device operating system') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)), by=operatingSystem][
  revenue > 0, 
  ] %>%
  ggplot(aes(x=reorder(operatingSystem, -revenue), y=revenue)) +
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Operating System',
       y='Revenue (unit dollars)',
       title='Distribution of revenue by device operating system') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

grid.arrange(g1, g2, nrow=2)

# visits/revenue by browser

g1 <- dtrain[, .(n=.N/1000), by=browser][
  1:10
  ] %>%
  ggplot(aes(x=reorder(browser, -n), y=n)) + 
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Browser', 
       y='# of visits in data set (000s)',
       title='Distribution of visits by browser (Top 10 browsers)') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)/1000), by=browser][
  1:10
  ] %>%
  ggplot(aes(x=reorder(browser, -revenue), y=revenue)) +
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Browser',
       y='Revenue (dollars, 000s)',
       title='Distribution of revenue by browser (top 10 browsers)') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

grid.arrange(g1, g2, nrow=2)

# visits/revenue by device category

g1 <- dtrain[, .(n=.N/1000), by=deviceCategory]%>%
  ggplot(aes(x=reorder(deviceCategory, -n), y=n)) + 
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Device Category', 
       y='# of records in data set (000s)',
       title='Distribution of records by device category') + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)/1000), by=deviceCategory] %>%
  ggplot(aes(x=reorder(deviceCategory, -revenue), y=revenue)) +
  geom_bar(stat='identity', fill='steelblue') +
  labs(x='Device category',
       y='Revenue (dollars, 000s)',
       title='Distribution of revenue by device category') +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

grid.arrange(g1, g2, ncol=2)

# difference in transaction revenue between mobile and non-mobile devices
dtrain %>%
  ggplot(aes(x=log(transactionRevenue), y=..density.., fill=isMobile)) +
  geom_density(alpha=0.5) + 
  scale_fill_manual(values = c('steelblue', 'orange')) + 
  labs(title='Distribution of log revenue by mobile and non-mobile devices')

#########################
## geographic features ##
#########################

# revenue by continent
dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)/1000), by = continent][
  !is.na(continent),
  ] %>%
  ggplot(aes(x=reorder(continent, revenue), y=revenue)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  coord_flip() + 
  labs(x='', y='Revenue (dollars, 000s)', title='Total transaction revenue by continent')

# group by country and calculate total transaction revenue (log)
by_country <- dtrain[, .(n = .N, revenue = log(sum(transactionRevenue, na.rm=TRUE))), by = country]
by_country$iso3 <- countrycode(by_country$country, origin='country.name', destination='iso3c')
by_country[, rev_per_visit := revenue / n]

# create the highcharter map of revenue by country
highchart() %>%
  hc_add_series_map(worldgeojson, by_country, value = 'revenue', joinBy = 'iso3') %>%
  hc_title(text = 'Total transaction revenue by country (natural log)') %>%
  hc_subtitle(text = "August 2016 to August 2017") %>%
  hc_tooltip(useHTML = TRUE, headerFormat = "",
             pointFormat = "{point.country}: ${point.revenue:.0f}")

# function to map transaction revenue by continent
map_by_continent <- function(continent, map_path) {
  
  mdata <- dtrain[
    continent == continent, .(n = .N, revenue = log(sum(transactionRevenue, na.rm=TRUE))), by=country]
  
  mdata$iso3 <- countrycode(mdata$country, origin='country.name', destination='iso3c')
  
  hcmap(map=map_path, data=mdata, value='revenue', joinBy=c('iso-a3', 'iso3')) %>%
    hc_title(text = 'Total transaction revenue by country (natural log of unit dollars)') %>%
    hc_subtitle(text = "August 2016 to August 2017") %>%
    hc_tooltip(useHTML = TRUE, headerFormat = "",
               pointFormat = "{point.country}: {point.revenue:.0f}")
}

# call function for Europe
map_by_continent(continent='Europe', map_path='custom/europe')

# call function for Africa
map_by_continent('Africa', 'custom/africa')

# call function for Asia
map_by_continent('Asia', 'custom/asia')

# call function for South America
map_by_continent('Americas', 'custom/south-america')

# call function for North America
map_by_continent('Americas', 'custom/north-america')

# call function for Antartica
map_by_continent('Antarctica', 'custom/antarctica')

##########################################
## visits and revenue by network domain ##
##########################################

# split networkDomain column on '.', add to dtrain
dtrain[, domain := tstrsplit(dtrain$networkDomain, '\\.', keep=c(2))][
  # add the '.' back in
  !is.na(domain), domain := paste0('.', domain)
  ]

g1 <- dtrain[!is.na(networkDomain), .(n = .N), by = domain][order(-n)][!is.na(domain), ][1:20] %>%
  ggplot(aes(x=reorder(domain, -n), y=n/1000)) +
  geom_bar(stat='identity', fill='steelblue') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title='Number of visits from top-level domains',
       y='Visits (000s)',
       x='Top-level domain',
       subtitle='Unknown domains excluded')

g2 <- dtrain[!is.na(networkDomain), .(revenue = sum(transactionRevenue, na.rm=TRUE)), by = domain][
  order(-revenue)][
    !is.na(domain), ][1:20] %>%
  ggplot(aes(x=reorder(domain, -revenue), y=revenue/1000)) +
  geom_bar(stat='identity', fill='steelblue') + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(
    title='Revenue from top-level domains', 
    y='Revenue (000s)', 
    x='Top-level domain',
    subtitle='Unknown domains excluded')

grid.arrange(g1, g2)

#####################
## totals features ##
#####################

# features probably correlated with revenue
g1 <- ggplot(dtrain, aes(x=log(pageviews), y=log(transactionRevenue))) + 
  geom_point(color='steelblue') +
  geom_smooth(method='lm', color='orange') + 
  labs(
    y='Transaction revenue (log)',
    title='Pageviews vs transaction revenue',
    subtitle='visit-level')


g2 <- ggplot(dtrain, aes(x=log(hits), y=log(transactionRevenue))) + 
  geom_point(color='steelblue') +
  geom_smooth(method='lm', color='orange') + 
  labs(
    y='Transaction revenue (log)',
    title='Hits vs transaction revenue',
    subtitle='visit-level')

m1 <- ggMarginal(g1, type='histogram', fill='steelblue')
m2 <- ggMarginal(g2, type='histogram', fill='steelblue')

grid.arrange(m1, m2, nrow = 1, ncol = 2)

#############################
## Traffic Source Features ##
#############################

g1 <- dtrain[, .(visits = .N), by = medium][
  !is.na(medium)] %>%
  ggplot(aes(x=reorder(medium, visits), y=visits / 1000)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  coord_flip() + 
  labs(
    x='Medium',
    y='Visits (000s)',
    title='Distribution of visits by medium')

g2 <- dtrain[, .(revenue = sum(transactionRevenue, na.rm=TRUE)), by = medium][
  !is.na(medium)] %>%
  ggplot(aes(x=reorder(medium, revenue), y=revenue / 1000)) + 
  geom_bar(stat='identity', fill='steelblue') + 
  coord_flip() + 
  labs(
    x='',
    y='Transaction revenue (dollars, 000s)',
    title='Distribution of revenue by medium')

grid.arrange(g1, g2, ncol=2)
