### Web Scraping for Indeed.com & Predicting Salaries

The goal of the project was to investigate relationship between certain features present in online job adverts and the salary levels of data scientists. Analysis and modeling has been performed on data concerning data scientists roles in some large cities and ‘tech capitals’ in USA advertised on a job portal Indeed.com.

#### Methods

Th data has been collected by scraping job ads from Indeed.com with help of Requests and Beautiful Soup libraries. 

10222 job adverts in 23 US locations has been retrieved from Indeed.com. After initial data cleaning and assessment 306 observations containing annual salary has been deemed complete and fit for purpose of modeling. 

Using job location, job title, and description summary keywords (extracted using Count Vectorizer) as features and salary as a target a Logistic Regression model has been built that served as binary predictor for the salary ie. whether the salary predicted would be higher or lower than median salary for the locations considered.

#### Results

The pay range for data scientists is very wide, starting from $24,000 at low end and ending at $250,000 at the high end of the scale. Having run multiple models incorporating various combinations of influencing features, we can conclude that locations associated with salaries are San Jose, Philadephia, Seattle and San Francisco.
When it comes to job titles associated with high salaries, all those indicating higher position in professional hierarchy are indeed deemed important by our model, with ‘senior’, ‘lead’, ‘vp’ and ‘principal’ in the lead. One more characteristic closely related to the high salary is summary keywords. For our analysis we have chosen keywords that may indicate skills required from candidate for a job role. Out of these, ‘hadoop’, ‘spark’ and ‘big data’ seem to have strongest associations with high salaries. 
