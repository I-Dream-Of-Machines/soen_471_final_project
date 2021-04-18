# BigDataAnalysis

### Abstract 

Proficiency in academic skills such as reading and numeracy, as well as educational attainment are correlated with individual economic outcomes and mortality. Given this correlation, it is imperative to identify potential causes for disparities in academic skills and educational attainment. Identified potential causes serve as a starting point for interventions aimed at improving economic outlook and extending life expectancy. 

This study aims to identify potential causes for disparities in educational attainment and achievement within Massachusetts Public Schools using data extracted from 2017 Massachusetts Department education reports. This data characterizes Massachusetts public elementary schools, middle schools, and high schools and provides academic proficiency data for all schools as well as educational attainment data for high schools.  This study will perform regression analysis employing random forest regression and support vector regression. 

### Introduction

In 2020, the US Bureau of Labor Statistics reported that those with less than a high school diploma had an unemployment rate of 5.4% and median weekly earnings of $592. In contrast, those with a Bachelor's degree had an unemployment rate of 2.2% and median weekly earnings of $1248. Longitudinal studies, such as that conducted by Watts, have found measures of reading and mathematical skills to be correlated with career earnings. A 2011 report published by the US National Institute of Health found a 10-year disparity in life expectancy between those without a high school degree and those with a college degree. 

Despite these correlations, levels of academic performance and educational attainment vary widely. In 2019, the U.S. Census Bureau reported that 13% of the population had no high school diploma, 48.5% had a high school diploma as their highest level of educational attainment, while 38.5% had a higher degree. Research has identified a range of individual factors contributing to these disparities including race, gender, and economic background.

The objective of this study is to identify potential contextual factors that contribute to disparities in academic performance and education attainment within Massachusetts schools. Contextual factors include characteristics of individual schools as well as characteristics of the location of the school. A related study carried out by Everson and Millsap (2004) looking at individual and school characteristics and SAT performance found that school characteristics impact SAT performance. 

### Materials 
	
Our primary data set is 1.54 MB in size and covers a total of 1800 middle schools and high schools. Our primary data set was found on Kaggle (https://www.kaggle.com/ndalziel/massachusetts-public-schools-data).  We will supplement this data with American survey community data provided by the United States Census Bureau for places with populations over 5000. 

In the primary data set, each school is characterized by location, the gender and racial composition of the student body, the percentage of students with disabilities and high needs as well as the percentage of students in economic distress. Schools are also characterized by the compensation offered to employees and the average expenditure per pupil. American survey community data provides data on school locations detailing their social, economic, housing, and demographic characteristics. 

For each high school in the primary dataset, indicators for educational attainment include graduation rates and the percentage of students pursuing various postsecondary accreditations. Reported measures of academic skills include average MCAS (Massachusetts Comprehensive Assessment System) scores for each grade, average SAT scores and the number of students taking AP classes along with the number of students attaining each grade. 

### Methods

We will perform regression analysis for each educational attainment and academic skill indicator using random forest regression and support vector regression, two supervised machine learning techniques. Regression analysis is a statistical method that allows us to estimate the correlation between dependent and independent variables. In our project, variables characterizing schools and location will serve as our independent variables while educational attainment and educational skill performance indicators will serve successively as sucessively as the dependent variable.

In order to perform regression analysis, we will use support vector regression (SVR) and random forest regression. SVR (Support Vector Regression) was selected as it is a supervised learning technique that works well with a limited amount of data, and we have a small data set. In this technique. multiple decision trees are constructed and the output is the mean regression of all decision trees.  We selected random forest, as it is interpretable and less impacted by colinearity than other techniques.

We will use Python, which is one of the most popular programming languages with a great number of libraries and frameworks for machine learning. In this project, we will use PySpark and scikit-learn PySpark is a data processing framework that enables distributed data analysis, allowing for rapid data analysis. Scikit-learn is a free software machine learning library for Python which provides an API for regression algorithms.

