# Instacart Shopping Cart Analysis

<img src = 'https://d2guulkeunn7d8.cloudfront.net/assets/beetstrap/brand/logo@3x-c01b12eeb889d8665611740b281d76fa1cf88d06bcbd8a50dbcae6baecdbe9da.png'>

### Problem Statement
The goal of this project is to conduct a market basket analysis on a sample of Instacart users to better understand better customer purchasing patterns. Market basket analysis can help to analyze a wide variety use cases including predicting the likelihood that a user will buy again, try a product for the first time, or add a particular product to their cart next during a session. For this project, I will be focusing primarily on predicting which previously purchased products will be in a user’s next Instacart order. 

The applications of this project are abundant. At the foreground, Using product association insight gathered from data helps to optimize on in-app recommendations and product placement on the Instacart site, which can increase sales while also making the shopping experience more productive for customers. In addition, having a better grasp of products that are being purchased and when can also help Instacart strengthen its relationships with retail partners by supplementing insight into inventory planning. 


#### Hypothesis/assumptions
The hypothesis is that previous buying behavior predicts future buying behavior. 
Product purchasing habits are mainly a product of habit, indicated by patterns of re-purchasing everyday groceries, or a set cadence of order-placing.   

#### Goals and success metrics
The goals are to: <br>
> 1. characterize repeat users of Instacart <br>
> 2. segment and characterize purchase behaviors based on time/day <br>
> 3. segment and understand the kinds of products that are purchased and re-ordered<br>
> 4. predict and measure repeated purchase by existing users, enabling Donorschoose.org to build more targeted recommendations
#### Limitations 
We are currently unable to characterize Instacart users as all personal user data are anonymized, and the data includes orders across many different retailers. The most we can uncover using the given dataset is segmented by time/date of purchase. The dataset provided represents a subset of Instacart's production data, which may be heavily biased. One example of this bias is that orders per customer are limited to 4-100 orders per customer. 

#### Data Source 
The data used for this study was supplied by Instacart as a coding and modeling challenge. I will be using anonymized data of purchase patterns of different users to test: given the prior order metadata on products that were reordered, predict the products that will be reordered in these new orders.

