---
layout: post
title: Building an AI SQL Agent on a Live Cloud-Based Database
image: "/posts/gen-ai-sql-agent-title-img.png"
tags: [GenAI, SQL, Agents, Python, LangChain, PostgreSQL]
---

In this project I build a **SQL AI Agent** for an imaginary grocery retail client, capable of taking natural language questions and turning them into accurate PostgreSQL queries against the database.

The agent is able to interpret the user's intent, write the appropriate SQL query to answer the question, execute the query on the database, and return a clear natural language answer.

I achieve this by combining:

* A secure database connection
* A carefully scoped database wrapper
* A purpose-built SQL system prompt
* A set of SQL-aware tools
* A modern LLM configured as an agent

# Table of Contents

- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth and Next Steps](#overview-growth)
- [01. Data Overview](#data-overview)
- [02. SQL Agent Overview](#agent-overview)
- [03. Setting Up the Database Connection](#db-setup)
    - [Environment Variables](#db-env)
    - [Postgres Connection String](#db-uri)
    - [Engine and Health Check](#db-engine)
    - [SQLDatabase Wrapper](#db-sqldatabase)
- [04. Building the SQL Agent](#agent-build)
    - [LLM Setup](#agent-llm)
    - [SQL Toolkit and Tools](#agent-toolkit)
    - [System Prompt](#agent-prompt)
    - [Creating the Agent](#agent-create)
- [05. Application and Examples](#agent-application)
- [06. Growth and Next Steps](#growth-next-steps)

___

# Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Suppose a client, ABC Grocery, holds rich customer and transaction data in a PostgreSQL database. Data Science and Analytics teams often ask questions such as:

* Which customers live furthest from the store on average?
* What is the average transaction value over a given period?
* How do spending patterns differ by gender or credit score?

Answering these kinds of questions typically requires writing SQL queries, which can be too advanced for non-technical stakeholders, creating a bottleneck as they wait for technical resources to find or make time to write queries and get them the answer.

The goal of this project is to build an **AI SQL Agent** that can take a question in everyday English and do the following:

1. Understand what is being asked.
2. Design a query using good SQL practices.
3. Execute the query safely against the database.
4. Return a concise natural language summary.

### Actions <a name="overview-actions"></a>

I built an end-to-end SQL Agent that:

* Securely connects to the PostgreSQL database
* Exposes only the required schema and tables to the agent
* Uses a modern LLM (gpt-4.1) configured with a SQL system prompt
* Uses LangChain’s SQL tooling to inspect schemas and run queries
* Returns both the SQL results and a readable explanation

I traced and inspected runs in LangSmith, validating that queries were correct, efficient, and aligned with some design rules.

### Results <a name="overview-results"></a>

The final SQL Agent:

* Correctly answered natural language questions that involved both tables
* Automatically handled grouping, aggregation, and joins
* Followed constraints such as the allowed date range and type of aggregation
* Demonstrated good practices (for example, handling transaction-level aggregation correctly)

The end result is essentially a **self-serve analytics layer** on top of the SQL database, powered by an LLM and kept safe and controlled through careful prompting and tooling.

### Growth and Next Steps <a name="overview-growth"></a>

Potential future enhancements could include:

* Exposing additional tables for querying (for example, campaign data)
* Adding a lightweight UI so that non-technical users can ask questions without having to use SQL
* Logging queries and responses for audit and training
* Adding automated evaluations to check query correctness
* Allowing the agent to ask clarification questions in more complex scenarios

___

# Data Overview <a name="data-overview"></a>

For this project, the SQL Agent interacts with two core tables in the *grocery_db* schema:

1. **grocery_db.customer_details** – one row per customer
2. **grocery_db.transactions** – one row per combination of customer, transaction, and product area (category)

<br>

#### grocery_db.customer_details

This table stores customer attributes:

* *customer_id* – Unique customer identifier
* *distance_from_store* – Distance in miles from the store, rounded to two decimal places
* *gender* – M, F, or NULL
* *credit_score* – Decimal value between 0.00 and 1.00

Sample rows:

| **customer_id** | **distance_from_store** | **gender** | **credit_score** |
|---|---|---|---|
| 630 | 0.70 | F | 0.57 |
| 809 | 0.09 | M | 0.44 |
| 489 | 0.97 | F | 0.52 |
| 504 | 2.72 | F | 0.57 |
| 806 | 3.39 | F | 0.84 |

<br>

#### grocery_db.transactions

This table is at the *customer_id, transaction_id, product_area_id* level. This means:

* A single transaction can have multiple rows (one per product area)  
* Aggregations must be done carefully to avoid double counting  

Columns:

* *customer_id* – The unique customer identifier
* *transaction_date* – Date of the transaction
* *transaction_id* – Unique identifier per transaction
* *product_area_id* – Department (product area) the purchase belongs to (1–5)
* *num_items* – Number of items purchased in this product area
* *sales_cost* – Value of the purchased items items

Sample rows:

| **customer_id** | **transaction_date** | **transaction_id** | **product_area_id** | **num_items** | **sales_cost** |
|---|---|---|---|---|---|
| 306 | 2020-07-12 | 436589611570 | 2 | 3 | 9.10 |
| 5 | 2020-05-03 | 435884241159 | 2 | 17 | 44.74 |
| 209 | 2020-04-23 | 435786207807 | 2 | 39 | 72.89 |
| 556 | 2020-06-16 | 436326836359 | 4 | 10 | 38.35 |
| 782 | 2020-06-09 | 436259895006 | 3 | 1 | 4.06 |

The tables can be joined together via the *customer_id*. The **transactions** table covers the period from 2020-04-01 to 2020-09-30.

___

# 02. SQL Agent Overview <a name="agent-overview"></a>

Better than asking an LLM to *write some SQL*, I build a full **SQL Agent**. A simple SQL writer outputs a SQL query; it has no direct access to the database and cannot inspect schemas or data. A SQL Agent can read the schema, inspect sample rows, choose tools to run queries, iterate based on tool results, and return a final answer.

In this project, the agent will:

1. Receive a natural language question
2. Use the tools provided by LangChain’s SQL toolkit to understand the schema and data
3. Generate a query that follows the design rules given to it
4. Execute the query against the database
5. Summarize the results clearly for the user

This makes the agent both powerful and constrained.

___

# Setting Up the Database Connection <a name="db-setup"></a>

### Environment Variables <a name="db-env"></a>

The following code loads the database and API credentials from a *.env* file:

```python
import os
from dotenv import load_dotenv
load_dotenv()
```

<br>

### Postgres Connection String <a name="db-uri"></a>

Next I construct a PostgreSQL connection string using environment variables from the .env file.

```python
POSTGRES_URI = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}?sslmode=require")
```

<br>

### Engine and Health Check <a name="db-engine"></a>

I create a SQLAlchemy engine and perform a quick health check.

```python
# Create the database engine
import sqlalchemy as sa

# Create the database engine
engine = sa.create_engine(POSTGRES_URI,
                          pool_pre_ping = True,
                          connect_args = {"options": "-c statement_timeout=15000"})

# Check the connection
with engine.connect() as conn:
    conn.exec_driver_sql("select 1")
```

Key parameters:

* **`pool_pre_ping = True`**: Validates connections before use  
* *statement_timeout=15000*: Prevents long-running queries from getting hung up by setting 15 second timeout
* *select 1*: A check to confirm that credentials and network access are correct  

<br>

### SQLDatabase Wrapper <a name="db-sqldatabase"></a>

Next, the following code wraps the engine in LangChain’s **`SQLDatabase`** utility and sets the scope of the agent to only the tables it should have access to:

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase(engine = engine,
                 schema = "grocery_db",
                 include_tables = ["customer_details", "transactions"],
                 sample_rows_in_table_info = 5)

print("Usable tables:", db.get_usable_table_names())
```

Key parameters:

* **`schema`**: Explicitly set to *grocery_db*
* **`include_tables`**: Restricts access to only the *customer_details* and *transactions* tables
* **`sample_rows_in_table_info = 5`**: Provides a sampling of real data (the first 5 rows) to the agent

This gives the agent the context to understand and reason about column types and values, while keeping the scope tight and safe.

___

# Building the SQL Agent <a name="agent-build"></a>

### LLM Setup <a name="agent-llm"></a>

The following code configures a dedicated LLM for the SQL Agent:

```python
from langchain_openai import ChatOpenAI

sql_agent = ChatOpenAI(model = "gpt-4.1",
                       temperature = 0)
```

Using the *gpt-4.1* model with a **`temperature`** of 0 prioritizes determinism, consistency, and reduced creativity (which is best for SQL querying).

<br>

### SQL Toolkit and Tools <a name="agent-toolkit"></a>

The following creates a toolkit that gives the agent SQL-specific abilities:

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db = db, llm = sql_agent)
tools = toolkit.get_tools()
```

These tools allow the agent to:

* Inspect which tables exist
* Look at table info
* Sample rows
* Construct and execute SQL queries
* Refine or correct queries based on feedback

Rather than guessing the SQL to use based on the user question, the agent can actively work with the database.

<br>

### System Prompt <a name="agent-prompt"></a>

The system prompt is an important part of this SQL agent. It defines the agent’s role, scope, and guardrails. The code below reads the system prompt from a text file.

```python
# Bring in the system instructions
with open("sql-agent-system-prompt.txt", "r", encoding = "utf-8") as f:
    system_text = f.read()
```

The content of *sql-agent-system-prompt.txt* is shown below:

```text
ROLE:

You are an expert SQL data analyst and helpful assistant. Convert the user’s question into an accurate, efficient PostgreSQL 16 SELECT query and a concise natural-language answer.


SCOPE:

- Use read-only SELECT only (never INSERT/UPDATE/DELETE/DDL/admin).

- All tables live in schema grocery_db. Always schema-qualify (e.g grocery_db.customer_details)


TABLE INFORMATION:

1. grocery_db.customer_details

grocery_db.customer_details holds data at the level of one row per unique customer_id

Columns:

- customer_id INT (the unique identifier for the customer)
- distance_from_store NUMERIC(10,2) (the distance the customer lives from the store, in miles)
- gender VARCHAR(2) (values include 'M', 'F', and NULL)
- credit_score NUMERIC(3,2) (a decimal value between 0.00 and 1.00 pertaining to the customer's credit score)

2. grocery_db.transactions

grocery_db.transactions holds data at the level of one row per customer_id, transaction_id, product_area_id

Columns:

- customer_id INT (the unique identifier for the customer)
- transaction_date DATE (the date of the transaction, for example 2020-04-10)
- transaction_id INT (a unique id for each individual transaction)
- product_area_id INT (a number from 1 to 5 that represents the product area that was shopped in. There can be multiple product areas within a transaction)
- num_items INT (the number items within that product area, for that transaction)
- sales_cost NUMERIC(10,2) (the monetary value for the items purchased within that product area, for that transaction)


TABLE JOIN RELATIONSHIPS:

grocery_db.customer_details can be joined to grocery_db.transactions using the shared customer_id column


DATA WINDOW:

Transactions data is available between 2020-04-01 to 2020-09-30. If the user asks about "this period", assume that range. If they ask about a different period, filter explicitly. If they ask for data from outside this window, reply to them with the information pertaining to the period that is available.


QUERY DESIGN RULES:

SELECT-only. If returning raw rows, add LIMIT 100.

Aggregation discipline.

"Number of customers" = COUNT(DISTINCT customer_id) at the appropriate filter.

"Number of transactions/visits" = COUNT(DISTINCT transaction_id).

Revenue and Items = SUM(sales_cost) and SUM(num_items) at the right grouping.

Avoid double-counting: the transactions table is at the level of multiple product_area_id's per transaction_id.

Categoricals: Use actual domain values shown in samples (for example, gender IN ('M','F')). If uncertain, first check SELECT DISTINCT ... LIMIT 10.

Performance & clarity. Select only needed columns; use CTEs for readability; alias columns clearly; round monetary outputs for readability (for example, ROUND(SUM(sales_cost), 2)).

Ambiguity. If a question is ambiguous (for example, "top products" but no timeframe), ask a brief clarifying question before querying.



EXAMPLE QUERIES & RESPONSE FOR GUIDANCE:

Question A: 

"How many customers are male?"

SQL Query A:

select
  count(*) as num_custs
  
from 
  grocery_db.customer_details
  
where
  gender = 'M';

Response A:

"There are 380 male customers"


Question B: 

"What is the average credit score, by gender?"

SQL Query B:

select
  gender,
  avg(credit_score) as average_credit_score
  
from
  grocery_db.customer_details
  
group by
  gender;

Response B:

"The average credit score by gender is as follows:

- Female (F): 0.601
- Male (M): 0.593
- Unspecified (U): 0.563"


Question C: 

"Which customer had the highest average transaction value in July 2020, and what was that value?"

SQL Query B:

with transaction_values as (
select
  customer_id,
  transaction_id,
  sum(sales_cost) as transaction_value
  
from
  grocery_db.transactions
  
where
  transaction_date >= '2020-07-01'
  and transaction_date <=  '2020-07-31'
  
group by
  customer_id,
  transaction_id

),

avg_per_customer as (
select
  customer_id,
  avg(transaction_value) as avg_transaction_value
  
from
  transaction_values
  
group by
  customer_id
  
)

select
  customer_id,
  round(avg_transaction_value, 2) as avg_transaction_value
  
from
  avg_per_customer
  
order by
  avg_transaction_value desc,
  customer_id
  
limit 1;

Response C:

"Customer 514 had the highest average transaction value in July 2020, at $1027.77"
```

This system prompt gives the agent:

* A clear role
* Strict scope and safety rules
* Table descriptions and relationships
* Time constraints (a window of dates)
* Query design rules
* A small number of examples

This vastly increases the chance of correct, production-quality queries compared to an LLM attempting to write a SQL query without being able to actually access and examine the databse.

<br>

### Creating the Agent <a name="agent-create"></a>

It is time to create the agent itself.

```python
from langchain.agents import create_agent

agent = create_agent(model = sql_agent,
                     tools = tools,
                     system_prompt = system_text)
```

The agent now has:

* The LLM (*sql_agent*)  
* The SQL tools (*tools*)  
* The system prompt (*system_text*)  

Given a user question, this agent can plan, call tools, and reason toward a final answer.

___

# Application and Examples <a name="agent-application"></a>

I send a query to the agent, using LangChain’s **`HumanMessage`** to format the query:

```python
from langchain_core.messages import HumanMessage

user_query = "On average, which gender lives furthest from store?"
# user_query = "What is the average transaction value in September 2020 for male customers who have a credit score above 0.5"

result = agent.invoke({"messages": [HumanMessage(content = user_query)]})
print(result["messages"][-1].content)
```

Two examples of the SQL agent's responses to different queries are seen below:

1. **Question:** *On average, which gender lives furthest from the store?*  

The agent: 
   
* Recognized that this is a question about customer-level data
* Used the *customer_details* table
* Computed the average distance by gender
* Wrote both the SQL and a clear explanation

The SQL used by the agent, as seen when doing a trace in LangSmith, is as follows:

```SQL
select
  gender,
  round(avg(distance_from_store), 2) as avg_distance
from
  grocery_db.customer_details
group by
  gender
order by
  avg_distance desc;
```

The SQL agent's explanation is the following:

```
On average, female customers (F) live furthest from the store at 3.98 miles, followed by customers with unspecified gender at 3.36 miles, and male customers (M) at 0.87 miles.
```

This SQL code was verified manually using the SQL database to confirm that the answer was correct.

2. **Question:** *What is the average transaction value in September 2020 for male customers who have a credit score above 0.5?*  

This required both the *customer_details* and the *transactions* tables and more careful logic. The agent:  

* Filtered customers by gender and credit score
* Joined the *customer_details* and *transactions* tables
* Correctly aggregated sales at the transaction level, summing by *transaction_id* first and filtering to the requested dates
* Averaged these transaction values per customer

```SQL
with male_customers as (
  select customer_id
  from grocery_db.customer_details
  where gender = 'M' and credit_score > 0.5
),
transaction_totals as (
  select
    t.transaction_id,
    sum(t.sales_cost) as transaction_value
  from
    grocery_db.transactions t
    join male_customers mc on t.customer_id = mc.customer_id
  where
    t.transaction_date >= '2020-09-01'
    and t.transaction_date <= '2020-09-30'
  group by
    t.transaction_id
)
select
  round(avg(transaction_value), 2) as avg_transaction_value
from
  transaction_totals;
```

The SQL agent's answer is:

```
The average transaction value in September 2020 for male customers with a credit score above 0.5 was $88.49.
```

The SQL and results were again cross-checked, and the run was inspected in LangSmith to confirm that the agent used a correct approach.

These two examples demonstrate that not only is the agent writing plausible SQL, but is reasoning correctly about joins, aggregations, and other concepts, all while guided by the system prompt and tools.

___

# Growth and Next Steps <a name="growth-next-steps"></a>

Potential future enhancements could include:

* Exposing additional tables for querying (for example, campaign data)
* Adding a lightweight UI so that non-technical users can ask questions without having to use SQL
* Logging queries and responses for audit and training
* Adding automated evaluations to check query correctness
* Allowing the agent to ask clarification questions in more complex scenarios

This project provides a foundation for an AI-powered, "self-serve" analytics layer on top of an SQL database, with safety and correctness built in.

___
